import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from evaluate import  calculate_top_map
from load_dataset import  load_dataset
from metric import ContrastiveLoss
from model import FuseTransEncoder, ImageMlp, TextMlp
from os import path as osp
from utils import load_checkpoints, save_checkpoints
from torch.optim import lr_scheduler
import time

class Solver(object):
    def __init__(self, config):
        self.batch_size = 128  
        self.total_epoch = config.epoch
        self.dataset  = config.dataset
        self.model_dir = "./checkpoints"

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device(config.device if USE_CUDA else "cpu")

        self.task = config.task
        self.feat_lens = 512
        self.nbits = config.hash_lens
        num_layers, self.token_size, nhead = 2, 1024, 4
 
        self.FuseTrans = FuseTransEncoder(num_layers, self.token_size, nhead).to(self.device)
        self.ImageMlp = ImageMlp(self.feat_lens, self.nbits).to(self.device)
        self.TextMlp = TextMlp(self.feat_lens, self.nbits).to(self.device)
        
        paramsFuse_to_update = list(self.FuseTrans.parameters()) 
        paramsImage = list(self.ImageMlp.parameters()) 
        paramsText = list(self.TextMlp.parameters()) 
        
        total_param = sum([param.nelement() for param in paramsFuse_to_update])+sum([param.nelement() for param in paramsImage])+sum([param.nelement() for param in paramsText])
        print("total_param:",total_param)
        self.optimizer_FuseTrans = optim.Adam(paramsFuse_to_update, lr=1e-4, betas=(0.5, 0.999))
        self.optimizer_ImageMlp = optim.Adam(paramsImage, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_TextMlp = optim.Adam(paramsText, lr=1e-3, betas=(0.5, 0.999))

        if self.dataset == "mirflickr" or self.dataset=="nus-wide":
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp,milestones=[30,80], gamma=1.2)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp,milestones=[30,80], gamma=1.2)
        elif self.dataset == "mscoco":
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp,milestones=[200], gamma=0.6)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp,milestones=[200], gamma=0.6)

        data_loader = load_dataset(self.dataset, self.batch_size)
        self.train_loader = data_loader['train']
        self.query_loader = data_loader['query']
        self.retrieval_loader = data_loader['retrieval']
              
        self.ContrastiveLoss = ContrastiveLoss(batch_size=self.batch_size, device=self.device)
     
     
    def train(self):
        if self.task == 0: # train real
            print("Training Fusion Transformer...")
            for epoch in range(self.total_epoch):
                print("epoch:",epoch+1)
                train_loss = self.trainfusion()
                if((epoch+1)%10==0):
                    print("Testing...")
                    img2text, text2img = self.evaluate() 
                    print('I2T:',img2text, ', T2I:',text2img)
            save_checkpoints(self)
           
        elif self.task == 1: # train hash 
            print("Training Hash Fuction...")
            I2T_MAP = []
            T2I_MAP = []
            start_time = time.time()
            for epoch in range(self.total_epoch):
                print("epoch:",epoch+1)
                train_loss = self.trainhash()
                print(train_loss)
                if((epoch+1)%10==0):
                    print("Testing...")
                    img2text, text2img = self.evaluate() 
                    I2T_MAP.append(img2text)
                    T2I_MAP.append(text2img)
                    print('I2T:',img2text, ', T2I:',text2img)
            print(I2T_MAP,T2I_MAP)
            save_checkpoints(self)
            time_elapsed = time.time() - start_time
            print(f'Total Train Time: {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
                
        elif self.task == 2: # test real
            file_name = self.dataset + '_fusion.pth'
            ckp_path = osp.join(self.model_dir,'real', file_name)
            load_checkpoints(self, ckp_path)

        elif self.task == 3: # test hash 
            
            file_name = self.dataset + '_hash_' + str(self.nbits)+".pth"
            ckp_path = osp.join(self.model_dir,'hash', file_name)
            load_checkpoints(self, ckp_path)

        print("Final Testing...")
        img2text, text2img = self.evaluate() 
        print('I2T:',img2text, ', T2I:',text2img)
        return (img2text + text2img)/2., img2text, text2img
      
    def evaluate(self):
        self.FuseTrans.eval()
        self.ImageMlp.eval()
        self.TextMlp.eval()
        qu_BI, qu_BT, qu_L = [], [], []
        re_BI, re_BT, re_L = [], [], []
      
        with torch.no_grad():
            for _,(data_I, data_T, data_L,_) in enumerate(self.query_loader):
                data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                temp_tokens = torch.concat((data_I, data_T), dim = 1)
                img_query ,txt_query =  self.FuseTrans(temp_tokens)
                if self.task == 1 or self.task == 3:
                    img_query = self.ImageMlp(img_query)
                    txt_query = self.TextMlp(txt_query)
                img_query, txt_query = img_query.cpu().numpy(), txt_query.cpu().numpy()
                qu_BI.extend(img_query)
                qu_BT.extend(txt_query)
                qu_L.extend(data_L.cpu().numpy())  

            for _,(data_I, data_T, data_L,_) in enumerate(self.retrieval_loader):
                data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                temp_tokens = torch.concat((data_I, data_T), dim = 1)
                img_retrieval ,txt_retrieval =  self.FuseTrans(temp_tokens)
                if self.task ==1 or self.task ==3:
                    img_retrieval = self.ImageMlp(img_retrieval)
                    txt_retrieval = self.TextMlp(txt_retrieval)
                img_retrieval, txt_retrieval = img_retrieval.cpu().numpy(), txt_retrieval.cpu().numpy()
                re_BI.extend(img_retrieval)
                re_BT.extend(txt_retrieval)
                re_L.extend(data_L.cpu().numpy())
        
        re_BI = np.array(re_BI)
        re_BT = np.array(re_BT)
        re_L = np.array(re_L)

        qu_BI = np.array(qu_BI)
        qu_BT = np.array(qu_BT)
        qu_L = np.array(qu_L)

        if self.task ==1 or self.task ==3:   # hashing
            qu_BI = torch.sign(torch.tensor(qu_BI)).cpu().numpy()
            qu_BT = torch.sign(torch.tensor(qu_BT)).cpu().numpy()
            re_BT = torch.sign(torch.tensor(re_BT)).cpu().numpy()
            re_BI = torch.sign(torch.tensor(re_BI)).cpu().numpy()
        elif self.task ==0 or self.task ==2:  # real value
            qu_BI = torch.tensor(qu_BI).cpu().numpy()
            qu_BT = torch.tensor(qu_BT).cpu().numpy()
            re_BT = torch.tensor(re_BT).cpu().numpy()
            re_BI = torch.tensor(re_BI).cpu().numpy()
        
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        return MAP_I2T, MAP_T2I 
    
    def trainfusion(self):
        self.FuseTrans.train()
        running_loss = 0.0
        for idx, (img, txt, _,_) in enumerate(self.train_loader):
            temp_tokens = torch.concat((img, txt), dim = 1).to(self.device)
            temp_tokens = temp_tokens.unsqueeze(0)
            img_embedding, text_embedding = self.FuseTrans(temp_tokens)
            loss = self.ContrastiveLoss(img_embedding, text_embedding)
            self.optimizer_FuseTrans.zero_grad()
            loss.backward()
            self.optimizer_FuseTrans.step()
            running_loss += loss.item()
        return running_loss
    
    def trainhash(self):
        self.FuseTrans.train()
        self.ImageMlp.train()
        self.TextMlp.train()
        running_loss = 0.0
        for idx, (img, txt, _,_) in enumerate(self.train_loader):
            img, txt = img.to(self.device), txt.to(self.device)
            temp_tokens = torch.concat((img, txt), dim = 1)
            temp_tokens = temp_tokens.unsqueeze(0)
            img_embedding, text_embedding = self.FuseTrans(temp_tokens)
            loss1 = self.ContrastiveLoss(img_embedding, text_embedding)

            img_embedding = self.ImageMlp(img_embedding)
            text_embedding = self.TextMlp(text_embedding)
            loss2 = self.ContrastiveLoss(img_embedding, text_embedding)

            loss = loss1  + loss2*0.5
            self.optimizer_FuseTrans.zero_grad()
            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()
            loss.backward()
            self.optimizer_FuseTrans.step()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()
            running_loss += loss.item()
        
            self.ImageMlp_scheduler.step()
            self.TextMlp_scheduler.step()
        return running_loss