from torch.utils.data.dataset import Dataset
import pickle
from torch.utils.data import DataLoader
import torch
from evaluate import calculate_top_map

class CustomDataSet(Dataset):
    def __init__(self, images, texts, labs):
        self.images = images
        self.texts = texts
        self.labs = labs
   
    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        lab = self.labs[index]
        return img, text, lab, index

    def __len__(self):
        count = len(self.texts)
        return count

def load_dataset(dataset, batch_size):
    '''
        load datasets : mirflickr, mscoco, nus-wide
    '''
    train_loc = '/home/summer/data/Dataset/'+dataset+'/train.pkl'
    query_loc = '/home/summer/data/Dataset/'+dataset+'/query.pkl'
    retrieval_loc = '/home/summer/data/Dataset/'+dataset+'/retrieval.pkl'


  
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = torch.tensor(data['label'],dtype=torch.int64)
        train_texts = torch.tensor(data['text'], dtype=torch.float32)
        train_images = torch.tensor(data['image'], dtype=torch.float32)
    
    with open(query_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        query_labels = torch.tensor(data['label'],dtype=torch.int64)
        query_texts = torch.tensor(data['text'], dtype=torch.float32)    
        query_images = torch.tensor(data['image'], dtype=torch.float32)
      
    with open(retrieval_loc,'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_lables = torch.tensor(data['label'],dtype=torch.int64)
        retrieval_texts = torch.tensor(data['text'], dtype=torch.float32) 
        retrieval_images = torch.tensor(data['image'], dtype=torch.float32)
        
    imgs = {'train': train_images[:4992], 'query': query_images, 'retrieval': retrieval_images}
    texts = {'train': train_texts[:4992],  'query': query_texts, 'retrieval': retrieval_texts}
    labs = {'train': train_labels[:4992], 'query': query_labels, 'retrieval': retrieval_lables}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labs=labs[x]) for x in ['train', 'query', 'retrieval']}
    shuffle = {'train': True, 'query': False, 'retrieval': False}
    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, drop_last=True, pin_memory=True,shuffle=shuffle[x], num_workers=4) for x in ['train', 'query', 'retrieval']}
    return  dataloader

if __name__ == '__main__':
    import numpy as np
    dataset = 'mscoco'
    data_loader= load_dataset(dataset, 128)
    query_loader = data_loader['query']
    retrieval_loader = data_loader['retrieval']

    qu_BI, qu_BT, qu_L = [], [], []
    re_BI, re_BT, re_L = [], [], []
      
    with torch.no_grad():
        for _,(data_I, data_T, data_L) in enumerate(query_loader):
            img_query, txt_query = data_I.cpu().numpy(), data_T.cpu().numpy()
            qu_BI.extend(img_query)
            qu_BT.extend(txt_query)
            qu_L.extend(data_L.cpu().numpy())  
            
        for _,(data_I, data_T, data_L) in enumerate(retrieval_loader):
            img_retrieval, txt_retrieval = data_I.cpu().numpy(), data_T.cpu().numpy()
            re_BI.extend(img_retrieval)
            re_BT.extend(txt_retrieval)
            re_L.extend(data_L.cpu().numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.array(re_L)

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.array(qu_L)

    print(re_BI.shape, re_BT.shape, re_L.shape)
    print(qu_BI.shape, qu_BT.shape, qu_L.shape)
   
    qu_BI = torch.sign(torch.tensor(qu_BI)).cpu().numpy()
    qu_BT = torch.sign(torch.tensor(qu_BT)).cpu().numpy()
    re_BT = torch.sign(torch.tensor(re_BT)).cpu().numpy()
    re_BI = torch.sign(torch.tensor(re_BI)).cpu().numpy()

    print(qu_BI.dtype)
    MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
    print(MAP_I2T)
    MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
    print(MAP_T2I)

       


    
  
        