import torch
from os import path as osp

def save_checkpoints(self):
    if self.task ==0:
        file_name = self.dataset + '_fusion.pth'
        ckp_path = osp.join(self.model_dir, 'real', file_name)
        obj = {
        'FusionTransformer': self.FuseTrans.state_dict()
        }
    if self.task ==1:
        file_name = self.dataset + '_hash_' + str(self.nbits)+".pth"
        ckp_path = osp.join(self.model_dir, 'hash', file_name)
        obj = {
        'FusionTransformer': self.FuseTrans.state_dict(),
        'ImageMlp': self.ImageMlp.state_dict(),
        'TextMlp': self.TextMlp.state_dict()
        }
    torch.save(obj, ckp_path)
    print('**********Save the {0} model successfully.**********'.format("real" if self.task==0 else "hash"))


def load_checkpoints(self, file_name):
    ckp_path = file_name
    try:
        obj = torch.load(ckp_path, map_location= self.device)
        print('**************** Load checkpoint %s ****************' % ckp_path)
    except IOError:
        print('********** Fail to load checkpoint %s!*********' % ckp_path)
        raise IOError
    if self.task==2:
        self.FuseTrans.load_state_dict(obj['FusionTransformer'])
    elif self.task==3:
        self.FuseTrans.load_state_dict(obj['FusionTransformer'])
        self.ImageMlp.load_state_dict(obj['ImageMlp'])
        self.TextMlp.load_state_dict(obj['TextMlp'])