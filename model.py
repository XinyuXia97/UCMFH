import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, Dropout, Linear
from torch.nn.functional import normalize
from torch.nn import functional as F

class FuseTransEncoder(nn.Module):
    def __init__(self,  num_layers, hidden_size, nhead):
        super(FuseTransEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        self.transformerEncoder = TransformerEncoder(encoder_layer= encoder_layer, num_layers= num_layers)
        self.d_model = hidden_size
        self.sigal_d = int(self.d_model/2)
       
    def forward(self, tokens):
        encoder_X = self.transformerEncoder(tokens)
        encoder_X_r = encoder_X.reshape( -1,self.d_model)
        encoder_X_r = normalize(encoder_X_r, p =2 ,dim =1)
        img, txt = encoder_X_r[:,:self.sigal_d], encoder_X_r[:,self.sigal_d:]
        return img, txt


class ImageMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024,128,1024], dropout=0.1):
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()
    def _ff_block(self, x):
        x = normalize(x, p =2 ,dim =1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out
        
    def forward(self, X):  
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p =2 ,dim =1)
        return mlp_output

class TextMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024,128,1024], dropout=0.1): 
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()
       
    def _ff_block(self, x):
        x = normalize(x, p =2 ,dim =1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out
        
    def forward(self, X):  
        mlp_output =  self._ff_block(X)
        mlp_output = normalize(mlp_output, p =2 ,dim =1)
        return mlp_output


