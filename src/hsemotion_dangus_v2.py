import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torchvision.models as tv_models
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import timm
from .dan import *
from .gus import *


class HSEmotionEffNetDANGUS_V2(nn.Module):
    def __init__(self, num_class=6,num_head=4, #pretrained=False, pretrained_ckp=None, 
                 drop_rate = 0.2, _out_features=512, fc_in_dim=512,pretrained=None):
        super(HSEmotionEffNetDANGUS_V2, self).__init__()
        self.threshold = 0.5
        # feature extractor
        self.drop_rate = drop_rate

        self.features=torch.load(pretrained)
        self.features.classifier=torch.nn.Identity()
        
        for param in self.features.parameters():
            param.requires_grad = False

        self.features=nn.Sequential(
            *list(self.features.children())[:-2],
            nn.Conv2d(1280,_out_features,(1,1))
        )

        # dan feature extractor
        # self.dan_fe = nn.Conv2d(1280,_out_features,(1,1))

        # gus feature extractor
        self.features_gus=torch.load(pretrained)
        self.features_gus.classifier=torch.nn.Identity()
        
        for param in self.features_gus.parameters():
            param.requires_grad = False
        self.features_gus.classifier=nn.Sequential(nn.Linear(in_features=1280, out_features=_out_features))                 
        
        # Attention head
        self.num_head = num_head
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead(s1=_out_features,s2=256,c1=_out_features,c2=32))
        self.sig = nn.Sigmoid()
        
        # Detection head: DAN
        _output_size = _out_features
        self.fc = nn.Linear(_output_size, num_class)
        self.bn = nn.BatchNorm1d(num_class)

        # Detection head: GUS
        self.fc_6 = nn.Linear(fc_in_dim, num_class) # new fc layer 512x7
        self.gus1 = GUS(fc_in_dim, fc_in_dim)
        self.gus2 = GUS(fc_in_dim, fc_in_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.t = 0.5

        # Detection head: ensemble
        self.fc_e = nn.Sequential(
            nn.Linear(num_class*2, num_class),
            nn.BatchNorm1d(num_class)
        )


    def forward(self, x):
        # GUS feature extraction
        z = self.features_gus(x)
        if self.training:
            phase='train'
        else:
            phase='eval'

        if self.drop_rate > 0 and phase=="train":
            z =  nn.Dropout(self.drop_rate)(z)
            z = z.view(z.size(0), -1)
        else:
            z = z.view(z.size(0), -1)
        
        cs1 = cosine_similarity(z.cpu().detach())
        z = self.gus1(z, cs1, phase, self.threshold)
        z = self.relu(z)

        cs2 = cosine_similarity(z.cpu().detach())
        z = self.gus2(z, cs2, phase, self.threshold)

        out_gus = self.fc_6(z)

        # DAN forward
        x = self.features(x)

        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        
        heads = torch.stack(heads).permute([1,0,2])
        if heads.size(1)>1:
            heads = F.log_softmax(heads,dim=1)
            
        out_dan = self.fc(heads.sum(dim=1))
        out_dan = self.bn(out_dan)

        # print('shape out_gus:', out_gus.shape, 'out_dan:',out_dan.shape)
        out = torch.cat((out_gus, out_dan),dim=1)
        # print('out shape:', out.shape)
        out = self.fc_e(out)

        if self.training:
            return out_gus, z, out_dan, x, heads, out 
        else:
            return out