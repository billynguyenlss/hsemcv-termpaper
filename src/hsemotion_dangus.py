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


class HSEmotionEffNetDANGUS(nn.Module):
    def __init__(self, num_class=6,num_head=4, #pretrained=False, pretrained_ckp=None, 
                 drop_rate = 0.2, _out_features=512, fc_in_dim=512,pretrained=None):
        super(HSEmotionEffNetDANGUS, self).__init__()
        self.threshold = 0.5
        # feature extractor
        self.drop_rate = drop_rate
        if pretrained:
            self.features=torch.load(pretrained)
            self.features.classifier=torch.nn.Identity()
            
            for param in self.features.parameters():
                param.requires_grad = False

            self.features=nn.Sequential(
                *list(self.features.children())[:-2],
                nn.Conv2d(1280,_out_features,(1,1))
            )
        else:
            self.features=timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
            self.features.classifier=torch.nn.Identity()
            self.features.load_state_dict(torch.load('../face-emotion-recognition/models/pretrained_faces/state_vggface2_enet0_new.pt'))
            for param in self.features.parameters():
                param.requires_grad = False
            
            self.features.classifier=nn.Sequential(nn.Linear(in_features=1280, out_features=_out_features))              
        
        # Attention head
        self.num_head = num_head
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead(s1=_out_features,s2=256,c1=_out_features,c2=32))
        self.sig = nn.Sigmoid()

        # Detection head
        # self.fc = nn.Linear(_output_size, num_class)
        # self.bn = nn.BatchNorm1d(num_class)

        self.fc_6 = nn.Linear(fc_in_dim, num_class) # new fc layer 512x7
        # self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())
        self.gus1 = GUS(fc_in_dim, fc_in_dim)
        self.gus2 = GUS(fc_in_dim, fc_in_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.t = 0.5
        # self.BN = nn.BatchNorm1d(fc_in_dim)


    def forward(self, x):
        # feature extraction
        x = self.features(x)
        if self.training:
            phase='train'
        else:
            phase='val'
        # attention head
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        
        heads = torch.stack(heads).permute([1,0,2])
        if heads.size(1)>1:
            heads = F.log_softmax(heads,dim=1)
        
        # detection head
        x_gus = heads.sum(dim=1)
        cs1 = cosine_similarity(x_gus.cpu().detach())
        x_gus = self.gus1(x_gus, cs1, phase, self.threshold)
        x_gus = self.relu(x_gus)

        cs2 = cosine_similarity(x_gus.cpu().detach())
        x_gus = self.gus2(x_gus, cs2, phase, self.threshold)

        out = self.fc_6(x_gus)
        
        # out = self.fc(heads.sum(dim=1))
        # out = self.bn(out)
        if self.training:
            return heads, x, x_gus, out
        else:
            return out