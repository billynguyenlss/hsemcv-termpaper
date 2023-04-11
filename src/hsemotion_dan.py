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


class HSEmotionEffNetDAN(nn.Module):
    def __init__(self, num_class=6,num_head=4, pretrained=False, pretrained_ckp=None,):
        super(HSEmotionEffNetDAN, self).__init__()
        
        # self.features=timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
        # self.features.classifier=torch.nn.Identity()
        # self.features.load_state_dict(torch.load('../face-emotion-recognition/models/pretrained_faces/state_vggface2_enet0_new.pt'))

        _output_size = 512
        _out_features = 512
        # for param in self.features.parameters():
        #     param.requires_grad = False

        # self.features=nn.Sequential(
        #     *list(self.features.children())[:-2],
        #     nn.Conv2d(1280,512,(1,1))
        # )

        if pretrained:
            self.features=torch.load(pretrained_ckp)
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

        self.fc = nn.Linear(_output_size, num_class)
        self.bn = nn.BatchNorm1d(num_class)


    def forward(self, x):
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        
        heads = torch.stack(heads).permute([1,0,2])
        if heads.size(1)>1:
            heads = F.log_softmax(heads,dim=1)
            
        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)

        if self.training:
            return out, x, heads
        else:
            return out