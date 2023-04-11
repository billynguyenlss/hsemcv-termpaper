import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

import timm
from .gus import *



def create_hsemotion_model(_PRETRAINED_HSEMOTION):
    hsemotion_model=torch.load(_PRETRAINED_HSEMOTION)
    hsemotion_model.classifier=torch.nn.Identity()

    for param in hsemotion_model.parameters():
        param.requires_grad=False

    hsemotion_model.classifier=nn.Sequential(
        nn.Linear(in_features=1280,out_features=512),
        nn.Linear(in_features=512,out_features=6)
    )
    return hsemotion_model


class HSEmotionEfficientNetFeature(nn.Module):
    def __init__(self, num_classes = 6, drop_rate = 0.2, _out_features=512, fc_in_dim=512,
                 pretrained = True, pretrained_ckp=None):
        super(HSEmotionEfficientNetFeature, self).__init__()
        self.drop_rate = drop_rate
        self.threshold=0.5
        if pretrained:
            self.features=torch.load(pretrained_ckp)
            self.features.classifier=torch.nn.Identity()
            
            for param in self.features.parameters():
                param.requires_grad = False

            # self.features=nn.Sequential(
            #     *list(self.features.children())[:-2],
            #     nn.Conv2d(1280,_out_features,(1,1))
            # )
            self.features.classifier=nn.Sequential(nn.Linear(in_features=1280, out_features=_out_features))   
        else:
            self.features=timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
            self.features.classifier=torch.nn.Identity()
            self.features.load_state_dict(torch.load('../face-emotion-recognition/models/pretrained_faces/state_vggface2_enet0_new.pt'))
            for param in self.features.parameters():
                param.requires_grad = False

            # _out_features= 512 # 256 (not really good) #1024 (not good f1<0.7)  # 512 (f1=0.75)
            # fc_in_dim =  512 # 256 #1024 # 512
        
            self.features.classifier=nn.Sequential(nn.Linear(in_features=1280, out_features=_out_features))         
   
        self.fc_6 = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())
        self.gus1 = GUS(fc_in_dim, fc_in_dim)
        self.gus2 = GUS(fc_in_dim, fc_in_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.t = 0.5
        self.BN = nn.BatchNorm1d(fc_in_dim)


    # def forward(self, x, phase='val', threshold=0.5):
    def forward(self, x):
        x = self.features(x)
        # if self.drop_rate > 0 and phase=="train":
        if self.training:
            phase='train'
        else:
            phase='eval'
        if self.drop_rate > 0 and self.training:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        
        cs1 = cosine_similarity(x.cpu().detach())
        x = self.gus1(x, cs1, phase, self.threshold)
        x = self.relu(x)

        cs2 = cosine_similarity(x.cpu().detach())
        x = self.gus2(x, cs2, phase, self.threshold)

        out = self.fc_6(x)
        if self.training:
            return out, x
        else:
            return out