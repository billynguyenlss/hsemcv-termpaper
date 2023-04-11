import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torchvision.models as tv_models
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


class DAN2(nn.Module):
    def __init__(self, num_class=7,num_head=4, pretrained=False, pretrained_ckp=None):
        super(DAN2, self).__init__()
        
        resnet = tv_models.resnet18(pretrained)
        
        if pretrained:
            # checkpoint = torch.load('./models/resnet18_msceleb.pth')
            checkpoint = torch.load(pretrained_ckp)
            resnet.load_state_dict(checkpoint['model_state_dict'],strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
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
        

class CrossAttentionHead(nn.Module):
    def __init__(self, s1=512,s2=256,c1=512,c2=32):
        super().__init__()
        self.s1=s1
        self.s2=s2
        self.c1=c1
        self.c2=c2
        self.sa = SpatialAttention(s1,s2)
        self.ca = ChannelAttention(c1,c2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca


class SpatialAttention(nn.Module):

    def __init__(self, f1=512, f2=256):
        super().__init__()
        self.f1=512
        self.f2=256
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(self.f1, self.f2, kernel_size=1),
            nn.BatchNorm2d(self.f2),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(self.f2, self.f1, kernel_size=3,padding=1),
            nn.BatchNorm2d(self.f1),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(self.f2, self.f1, kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(self.f1),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(self.f2, self.f1, kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(self.f1),
        )
        self.relu = nn.ReLU()


    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1,keepdim=True) 
        out = x*y
        
        return out 


class ChannelAttention(nn.Module):

    def __init__(self,f1=512, f2=32):
        super().__init__()
        self.f1 = f1
        self.f2 = f2
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(self.f1, self.f2),
            nn.BatchNorm1d(self.f2),
            nn.ReLU(inplace=True),
            nn.Linear(self.f2, self.f1),
            nn.Sigmoid()    
        )


    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0),-1)
        y = self.attention(sa)
        out = sa * y
        
        return out
    

class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class PartitionLoss(nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            ## add eps to avoid empty var case
            loss = torch.log(1+num_head/(var+eps))
        else:
            loss = 0
            
        return loss


def inference_batch_DAN(model, dataloader, dtype=torch.float32, device='cuda'):

    with torch.no_grad():
        model.eval()
        val_epoch_labels = []
        val_epoch_preds = []


        for imgs, targets in tqdm(dataloader):
            imgs = imgs.to(dtype).to(device)
            targets = targets.to(dtype).to(device)

            out,feat,heads = model(imgs)
            _, predicts = torch.max(out, 1)

            val_epoch_preds.append(predicts.detach().cpu().numpy())
            val_epoch_labels.append(targets.detach().cpu().numpy())
            
        val_epoch_f1 = f1_score(np.concatenate(val_epoch_labels), 
                                np.concatenate(val_epoch_preds), 
                                average='macro')
        val_epoch_f1_by_class = f1_score(np.concatenate(val_epoch_labels), 
                                         np.concatenate(val_epoch_preds), 
                                         average=None)

        acc = accuracy_score(np.concatenate(val_epoch_labels), 
                                np.concatenate(val_epoch_preds))

        return acc, val_epoch_f1, val_epoch_f1_by_class