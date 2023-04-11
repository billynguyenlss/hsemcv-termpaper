# https://arxiv.org/pdf/2207.13235.pdf

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from torchvision import models

import cv2
import random
import math

def get_now_time():
    now =  time.localtime()
    now_time = time.strftime("%Y_%m_%d_%H_%M_%S", now)
    return now_time
    
def mean_f1(preds,targets):
    f1=[]
    temp_exp_pred = np.array(preds)
    temp_exp_target = np.array(targets)
    temp_exp_pred = torch.eye(6)[temp_exp_pred]
    temp_exp_target = torch.eye(6)[temp_exp_target]
    for i in range(0, 6):
        exp_pred = temp_exp_pred[:, i]
        exp_target = temp_exp_target[:, i]
        f1.append(f1_score(exp_pred, exp_target))
    print(f1)
    #logger.info(str(f1))
    return np.mean(f1)

def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def color2gray(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_img_3d = image_array.copy()
    gray_img_3d[:, :, 0] = gray
    gray_img_3d[:, :, 1] = gray
    gray_img_3d[:, :, 2] = gray
    return gray_img_3d


def data_augment(image, brightness):
    factor = 1.0 + random.uniform(-1.0*brightness, brightness)
    table = np.array([(i / 255.0) * factor * 255 for i in np.arange(0, 256)]).clip(0,255).astype(np.uint8)
    image = cv2.LUT(image, table)
    return 


class GUS(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GUS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalize_weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = 'cpu'

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.normalize_weight.size(1))
        self.normalize_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj


    def gen_A(self, cs, phase, threshold):
        self.cs = cs
        sample = 0
        sample = np.random.rand(1) 
        # sample = torch.rand(1)
        if phase != "train":
            sample = threshold
            # sample = torch.Tensor(threshold)
        cs[sample < cs] = 1
        cs[sample >= cs] = 0
        _adj = torch.from_numpy(cs).float().cuda()
        # _adj = torch.from_numpy(cs).float().to(self.device)
        _adj = _adj * 0.5/ (_adj.sum(0, keepdims=True) + 1e-6)
        _adj = _adj + torch.from_numpy(np.identity(_adj.shape[0], np.int32)).float().cuda()
        # _adj = _adj + torch.from_numpy(np.identity(_adj.shape[0], np.int)).float().to(self.device)
        return _adj


    def forward(self, input, cs, phase, threshold):
        support = torch.matmul(input, self.normalize_weight)
        adj_A = self.gen_adj(self.gen_A(cs, phase, threshold)).cuda()
        # support = torch.matmul(input, self.normalize_weight).to(self.device)
        # adj_A = self.gen_adj(self.gen_A(cs, phase, threshold)).to(self.device)
        self.temp_adjwithgrad = torch.Tensor(adj_A.shape[0], adj_A.shape[0]).requires_grad_().to(self.device)
        self.temp_adjwithgrad.data = adj_A
        output = torch.matmul(self.temp_adjwithgrad, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

               
class Res18Feature(nn.Module):
    def __init__(self, pretrained = True, num_classes = 6, drop_rate = 0.2):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        self.threshold=0.5
        resnet  = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
   
        self.fc_6 = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())
        self.gus1 = GUS(512, 512)
        self.gus2 = GUS(512, 512)
        self.relu = nn.LeakyReLU(0.2)
        self.t = 0.5
        self.BN = nn.BatchNorm1d(fc_in_dim)



    def forward(self, x):
        x = self.features(x)
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


class EfficientNetFeature(nn.Module):
    def __init__(self, pretrained = True, num_classes = 6, drop_rate = 0.2):
        super(EfficientNetFeature, self).__init__()
        self.threshold=0.5
        self.drop_rate = drop_rate
        net  = models.efficientnet_b2(weights='IMAGENET1K_V1')
        
        self.features = nn.Sequential(
            *list(net.children())[:-1],
            # nn.Linear(1408, 512)
        ) # after avgpool 512x1

        # fc_in_dim = list(net.children())[-1].in_features # original fc layer's in dimention 512
        fc_in_dim = 1408
   
        self.fc_6 = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())
        self.gus1 = GUS(fc_in_dim, fc_in_dim)
        self.gus2 = GUS(fc_in_dim, fc_in_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.t = 0.5
        self.BN = nn.BatchNorm1d(fc_in_dim)



    def forward(self, x):
        x = self.features(x)
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