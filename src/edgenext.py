import torch
from torch import nn, optim

import torchvision
from torchvision.models.mobilenetv3 import MobileNetV3
import torchvision.models as tv_models
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys
# sys.path.append('../')
sys.path.insert(0, './EdgeNeXt/models')
from EdgeNeXt.models import model as EdgeNeXT_models

def create_edgeNext_x_small(pretrained_ckp='EdgeNeXt/pretrained/edgenext_x_small.pth'):
    if '_x_small' in pretrained_ckp:
        edgeNext_small = EdgeNeXT_models.edgenext_x_small(classifier_dropout=0.2)
    elif 'edgenext_small' in pretrained_ckp:
        edgeNext_small = EdgeNeXT_models.edgenext_small(classifier_dropout=0.2)
    else:
        raise Exception("Probably the pretrained model was not declared in this function. Please add other definition to this function.")
    ckp = torch.load(pretrained_ckp)
    # print(ckp['model'])
    edgeNext_small.load_state_dict(ckp['model'])
    for params in edgeNext_small.parameters():
        params.requires_grad = False
        
    model = torch.nn.Sequential(
        edgeNext_small,
        nn.Linear(1000,128),
        nn.Linear(128,2)
    )
    return model