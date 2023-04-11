import torch
from torch import nn, optim

import torchvision
from torchvision.models.mobilenetv3 import MobileNetV3
import torchvision.models as tv_models
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def create_mobilenetv3_model(isSmall=True):
    if isSmall:
        pretrained_model = tv_models.mobilenetv3.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        pretrained_model = tv_models.mobilenetv3.mobilenet_v3_large(weights=tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
#     pretrained_model.classifier=torch.nn.Identity()
    for param in pretrained_model.parameters():
        param.requires_grad=False
    
    layer = list(pretrained_model.classifier.children())[0]
    if hasattr(layer, 'in_features'):
#         print(layer.in_features)
        hidden_dim = layer.in_features
    else:
        raise Exception("Incorrect embedding dim for classifier.")
    pretrained_model.classifier=torch.nn.Sequential(
        torch.nn.Linear(in_features=hidden_dim,out_features=256),
        torch.nn.Linear(in_features=256,out_features=2)
    )
    return pretrained_model

