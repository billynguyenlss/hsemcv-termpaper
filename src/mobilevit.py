import torch
from torch import nn, optim

import torchvision
from torchvision.models.mobilenetv3 import MobileNetV3
import torchvision.models as tv_models
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import AutoImageProcessor, MobileViTModel, MobileViTConfig, MobileViTFeatureExtractor
from transformers import get_scheduler

class CustomMobileViT(nn.Module):
    def __init__(self, pretrained_ckp='apple/mobilevit-xx-small', input_size=224):
        super(CustomMobileViT, self).__init__()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.feature_extractor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
        # image_processor(inputs)
#         configuration = MobileViTConfig()
        self.pretrained_ckp = pretrained_ckp
        self.pretrained_mbViT = MobileViTModel.from_pretrained(self.pretrained_ckp,num_labels=2)
        for params in self.pretrained_mbViT.parameters():
            params.requires_grad=False
        embedding_size = self.pretrained_mbViT(torch.randn((1,3,input_size,input_size), dtype=torch.float32), return_dict=False)[1].shape[-1]  
        print('embedding size:', embedding_size)
        self.classifier = torch.nn.Sequential(
            nn.Linear(embedding_size,128),
            nn.Linear(128,2)
        )
        
    def forward(self, x):
#         inputs = torch.Tensor(image_processor(x)['pixel_values']).to(torch.float32).to(self.device)
        hidden = self.pretrained_mbViT(x, return_dict=False)
        out = self.classifier(hidden[1])
        return out
    
    