import torch
import torchvision.models as tv_models

def create_mobilenetv3_model(isSmall=True):
    if isSmall:
        pretrained_model = tv_models.mobilenetv3.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        pretrained_model = tv_models.mobilenetv3.mobilenet_v3_large(weights=tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    for param in pretrained_model.parameters():
        param.requires_grad=False
    
    layer = list(pretrained_model.classifier.children())[0]
    if hasattr(layer, 'in_features'):
        hidden_dim = layer.in_features
    else:
        raise Exception("Incorrect embedding dim for classifier.")
    pretrained_model.classifier=torch.nn.Sequential(
        torch.nn.Linear(in_features=hidden_dim,out_features=256),
        torch.nn.Linear(in_features=256,out_features=2)
    )
    return pretrained_model

def create_mobilenetv2_model(pretrained=True):
    pretrained_model = tv_models.mobilenet_v2(pretrained)
    for param in pretrained_model.parameters():
        param.requires_grad=False
    
    pretrained_model.classifier=torch.nn.Sequential(
        torch.nn.Linear(in_features=1280,out_features=256),
        torch.nn.Linear(in_features=256,out_features=2)
    )
    return pretrained_model