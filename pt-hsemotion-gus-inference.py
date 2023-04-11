import psutil
base_ram = psutil.virtual_memory()[3]/1e9
print("RAM used before starting task (GB): {:.2f}".format(base_ram))

import torch
from src.gus import *
from src.hsemotion_gus import *
import os
import time
from src.utils import count_parameters, count_trainable_parameters

ram_after_import_dependencies = psutil.virtual_memory()[3]/1e9
print("RAM used after import dependencies (GB) - total: {:.2f}, increased {:.2f}".format(ram_after_import_dependencies, ram_after_import_dependencies - base_ram))


CWD = os.getcwd()
PRETRAINED_MODEL_DIR = os.path.join(CWD, 'pretrained/pt')
PTH = os.path.join(PRETRAINED_MODEL_DIR, 'hsemotion-gus_enet_b0_8_va_mtl.pt_epoch_4_acc0.7974_f1-0.6918.pth')
_PRETRAINED_HSEMOTION_MODEL_NAME = 'enet_b0_8_va_mtl.pt'
PRETRAINED_PT = './face-emotion-recognition/models/affectnet_emotions/' + _PRETRAINED_HSEMOTION_MODEL_NAME

# load model
model = HSEmotionEfficientNetFeature(num_classes = 6, drop_rate = 0.2,
                                     _out_features=512, fc_in_dim=512,
                                        pretrained = True, pretrained_ckp=PRETRAINED_PT)
model.load_state_dict(torch.load(PTH)['model_state_dict'])
print('number of parameters: {:.2f} million'.format(count_parameters(model)))

ram_after_loading_model = psutil.virtual_memory()[3]/1e9
print("RAM used after loading model (GB) - total: {:.2f}, increased: {:.2f}GB".format(ram_after_loading_model, ram_after_loading_model - ram_after_import_dependencies))
                

# dummy inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device).eval()

dummy_input = torch.randn((1,3,224,224), dtype=torch.float32, device=device)

K = 100
start = time.time()
for i in range(K):
    out = model(dummy_input)
end = time.time()
et = end - start

ram_after_inference = psutil.virtual_memory()[3]/1e9
print("RAM used after performing inference (GB) - total: {:.2f}, increased: {:.2f}".format(ram_after_inference, ram_after_inference - ram_after_loading_model))


print('hsemotion-gus: avg time: {:.3f} sec, fps: {:.2f}\n'.format(et/100, 100/et))

# from to_pytorch import quantized_model_for_mobile, quantize_model_int8
# quantized_model_for_mobile(model, 'hsemotion-gus', 'pretrained/pt-quantized-for-mobile')
# quantize_model_int8(model, 'hsemotion-gus', 'pretrained/pt-quantized')