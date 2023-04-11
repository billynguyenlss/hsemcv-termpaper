import sys,os
import torch
from torchvision import transforms,models
from torch.utils.mobile_optimizer import optimize_for_mobile

from PIL import Image
import timm

def quantize_model_int8(model, model_name, output_model_dir,device='cuda'):
    example = torch.randn((1,3,224,224),device=device)
    model.eval()

    model_int8 = torch.ao.quantization.quantize_dynamic(
    model,  # the original model
    dtype=torch.qint8)  # the target dtype for quantized weights
    torch.save(model_int8, os.path.join(output_model_dir, model_name + '.pt'))
    traced_script_module = torch.jit.trace(model, example)
    torch.save(traced_script_module, os.path.join(output_model_dir, model_name + '_scripted.pt'))
    return model_int8



def quantized_model_for_mobile(model, model_name, output_model_dir,device='cuda'):
    example = torch.randn((1,3,224,224),device=device)
    model.eval()
    traced_script_module = torch.jit.trace(model, example)
    torch.save(traced_script_module, os.path.join(output_model_dir, model_name + '_scripted.pt'))

    #traced_script_module.save(filename+'.pt')
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(os.path.join(output_model_dir, model_name + '_for_mobile.ptl'))

    quantized_model = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)
    traced_script_module = torch.jit.trace(quantized_model, example)
    #traced_script_module.save(filename+'_quant.pt')
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(os.path.join(output_model_dir, model_name + '_quant.ptl'))
