import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def check_images( s_dir, ext_list):
    bad_images=[]
    bad_ext=[]
    s_list= os.listdir(s_dir)
    for klass in s_list:
        klass_path=os.path.join (s_dir, klass)
        print ('processing class directory ', klass)
        if os.path.isdir(klass_path):
            file_list=os.listdir(klass_path)
            for f in file_list:               
                f_path=os.path.join (klass_path,f)
                index=f.rfind('.')
                ext=f[index+1:].lower()
                if ext not in ext_list:
                    print('file ', f_path, ' has an invalid extension ', ext)
                    bad_ext.append(f_path)
                if os.path.isfile(f_path):
                    try:
                        img=cv2.imread(f_path)
                        shape=img.shape
                    except:
                        print('file ', f_path, ' is not a valid image file')
                        bad_images.append(f_path)
                else:
                    print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
        else:
            print ('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
    return bad_images, bad_ext


def check_number_of_images_in_directory(img_path):
    _count = 0
    for dirname, _, filenames in os.walk(img_path):
        for filename in filenames:
            _count += 1
    return _count    


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def unzip_tarfile(fname):
    import tarfile

    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall()
        tar.close()

def get_output_shape(model, input_shape=224, device='cuda'):
    dummy_input = torch.randn((1,3, input_shape, input_shape), dtype=torch.float32, device=device)
    out = model.to(device)(dummy_input)
    output_size = out.shape[-1]
    return output_size

def test_model(model, input_shape=224, device='cuda'):
    dummy_input = torch.randn((1,3, input_shape, input_shape), dtype=torch.float32, device=device)
    out = model(dummy_input)
    return out

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())/1e6