import onnx
from onnxsim import simplify
import os
from tqdm.auto import tqdm

# ROOT = 'pretrained/onnx/'
# OUTPUT_DIR = 'pretrained/onnx-simplified'

ROOT = 'edges/onnx/'
OUTPUT_DIR = 'edges/onnx-simplified'

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

onnx_models = os.listdir(ROOT)
for i, onnx_model_name in enumerate(tqdm(onnx_models)):
    print('processing model: ', onnx_model_name)
    onnx_model_path = os.path.join(ROOT, onnx_model_name)
    onnx_model = onnx.load(onnx_model_path)
    model_simpified, check = simplify(onnx_model)
    output_filename = onnx_model_name.replace('.onnx', '_simplified.onnx')
    onnx.save(model_simpified, os.path.join(OUTPUT_DIR, output_filename))