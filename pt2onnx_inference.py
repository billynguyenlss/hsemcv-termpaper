import psutil
base_ram = psutil.virtual_memory()[3]/1e9
print("RAM used before starting task (GB): {:.2f}".format(base_ram))

import numpy as np
# import onnx
import time
import os
import argparse

ram_after_import_dependencies = psutil.virtual_memory()[3]/1e9
print("RAM used after import dependencies (GB) - total: {:.2f}, increased {:.2f}".format(ram_after_import_dependencies, ram_after_import_dependencies - base_ram))

def main(args):
    runtime = 'onnx' if args.runtime==1 else 'cv2'
    ets = {}
    onnx_pt_root = args.onnx_root
    if args.model is not None:
        models = [args.model]
    else:
        models = os.listdir(onnx_pt_root)

    providers = []
    if args.trtexecutionprovider:
        providers.append('TensorrtExecutionProvider')
    if args.cudaexecutionprovider:
        providers.append('CUDAExecutionProvider')
    if args.cpuexecutionprovider:
        providers.append('CPUExecutionProvider')

    for onnx_file in models:
        if '.onnx' in onnx_file:
            print(f'\n\n\nstart inference for model {onnx_file}')
            try:
                ets[onnx_file] = {}
                ets[onnx_file]['providers'] = providers
                ets[onnx_file]['runtime'] = runtime
                ets[onnx_file]['cv2_backend_cuda'] = args.cv2_backend_cuda

                # onnx_model=onnx.load(onnx_file)
                # onnx.checker.check_model(onnx_model)
                # print(onnx.helper.printable_graph(onnx_model.graph))
                # print('completed checked')

                dummy_input = np.random.rand(args.batch_size,3,224,224)

                if args.runtime==1:
                    import onnxruntime
                    ort_session = onnxruntime.InferenceSession(os.path.join(onnx_pt_root,  onnx_file),  providers=providers)
                elif args.runtime==2:
                    import cv2
                    cv_model = cv2.dnn.readNetFromONNX(os.path.join(onnx_pt_root, onnx_file))
                    if args.cv2_backend_cuda:
                        cv_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        cv_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

                ram_after_loading_model = psutil.virtual_memory()[3]/1e9
                print("RAM used after loading model (GB) - total: {:.2f}, increased: {:.2f}GB".format(ram_after_loading_model, ram_after_loading_model - ram_after_import_dependencies))
                
                start = time.time()
                if args.runtime==1:
                    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.astype(np.float32)}
                    ort_outs = ort_session.run(None, ort_inputs)
                end = time.time()
                print('warning up elapsed time: {:.3f}'.format(end - start))
                ram_after_warming_up = psutil.virtual_memory()[3]/1e9
                print("RAM used after warming up (GB) - total: {:.2f}, increased: {:.2f}".format(ram_after_warming_up, ram_after_warming_up - ram_after_loading_model))
                
                # compute ONNX Runtime output prediction
                start = time.time()
                for i in range(args.k):
                    if args.runtime==1:
                        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.astype(np.float32)}
                        ort_outs = ort_session.run(None, ort_inputs)
                    elif args.runtime==2:
                        cv_model.setInput(dummy_input.astype(np.float32))
                        _ = cv_model.forward()

                end = time.time()
                et = end - start
                ets[onnx_file]['et'] = et/args.k
                
                ram_after_inference = psutil.virtual_memory()[3]/1e9
                print("RAM used after performing inference (GB) - total: {:.2f}, increased: {:.2f}".format(ram_after_inference, ram_after_inference - ram_after_warming_up))
            except Exception as error:
                print('Something went wrong!', error)

    print('-------------Summary results-------------')
    for k, v in ets.items():
        print('model name:', k)
        print('runtime:', v['runtime'])
        if v['runtime'] == 'onnx':
            print('onnxruntime providers:', v['providers'])
        elif v['runtime'] == 'cv2':
            print('cv2_backend_cuda:', v['cv2_backend_cuda'])
        try:
            print('average elapsed time per input (sec): {:.3f}'.format(v['et']))
            print('fps: {:.2f}'.format(1/v['et']))
            print('\n')
            with open('records.txt', 'a') as f:
                f.write(f"{k},{v['runtime']},{v['providers']},{args.batch_size},{str(round(v['et'],4))},{str(round(1/v['et'],4))}\n")
        except:
            print('Something went wrong! Possibly failed to load model to cv2.dnn runtime!')

        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters for onnx inference")
    parser.add_argument('--trtexecutionprovider', type=int, default=1, help="True to add TRTExecutionProvider")
    parser.add_argument('--cudaexecutionprovider', type=int, default=1, help="True to add CUDAExecutionProvider")
    parser.add_argument('--cpuexecutionprovider', type=int, default=0, help="True to add CPUExecutionProvider")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--runtime', type=int, default=1, help="specify the runtime: 1 - onnxruntime, 2 - cv2.dnn.readNetFromONNX")
    parser.add_argument('--cv2_backend_cuda', type=int, default=1, help="specify the backend for cv2.dnn runtime")
    parser.add_argument('--k', type=int, default=100, help="number of repeat inference")
    parser.add_argument('--batch_size', type=int, default=1, help="batch_size of dummy input")
    parser.add_argument('--onnx_root', type=str, default="./pretrained/onnx", help="batch_size of dummy input")

    args = parser.parse_args()
    print(args)
    
    main(args)
