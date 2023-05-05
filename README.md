
# Set up environment
This project used python version 3.9 to be compatible with the python version of Colab at the time this project is experimented.

Please install all dependencies from `requirements.txt`.

Then, clone other related projects. There is some borrow code from those repository.

```
git submodule add https://github.com/Yeonggi-hong/ECCV2022-ABAW-competition.git
git submodule add https://github.com/HSE-asavchenko/face-emotion-recognition.git
```

The pretrained facial expression recognition models from part one are available [here](https://drive.google.com/file/d/1-A3D-QykIUWdvMNT-5Ls3xYwbR7-POxS/view?usp=sharing). Download and unzip in the the file inside this project directory. 

# Performance assessment

Run this code to experiment and compare the performance of models with pytorch format (*.pt):
```
chmod +x pt-inference.sh
pt-inference.sh
```

Run this code for onnx format (*.onnx):
```
chmod +x onnx-inference.sh
onnx-inference.sh
```
