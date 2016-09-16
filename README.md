# chainer-fast-neuralstyle-video
Simple video wrapper for [chainer fast neuralstyle app](https://github.com/yusuketomoto/chainer-fast-neuralstyle) by yusuketomoto. 

![edtaonisl](edtaonisl.gif?raw=true)
## Required Dependencies
- python 2.7
- numpy 
- opencv2 See: http://www.mobileway.net/2015/02/14/install-opencv-for-python-on-mac-os-x/
- chainer
- cuda sdk/toolkit See: http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/
- cudNN See: https://developer.nvidia.com/cudnn

## Installation
Make sure all cuda related environment variables are set. This is how it looks for my configuration:
```
export LD_LIBRARY_PATH=/opt/cudann/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/cudann/lib:$LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib
export CPATH=/opt/cudann/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
```
Run:
```
bash getDeps.sh # downloads the OLD! models from https://github.com/gafr/chainer-fast-neuralstyle-models 
python setup.py install
```

## Start
Before you run the app make sure to configure the constants in video.py.

```
python video.py
```
Swap the different models with keys 1-7



