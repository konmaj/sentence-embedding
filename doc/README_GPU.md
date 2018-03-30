# Configuration of GPU
This file decribes how to configure system to run tensorflow using GPU.
We assume that CUDA driver 384 is installed and *tensorflow-gpu* library in version 1.6.0 will be used.

## CUDA Toolkit 9.0
1. Download installation script for *CUDA Toolkit 9.0 (Sept 2017)*
   * select *runfile* installer type
   * e.g. Linux -> x86_64 -> Ubuntu -> 17.04 -> runfile(local)
2. Add permissions to execute downloaded script.
3. Run downloaded script:
   * deny installation of driver,
   * accept installation of *CUDA Toolkit 9.0*,
   * optionally you may also accept installation of samples.
   
## CUDA Deep Neural Network library
1. Download *cuDNN* library in version 7.1.2.
2. Unzip library to any location.

## Update environment variables

Into file `~/.bashrc` add following lines and complete assignments to `CUDA_TOOLKIT_PATH` and `CUDNN_PATH`
with paths, where you have installed libraries above.
```
CUDA_TOOLKIT_PATH=/path/to/cuda/toolkit/dir
CUDNN_PATH=/path/to/cudnn/dir
export PATH=${CUDA_TOOLKIT_PATH}/bin:${CUDNN_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib64:${CUDA_TOOLKIT_PATH}/lib64:${LD_LIBRARY_PATH}
```
