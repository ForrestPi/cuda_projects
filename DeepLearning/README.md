## Dependences

Download [CUDA 7.5](https://developer.nvidia.com/cuda-downloads) and [cuDNN 5](https://developer.nvidia.com/cudnn). 

```shell
LIB_DIR=lib$([[ $(uname) == "Linux" ]] && echo 64)
CUDNN_LIB_DIR=/usr/local/cudnn/v5/$LIB_DIR
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_LIB_DIR >> ~/.profile && ~/.profile

tar zxvf cudnn*.tgz
sudo cp cuda/$LIB_DIR/* $CUDNN_LIB_DIR/
sudo cp cuda/include/* /usr/local/cudnn/v5/include/
```

## Compilation

```shell
./compile.sh
```

