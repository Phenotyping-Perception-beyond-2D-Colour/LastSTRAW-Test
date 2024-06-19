# LastSTRAW dataset test

Data and code that this test set is based upon can be found at [LAST-Straw](https://lcas.github.io/LAST-Straw/)

This repo contains a data importer class that can download, unzip and visualise the LAST-Straw data. The class is based upon Pytorch's Dataset class and can therefore be used within Pytorch's dataloader used for training models for machine learning.

The visualisation uses [Open3D](https://www.open3d.org/) 

## Installation

```
conda create -n summerschool python=3.8 -y
conda activate summerschool
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric
pip install spconv-cu113
pip install pandas
```

Fun part: without docker check gpu achitecture at
https://developer.nvidia.com/cuda-gpus 


```
# PTv1 & PTv2 or precise eval
cd Pointcept/libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
#TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="8.6" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d
```

For ptv3 you need to isntall flash attention:

```
pip install packaging
pip install flash-attn --no-build-isolation


```

## Training model:
```
python Pointcept/tools/train.py --config-file example_configs/semseg-pt-v3m1-0-base.py --num-gpus 1 --options save_path=./work_dir/debug/
```

## Inference model and evaluation using pointcept
```
python Pointcept/tools/test.py --config-file example_configs/semseg-pt-v3m1-0-base.py --num-gpus 1 --options 
                weight=./work_dir/debug/model/model_best.pth
                save_path=./work_dir/debug/
```

## Evaluation of skeleton
WIP