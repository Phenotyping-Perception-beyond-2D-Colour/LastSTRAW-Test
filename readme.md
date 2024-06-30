# 3D tomato dataset

<!-- Data and code that this test set is based upon can be found at [LAST-Straw](https://lcas.github.io/LAST-Straw/) -->

IMPROTANT: The dataset is not yet published, and is currently being used for a paper. It is therefore, not possible to use this dataset for any publication without permission of Bart van Marrewijk and Gert Kootstra (bart.vanmarrewijk@wur.nl & gert.kootstra@wur.nl)

This repo contains two items. 
1. A basic data class that can be used to visualize the 3D tomato dataset. The class is based upon Pytorch's Dataset class and can therefore be used within Pytorch's dataloader used for training models for machine learning-> run python wurTomato.py/
The visualisation uses [Open3D](https://www.open3d.org/) 
2. An example to use the dataset to apply semantic segmentation using the pointcept git. Subsequently, convert the semantic algorithm to nodes, and evaluate the algorithm. 


## Installation (without pointcept)

```
conda create -n summerschool python=3.8 -y
conda activate summerschool
pip install open3d
pip install pandas
```

## Installation (including pointcept library)
```
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

FNow the fun part starts, to install the library. If you do not use docker, first check gpu achitecture at
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
```

For ptv3 you need to isntall flash attention:

```
pip install packaging
pip install flash-attn --no-build-isolation


```

## Visualising the dataset:
```
python wurTomato.py
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
Evaluation of skeletons consist of two parts:
- Evaluation of nodes. Are all nodes "detected". This can be determined using standard approaches like, TP, FP and FN. 
- Evaluation of edges. To our best knowledge there is no framework yet. The 4D plant registration seems to be an appropriate solution, but pre-testing showed that it is not yet perfect. Meaning that sometimes the nodes are not correctly connected to the ground truth dataset.

## Recommended literature & datasets:
- Last-Straw, strawberry dataset with annotations: https://arxiv.org/abs/2403.00566
- Pheno4D, tracking over time using skeletons: https://www.ipb.uni-bonn.de/data/pheno4d/ 
- Cucumber node detection using semantic segmenation: 



## Acknowledgement
This github would not be possible without open acces of several important libraries.

- Pointcept:              https://github.com/Pointcept/Pointcept
- Last-Straw:             https://github.com/LCAS/LAST-Straw
- 4d_plant_registration:  https://github.com/PRBonn/4d_plant_registration