#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
PYTHON=python

TEST_CODE=test.py

DATASET=LAST-Straw
DATASET=MarvinDatasetCSV
dataset_folder=/media/agro/PhDBart1/GARdata_local/
CONFIG=semseg-pt-v3m1-0-base_LASTSTRAW
CONFIG=semseg-pt-v3m1-0-base



EXP_NAME=20240608_1453_debug



# EXP_NAME=debug
WEIGHT=model_best
# WEIGHT=model_last
GPU=1

while getopts "p:d:c:n:w:g:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "GPU Num: $GPU"

# EXP_DIR=exp/${DATASET}/${EXP_NAME}
EXP_DIR=${dataset_folder}/experiments/${DATASET}/trained_models/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
# CODE_DIR=${EXP_DIR}/code
CODE_DIR=${EXP_DIR}/model

CONFIG_DIR=${EXP_DIR}/config.py
# CONFIG_DIR=example_configs/${DATASET}/${CONFIG}.py
CONFIG_DIR=example_configs/${CONFIG}.py



# if [ "${CONFIG}" = "None" ]
# then
#     CONFIG_DIR=${EXP_DIR}/config.py
# else
#     CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
# fi

echo "Loading config in:" $CONFIG_DIR
#export PYTHONPATH=./$CODE_DIR
# export PYTHONPATH=Pointcept

echo "SavePath" $EXP_DIR
WEIGHT_NAME="${MODEL_DIR}"/"${WEIGHT}".pth
echo "WeightName" $WEIGHT_NAME

echo " =========> RUN TASK <========="
echo "Running code in: $CODE_DIR"

# dangeous but we added this line, because otherwise if you change the settings not
rm -r "$EXP_DIR"/result 

# $PYTHON -u "$CODE_DIR"/tools/$TEST_CODE \
$PYTHON -u Pointcept/tools/$TEST_CODE \
  --config-file "$CONFIG_DIR" \
  --num-gpus "$GPU" \
  --options \
  weight="$WEIGHT_NAME" \
  save_path="$EXP_DIR" 

# export CUDA_LAUNCH_BLOCKING=1; \

# export dataset_folder=/media/agro/PhDBart1/GARdata_local/; \
# export dataset_name=marvin_pointcloud/; \
# $PYTHON -u Pointcept/tools/test.py --config-file "$CONFIG_DIR" --num-gpus 1 --options \
#   weight="$WEIGHT_NAME"
  # save_path=/media/agro/PhDBart1/GARdata_local/experiments/marvin_pointcloud/20240218_0215_swin3d/results_noaug \

