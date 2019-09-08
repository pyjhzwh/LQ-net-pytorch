#! /bin/bash

DIR=resnet_log

mkdir -p ${DIR}
mkdir -p saved_models

CUDA_VISIBLE_DEVICES=0,3 python3 main.py &> ${DIR}/big.log 2>&1


for SIZE in 8 12 16 20 
do
    CUDA_VISIBLE_DEVICES=0,3 python3 main.py --ds ${SIZE} &> ${DIR}/small.${SIZE}.log
done
