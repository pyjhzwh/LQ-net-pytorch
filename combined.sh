#! /bin/bash

DIR=saved_models/
SUFFIX=.ckp_origin.pth.tar

CUDA_VISIBLE_DEVICES=1 python3 combining.py --ds 8 -e  --pretrained ${DIR}/cifar10.8${SUFFIX} ${DIR}/cifar10.8.24${SUFFIX} --seed 1
#CUDA_VISIBLE_DEVICES=0,3 python3 combining.py --ds 8 -e True --pretrained ${DIR}/cifar10.8${SUFFIX} ${DIR}/cifar10.8.24${SUFFIX} &> resnet_log/combined.log
