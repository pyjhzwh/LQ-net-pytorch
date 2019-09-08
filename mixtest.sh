#! /bin/bash
export LD_LIBRARY_PATH=/home/panyj/.local/lib/:$LD_LIBRARY_PATH

DIR=resnet_log/imagenet

mkdir -p ${DIR}
mkdir -p saved_models

DS=84
CROP=224
#CUDA_VISIBLE_DEVICES=0,1 python3 main.py --ds 112 --crop 168 --arch 'resnet18' --dataset 'imagenet' -e -m --pretrained 'saved_models/best.resnet18.224.ckp_origin.pth.tar' 'saved_models/best.resnet18.112.168.ckp_origin.pth.tar' &> ${DIR}/mix.log
CUDA_VISIBLE_DEVICES=1 python3 main.py --ds ${DS} --crop ${CROP} --arch 'resnet18' --dataset 'imagenet' -e -m 1 --pretrained 'saved_models/best.resnet18.224.224.ckp_origin.pth.tar' 'saved_models/best.resnet18.'"${DS}.${CROP}"'.ckp_origin.pth.tar'  --skip &> ${DIR}/mix_${DS}_skip.log 
#--show_informloss_conf 


