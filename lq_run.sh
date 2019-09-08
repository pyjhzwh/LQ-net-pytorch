#! /bin/bash
export LD_LIBRARY_PATH=/home/panyj/.local/lib/:$LD_LIBRARY_PATH

DIR=all_cnn_net_log/cifar10

mkdir -p ${DIR}
mkdir -p saved_models

#echo "" >> ${DIR}/lqnet.layerwise.log

i=0


<<COMMAND1
for SIZE in 8 12 16 20 
do
    CUDA_VISIBLE_DEVICES=0,3 python3 main.py --ds ${SIZE} &> ${DIR}/small.${SIZE}.log
done
COMMAND1


#CUDA_VISIBLE_DEVICES=0,1 python3 main.py --arch all_cnn_c --dataset cifar10 --lr 1e-2 --epochs 450 --wd 1e-3 --admm --admm-iter 10 --pretrained saved_models/best.all_cnn_c.32.32.ckp_origin.pth.tar --bits 2 1 2 2 2 2 2 2 2  &> ${DIR}/admm.pretrained.log 2>&1

bitsets=(
    "  2 2  2 2 2  2 2 "
#    "  2 2  1 1 1  2 2 "
#    "  2 2  2 2 2  1 1 "
#    "  1 1  1 1 1  2 2 "
#    "  1 1  2 2 2  1 1 "
#    "  2 2  1 1 1  1 1 "
)
for i in "${bitsets[@]}"; do
    #echo "" >> ${DIR}/admm.layerwise.log
    #echo "$i" >> ${DIR}/admm.layerwise.log
    #echo "--arch all_cnn_c --dataset cifar10 --lr 1e-3 --epochs 1000 --wd 1e-3 --lq --lq_iter --bits ${i}" >> ${DIR}/admm.layerwise.log
    CUDA_VISIBLE_DEVICES=2 python3 main.py --arch all_cnn_c --dataset cifar10 --lr 1e-2 --epochs 1000 --wd 1e-3 --lq  --bits ${i} --lr_epochs 100 &> ${DIR}/lqnet.fromstrach.log 2>&1
    #tac ${DIR}/admm.pretrained.log | sed -e '/Acc@1/q' | tac >> ${DIR}/admm.layerwise.log
done
