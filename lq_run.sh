#! /bin/bash
export LD_LIBRARY_PATH=/home/panyj/.local/lib/:$LD_LIBRARY_PATH

DIR=all_cnn_net_log/cifar10

mkdir -p ${DIR}
mkdir -p saved_models

#echo "" >> ${DIR}/lqnet.layerwise.log


#CUDA_VISIBLE_DEVICES=0 python3 main.py --arch all_cnn_net --dataset cifar10 --lr 1e-2 --epochs 500 --wd 1e-4  --lr_epochs 150  2>&1 | tee ${DIR}/all_cnn_net_full.log 

<<COMMAND1
for SIZE in 8 12 16 20 
do
    CUDA_VISIBLE_DEVICES=0,3 python3 main.py --ds ${SIZE} &> ${DIR}/small.${SIZE}.log
done
COMMAND1


#CUDA_VISIBLE_DEVICES=0,1 python3 main.py --arch all_cnn_c --dataset cifar10 --lr 1e-2 --epochs 450 --wd 1e-3 --admm --admm-iter 10 --pretrained saved_models/best.all_cnn_c.32.32.ckp_origin.pth.tar --bits 2 1 2 2 2 2 2 2 2  &> ${DIR}/admm.pretrained.log 2>&1

bitsets=(
    "4"
    "8"
    "16"
#    "  3 3  3 3 3  3 3 "
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
    #CUDA_VISIBLE_DEVICES=2,3 python3 main.py --arch all_cnn_c --dataset cifar10 --lr 1e-3 --epochs 500 --wd 1e-3 --lq  --bits ${i} --lr_epochs 200 --pretrained  saved_models/best.all_cnn_c.32.32.ckp_origin.pth.tar 2>&1 | tee ${DIR}/all_cnn_net_pretrained.log 
    CUDA_VISIBLE_DEVICES=2,3 python3 main.py --arch all_cnn_net --dataset cifar10 --lr 1e-2 --epochs 500 --wd 1e-4 --lq  --bits ${i} --needbias  --lr_epochs 150  2>&1 | tee "${DIR}/all_cnn_net_${i}_from_stretch.log"
    #tac ${DIR}/admm.pretrained.log | sed -e '/Acc@1/q' | tac >> ${DIR}/admm.layerwise.log
done
