#! /bin/bash
export LD_LIBRARY_PATH=/home/panyj/.local/lib/:/home/panyj/intel/ipp/lib/:$LD_LIBRARY_PATH

#DIR=all_cnn_net_log/cifar10
DIR=alexnet_log/imagenet

mkdir -p ${DIR}
mkdir -p saved_models

bitsets=(
#    "4"
    "8"
#    "16"
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
    CUDA_VISIBLE_DEVICES=0 python3 main.py --arch alexnet --dataset imagenet --lr 1e-3 --epochs 200 --wd 1e-5 --lq  --bits ${i} --lr_epochs 30  2>&1 | tee "${DIR}/alexnet_${i}_from_stretch_convbnrelu.log"
    #CUDA_VISIBLE_DEVICES=0 python3 main.py --arch all_cnn_net --dataset cifar10 --lr 1e-2 --epochs 500 --wd 1e-4 --lq  --bits ${i} --needbias --block_type convrelubn --lr_epochs 150  2>&1 | tee "${DIR}/all_cnn_net_${i}_from_stretch_convrelubn.log"
    #tac ${DIR}/admm.pretrained.log | sed -e '/Acc@1/q' | tac >> ${DIR}/admm.layerwise.log
done
