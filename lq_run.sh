#! /bin/bash
export LD_LIBRARY_PATH=/home/panyj/.local/lib/:/home/panyj/intel/ipp/lib/:$LD_LIBRARY_PATH

ARCH=mobilenet_v2
DATASET=imagenet
#DIR=all_cnn_net_log/cifar10
DIR=log/

mkdir -p ${DIR}
mkdir -p saved_models

bitsets=(
#    "4"
#    "8"
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
    #CUDA_VISIBLE_DEVICES=2,3 python3 main.py --arch all_cnn_c --dataset cifar10 --lr 1e-3 --epochs 500 --wd 1e-3 --lq  --bits ${i} --lr_epochs 200 --needbias --pretrained  saved_models/best.all_cnn_c.32.32.ckp_origin.pth.tar 2>&1 | tee ${DIR}/all_cnn_net_pretrained.log 
    # all_cnn_net
    #CUDA_VISIBLE_DEVICES=0 python3 main.py --arch ${ARCH} --dataset ${DATASET} --lr 1e-4 --epochs 50 --wd 1e-4 --lq  --bits ${i} --lr_epochs 150 --needbias --quantAct --pretrained saved_models/best.lq.all_cnn_net_8.ckp_origin.pth.tar #2>&1 | tee "${DIR}/${ARCH}_${i}_from_stretch_convbnrelu.log"
    # alexnet
    #CUDA_VISIBLE_DEVICES=0 python3 main.py --arch ${ARCH} --dataset ${DATASET} --lr 1e-4 --epochs 20 --wd 1e-4 --bits ${i} --lr_epochs 20  --needbias --pretrained saved_models/best.alexnet_8.ckp_origin.pth.tar #2>&1 | tee "${DIR}/${ARCH}_${i}_from_stretch_convbnrelu.log"
    # resnet18
    #CUDA_VISIBLE_DEVICES=2 python3 main.py --arch ${ARCH} --dataset ${DATASET} --lr 1e-4 --epochs 10 --wd 1e-4 --lq  --bits ${i} --lr_epochs 30 --needbias #2>&1 | tee "${DIR}/${ARCH}_${i}_from_stretch_convbnrelu.log"
    # resnet50
    # CUDA_VISIBLE_DEVICES=2 python3 main.py --arch ${ARCH} --dataset ${DATASET} --lr 1e-4 --epochs 10 --wd 1e-4 --lq  --bits ${i} --lr_epochs 30 --needbias #2>&1 | tee "${DIR}/${ARCH}_${i}_from_stretch_convbnrelu.log"
    # resnet20
    #CUDA_VISIBLE_DEVICES=0 python3 main.py --arch ${ARCH} --dataset ${DATASET} --lr 1e-4 --epochs 10 --wd 1e-4 --lq  --bits ${i} --lr_epochs 30 --needbias #2>&1 | tee "${DIR}/${ARCH}_${i}_from_stretch_convbnrelu.log"
    # googlenet
    #CUDA_VISIBLE_DEVICES=0,1 python3 main.py --arch ${ARCH} --dataset ${DATASET} --lr 1e-4 --epochs 20 --wd 1e-4 --lq  --bits ${i} --lr_epochs 15 --needbias #2>&1 | tee "${DIR}/${ARCH}_${i}_from_stretch_convbnrelu.log"
    # squeezenet
    #CUDA_VISIBLE_DEVICES=1 python3 main.py --arch ${ARCH} --dataset ${DATASET} --lr 1e-3 --epochs 30 --wd 1e-4  --lq --bits ${i} --lr_epochs 15 --needbias --quantAct #2>&1 | tee "${DIR}/${ARCH}_${i}_from_stretch_convbnrelu.log"
    # mobilenet_v2
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 main.py --arch ${ARCH} --dataset ${DATASET} --lr 1e-3 --epochs 15 --wd 1e-4  --lq --bits ${i} --lr_epochs 15 --needbias # -e --pretrained saved_models/best.lq.mobilenet_v2_16.ckp_origin.pth.tar #2>&1 | tee "${DIR}/${ARCH}_${i}_from_stretch_convbnrelu.log"
    # CUDA_VISIBLE_DEVICES=1 python3 main.py -e --arch ${ARCH} --dataset ${DATASET} --lr 1e-3 --epochs 1 --wd 1e-4  --lq --bits ${i} --lr_epochs 15 --needbias -e --pretrained saved_models/best.lq.mobilenet_v2_16.ckp_origin.pth.tar #2>&1 | tee "${DIR}/${ARCH}_${i}_from_stretch_convbnrelu.log"
    # vgg
    #CUDA_VISIBLE_DEVICES=0 python3 main.py --arch ${ARCH} --dataset ${DATASET} --lr 1e-4 --epochs 5 --wd 1e-4 --lq  --bits ${i} --lr_epochs 30  #2>&1 | tee "${DIR}/${ARCH}_${i}_from_stretch_convbnrelu.log"
    #CUDA_VISIBLE_DEVICES=0 python3 main.py --arch ${ARCH} --dataset ${DATASET} --lr 1e-3 --epochs 500 --wd 1e-4 --lq  --bits ${i} --lr_epochs 30  #2>&1 | tee "${DIR}/${ARCH}_${i}_from_stretch_convbnrelu.log"
    #CUDA_VISIBLE_DEVICES=0 python3 main.py --arch all_cnn_net --dataset cifar10 --lr 1e-2 --epochs 500 --wd 1e-4 --lq  --bits ${i} --needbias --block_type convrelubn --lr_epochs 150  2>&1 | tee "${DIR}/all_cnn_net_${i}_from_stretch_convrelubn.log"
    #tac ${DIR}/admm.pretrained.log | sed -e '/Acc@1/q' | tac >> ${DIR}/admm.layerwise.log
done
