# LQ-net-pytorch
LQ-net implementation on pytorch
There is no need for full-precision pretrained models
weights will be quantized according to the formula below
$$w = Qw * w_{scale} + w_{bias}$$


## File structure
* lq_run.sh - bash file to run the LQ-net codes
* lqnet.py - that's where LQ-net quantizer work
* utils.py - quantizing network helpers: gen_target_weights, save_state, etc.
* main.py - main 
* modelarchs/\* - network architecture file
* saved_models/\* - pytorch saved models

### Explaination for some files/functions
* gen_target_weights(model, arch) in utils.py: By specifying the layers that need quantization to target_weights list, the LQ-net quantizer will quantize them in the training process. Usually to maintain high accuracy, the first or last layer will not be quantized
* learned_quant class in lqnet.py:
    - __init__: initilized by target_weights, bits set for each layer, etc.
    - update(): update the scale and bias value to optimize quantization error; self.B => Qw, self.v => $w_{scale}$, self.Wmean => $w_{bias}$; Note that Qw = [ ..., -5, -3 , -1 , 1, 3, 5, ...] rather than [ ..., -3, -2, -1, 0, 1 ,2 ,...]

## Usage
```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --arch [cnn-arch] --dataset [cifar10|imagenet] --lr 1e-2 --epochs 500 --wd 1e-4 --lq  --bits [specify each layer bits] [--needbias] --block_type [convrelubn|convbnrelu] --lr_epochs 150
```
Or you could modify the lq_run.sh according to your needs

### Aruguments

* -lq: You need add '-lq' to enable LQ-net quantization process
* --bits [specify each layer bits]: Either enter a single value, like "8", which will apply 8-bit quantization for all target layers
* --needbias: if add this argmuent, $w = Qw * w_scale + w_bias$; Otherwise, $w = Qw * w_scale$. Adding this argument is recommended
