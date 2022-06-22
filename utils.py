#!/usr/bin/env python3

import argparse
import os
import warnings
import time
import random
import shutil
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import modelarchs

def save_state(model, best_acc, epoch, args,optimizer, isbest, quant_info=None, quantActdict=None):
    if args.block_type == 'convbnsilu':
        dirpath = 'saved_models/swish/'
    else:
        dirpath = 'saved_models/'
    suffix = '.ckp_origin.pth.tar'
    state = {
            'acc': best_acc,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'isbest': isbest,
            'quant_info': quant_info,
            'quant_Actinfo': quantActdict
            }
    if not args.lq:
        filename = str(args.arch)+'_'+str(args.bits[0])+suffix
    else:
        if not args.quantAct:
            filename = 'lq.'+str(args.arch)+'_'+str(args.bits[0])+suffix
        else:
            filename = 'lq.'+str(args.arch)+'_'+str(args.bits[0])+'_'+str(args.bits[0])+suffix
    torch.save(state,dirpath+filename)
    if isbest:
        shutil.copyfile(dirpath+filename, dirpath+'best.'+filename)
    
    #torch.save(state,'saved_models/{}.{}.{}.ckp_origin.pth.tar'.format(args.arch,args.ds,args.crop))
    return

def load_state(model, state_dict):
    cur_state_dict = model.state_dict()
    state_dict_keys = state_dict.keys()
    for key in cur_state_dict:
        if key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key])
        elif key.replace('module.','') in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key.replace('module.','')])
        elif 'module.'+key in state_dict_keys:
            #print(key, state_dict['module.'+key].shape, cur_state_dict[key].shape)
            cur_state_dict[key].copy_(state_dict['module.'+key])

    
    #model.load_state_dict(state_dict)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch,optimizer=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if optimizer is not None:
            entries += ['lr: {:.1e}'.format(optimizer.param_groups[0]['lr'])]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every lr-epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def to_cuda_optimizer(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

def weightsdistribute(model):
    for key, value in model.named_parameters():
        if 'conv.weight' in key:
            unique, count = torch.unique(value.data, sorted=True, return_counts= True)
            print('layer ', key)
            print('first weight', value.data[0,0,0])
            #bias = torch.mean(value)
            #print('bits:', torch.log2(torch.max(unique.abs())/torch.min(unique.abs())+1))
            #print('bits:', math.log2(len(unique)))
            print(unique,":",count)
            #print('basis',torch.min(unique.abs()))
            #print('bias',bias)
            #print('bits',torch.log2((torch.max(unique.abs()) - bias)/(torch.min(unique.abs()) -bias) +1))

def gen_target_weights(model, arch):
    target_weights = []
    if arch == 'resnet18':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if (m.weight.data.shape[1] > 3) and (m.weight.data.shape[2] > 1):
                    target_weights.append(m.weight)

    elif arch == 'all_cnn_c' or arch == 'all_cnn_net' \
            or arch == 'squeezenet' or arch == 'resnet20' \
            or arch == 'resnet50' or arch == 'mobilenet_v2':
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                target_weights.append(m.weight)
        target_weights = target_weights[1:-1]
    
    #elif arch == 'resnet50':
    #    for m in model.modules():
    #        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #            target_weights.append(m.weight)
    #    target_weights = target_weights[1:4] + target_weights[5:14] + \
    #        target_weights[15:27] + target_weights[28:46] + target_weights[47:-1]
    

    elif arch == 'alexnet' or 'vgg' in arch or arch == 'googlenet' or arch == 'squeezenet':
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                target_weights.append(m.weight)
        target_weights = target_weights[1:-1]

    else:
        raise Exception ('{} not supported'.format(arch))
    print('\nQuantizing {} layers:'.format(len(target_weights)))
    for item in target_weights:
        print(item.shape)
    print('\n')
    return target_weights


def weight_mean(model,arch):
    i=0
    if arch == 'resnet18':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if (m.weight.data.shape[1] > 3) and (m.weight.data.shape[2] > 1):
                    print(i,'th layer mean',torch.mean(m.weight.data)/torch.min(m.weight.data.abs()))
                    i = i+1
                    

    elif arch == 'all_cnn_c' or arch == 'all_cnn_net' or arch == 'squeezenet' or arch == 'resnet20' or arch == 'resnet50':
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    print(i,'th layer mean',torch.mean(m.weight.data)/torch.min(m.weight.data.abs()))
                    print('mean',torch.mean(m.weight.data)/torch.min(m.weight.data.abs()), 'min',torch.min(m.weight.data)/ torch.min(m.weight.data.abs()), 'max', torch.max(m.weight.data)/ torch.min(m.weight.data.abs()))
                    i = i+1
    elif arch == 'alexnet' or 'vgg' in arch or arch == 'googlenet' or arch == 'squeezenet' or arch == 'mobilenet_v2':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                    print(i,'th layer mean',torch.mean(m.weight.data)/torch.min(m.weight.data.abs()))
                    print('mean',torch.mean(m.weight.data)/torch.min(m.weight.data.abs()), 'min',torch.min(m.weight.data)/ torch.min(m.weight.data.abs()), 'max', torch.max(m.weight.data)/ torch.min(m.weight.data.abs()))
                    i = i+1

    else:
        raise Exception ('{} not supported'.format(arch))
    print('\n')
    return 
