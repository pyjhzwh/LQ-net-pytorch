#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import warnings
import time
import random
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt#; plt.rcdefaults()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights as MobileNet_V2_Weights
from torch.autograd import Variable
from utils import *

import modelarchs
import lqnet


def test(val_loader, model, epoch, args, stats=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()


    with torch.no_grad():
        end = time.time()
        # apply quantized value to testing stage
        if args.lq:
            LQ.apply(test=True)
        #if args.lq:
            #LQ.storeW()
            #LQ.apply_quantval()

        for i, (images, target) in enumerate(val_loader):
            #if args.gpu is not None:
            #    images = images.cuda(args.gpu, non_blocking=True)
            #target = target.cuda(args.gpu, non_blocking=True)
            images, target = Variable(images.cuda()), Variable(target.cuda())

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        #restore the floating point value to W
        if args.lq:
            LQ.restoreW()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))


    return top1.avg


def train(train_loader,optimizer, model, epoch, args, stats=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()

    model.train()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images, target = Variable(images.cuda()), Variable(target.cuda())

        # apply quantized value to W
        #if args.lq:
            #LQ.apply_quantval()
        if args.lq:
            LQ.apply()

        # compute output
        output= model(images)
        #rint('stats',stats)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        #print('apply quantized')
        #LQ.print_weights()

        # use gradients of Bi to update Wi
        if args.lq:
            LQ.restoreW()
        #print('before step')
        #LQ.print_weights()
        optimizer.step()

        #print('after step')
        #LQ.print_weights()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0:
            progress.display(i,optimizer)

    print('Finished Training')

    if args.lq:
        #if epoch % 10 == 9:
        LQ.print_info()
        #if epoch == args.epochs -1:
        #    print('store quantized weights')
        #    LQ.storequntW()
        #model.print_Actinfo()

    return

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__=='__main__':
    # imagenet_datapath= '/datasets01/imagenet_full_size/061417/'
    imagenet_datapath= '/data2/jiecaoyu/imagenet/imgs/'
    parser = argparse.ArgumentParser(description='PyTorch MNIST ResNet Example')
    parser.add_argument('--no_cuda', default=False, 
            help = 'do not use cuda',action='store_true')
    parser.add_argument('--epochs', type=int, default=450, metavar='N',
            help='number of epochs to train (default: 450)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr_epochs', type=int, default=100, metavar='N',
            help='number of epochs to change lr (default: 100)')
    parser.add_argument('--pretrained', default=None, nargs='+',
            help='pretrained model ( for mixtest \
            the first pretrained model is the big one \
            and the sencond is the small net)')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    default=False, help='evaluate model on validation set')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--arch', action='store', default='resnet20',
                        help='the CIFAR10 network structure: \
                        resnet20 | resnet18 | resnet50 | resnet152 | all_cnn_net | alexnet')
    parser.add_argument('--dataset', action='store', default='cifar10',
            help='pretrained model: cifar10 | imagenet')
    parser.add_argument('--lq', default=False, 
            help = 'use lq-net quantization or not',action='store_true')
    parser.add_argument('--bits', default = [2,2,2,2,2,2,2,2,2], type = int,
                    nargs = '*', help = ' num of bits for each layer')
    parser.add_argument('--needbias', default=False, 
            help = 'use bias in quantized value or not',action='store_true')
    parser.add_argument('--block_type', action='store', default='convbnrelu',
            help='convbnrelu, convrelubn or convbnsilu')
    parser.add_argument('--quantAct', default=False, 
            help = 'quant activations or not',action='store_true')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.dataset == 'cifar10':
        # load cifa-10
        nclass = 10
        normalize = transforms.Normalize(
                mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=
                                            transforms.Compose([
                                                transforms.RandomCrop(32,padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                                ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=12)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=
                                           transforms.Compose([
                                               transforms.RandomCrop(32),
                                               transforms.ToTensor(),
                                               normalize,
                                               ]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=12)


    if args.dataset == 'imagenet':
        nclass=1000
        traindir = os.path.join(imagenet_datapath,'train')
        testdir = os.path.join(imagenet_datapath,'val')
        torchvision.set_image_backend('accimage')

        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trainset = torchvision.datasets.ImageFolder(root=traindir,transform=
                                            transforms.Compose([
                                                #transforms.Resize(256),
                                                #transforms.CenterCrop(args.crop),
                                                #transforms.RandomCrop(args.crop),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                                ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=12)

        testset = torchvision.datasets.ImageFolder(root=testdir,transform=
                                           transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               normalize,
                                               ]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=12)


    if args.arch == 'resnet50':
        pretrained = True
        model = modelarchs.resnet50(pretrained = pretrained)
        bestacc = 0
        

    elif args.arch == 'resnet18':
        #pretrained = False if args.pretrained is not None else True
        pretrained = True
        model = modelarchs.resnet18(pretrained = pretrained)
        bestacc = 0
    
    elif args.arch == 'resnet152':
        pretrained = True
        model = modelarchs.resnet152(pretrained = pretrained)
        bestacc = 0
    
    elif args.arch == 'resnet20':
        #pretrained = False if args.pretrained is not None else True
        pretrained = True
        model = modelarchs.resnet20(pretrained = pretrained)
        bestacc = 0

    elif args.arch == 'alexnet':
        pretrained = True
        model = modelarchs.alexnet(pretrained = pretrained, block_type=args.block_type)
        #model = torchvision.models.alexnet(pretrained = False)

    elif args.arch == 'all_cnn_net':
        model = modelarchs.all_cnn_net(block_type=args.block_type, quantAct=args.quantAct,
            bits = [args.bits[0]]*7)

    elif args.arch == 'squeezenet':
        model = modelarchs.squeezenet1_1(pretrained=True, progress=True, quantAct=args.quantAct,
            bits = args.bits[0])
    
    elif args.arch == 'mobilenet_v2':
        model = modelarchs.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    
    elif args.arch == 'googlenet':
        model = modelarchs.googlenet(pretrained=True, progress=True)

    elif args.arch == 'vgg11':
        pretrained = True
        model = modelarchs.vgg11_bn(pretrained = pretrained)

    elif args.arch == 'vgg13':
        pretrained = True
        model = modelarchs.vgg13_bn(pretrained = pretrained)

    elif args.arch == 'vgg16':
        pretrained = True
        model = modelarchs.vgg16_bn(pretrained = pretrained)

    elif args.arch == 'vgg19':
        pretrained = True
        model = modelarchs.vgg19_bn(pretrained = pretrained)
    else:
        raise ValueError("Unsupported arch {}".format(args.arch))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), 
                lr=args.lr, momentum=args.momentum, weight_decay= args.weight_decay)

    if not args.pretrained:
        bestacc = 0
    else:
        pretrained_model = torch.load(args.pretrained[0])
        bestacc = 0#pretrained_model['acc']
        args.start_epoch = pretrained_model['epoch']
        # model.load_state_dict(pretrained_model['state_dict'])
        load_state(model, pretrained_model['state_dict'])
        if 'quant_info' in pretrained_model:
            quantInfo = pretrained_model['quant_info']
            print('quant_info', quantInfo)
        if 'quant_Actinfo' in pretrained_model:
            quantActinfo = pretrained_model['quant_Actinfo']
            print('quant_Actinfo', quantActinfo)
        #optimizer.load_state_dict(pretrained_model['optimizer'])

    if args.cuda:
        model.cuda()
        model = nn.DataParallel(model, 
                    device_ids=range(torch.cuda.device_count()))
        #model = nn.DataParallel(model, device_ids=args.gpu)

    #print(model)
    '''
    for name, param in model.named_parameters():
        if param.requires_grad and 'conv2' in name:
            print(name, param.data[0])
    '''
    if args.lq:
        target_weights = gen_target_weights(model, args.arch)
        if len(args.bits) == 0:
            raise ValueError("Empty setting for quatized bits")
        elif len(args.bits) < len(target_weights):
            print("Warning: set all quantized bits as {}".format(str(args.bits[0])))
            args.bits = [args.bits[0]] * len(target_weights)
        LQ = lqnet.learned_quant( target_weights, b = args.bits,needbias=args.needbias)

    ''' evaluate model accuracy and loss only '''
    if args.evaluate:
        test(testloader, model, args.start_epoch, args)
        #if args.lq:
            #weightsdistribute(model)
            #weight_mean(model,args.arch)
        exit()


    ''' train model '''
    quantActdict={}
    
    for epoch in range(0,args.epochs):
        running_loss = 0.0
        adjust_learning_rate(optimizer, epoch, args)
        train(trainloader,optimizer, model, epoch, args, quantActdict)
        acc = test(testloader, model, epoch, args, quantActdict)
        quantdict=None
        if args.lq:
            quantdict = LQ.save_quantinfo()
        if args.lq:
            print('store quantized weights')
            LQ.apply(test=True)
        # print('quantActdict',quantActdict)
        if (acc > bestacc):
            bestacc = acc
            save_state(model,acc,epoch,args, optimizer, True, quantdict,quantActdict)
        else:
            save_state(model,bestacc,epoch,args,optimizer, False, quantdict,quantActdict)
        if args.lq:
            LQ.restoreW()
        print('best acc so far:{:4.2f}'.format(bestacc))
    
    if args.lq:
        filename='saved_models/best.lq.'+str(args.arch)+'_'+str(args.bits[0])+'.ckp_origin.pth.tar'
        best_model = torch.load(filename)
        load_state(model, best_model['state_dict'])
        weightsdistribute(model)



