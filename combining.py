#!/usr/bin/env python3

import argparse
import os
import warnings
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import resnet
from torch.autograd import Variable
from utils import *

def test(val_loader, model, epoch):
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

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def train(train_loader,optimizer, model, epoch):
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

        #if args.gpu is not None:
            #images = images.cuda(args.gpu, non_blocking=True)
        #target = target.cuda(args.gpu, non_blocking=True)
        images = images.cuda()
        target = target.cuda()
        #images, target = Variable(images.cuda()), Variable(target.cuda())

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            progress.display(i,optimizer)

    print('Finished Training')
    return

def testcombined(val_loader, model, model2, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1model0 = AverageMeter('Acc@1-M0', ':6.2f')
    top1model1 = AverageMeter('Acc@1-M1', ':6.2f')
    top1model2 = AverageMeter('Acc@1-M2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top1model0, top1model1, top1model2, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    model2.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images, target = images.cuda(), target.cuda()
            images1 = torch.flip(images,[3])

            # randomly crop 24*24 
            if args.seed is not None:
                random.seed(args.seed)
            cornerh = random.randint(0,7)
            cornerw = random.randint(0,7)
            images2 = images[:,:,cornerh:cornerh+24,cornerw:cornerw+24]


            # compute output
            output0 = model(images)
            loss0 = criterion(output0, target)

            output1 = model(images1)
            loss1 = criterion(output1, target)

            output2 = model2(images2)
            loss2 = criterion(output2, target)

            # measure accuracy and record loss
            acc1, acc5, acc1model0, acc1model1, acc1model2= accuracy_mv([output0, output1, output2], target, topk=(1, 5))
            losses.update((loss0.item()+loss1.item()+loss2.item())/3, images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            top1model0.update(acc1model0[0], images.size(0))
            top1model1.update(acc1model1[0], images.size(0))
            top1model2.update(acc1model2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Acc@1-M0 {top1model0.avg:.3f} Acc@1-M1 {top1model1.avg:.3f} Acc@1-M2 {top1model2.avg:.3f}'
              .format(top1=top1, top5=top5, top1model0 = top1model0, top1model1 = top1model1, top1model2 = top1model2))

    return top1.avg

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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_mv(outputs, target, topk=(1,)):
    """Computes the accuracy of combined networks over the k top predictions for the specified values of k
    using majority voting"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        corrects = torch.zeros(3, maxk,batch_size, dtype=torch.uint8).cuda()
        for index, output in enumerate(outputs):
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            #print(pred)
            corrects[index] = pred.eq(target.view(1, -1).expand_as(pred))

        correct = torch.sum(corrects,0)
        correct = correct > 1
        
        #print(corrects[:,0,:])
        #print(correct[0,:])
        

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        for i in range(3):
            correct_1_modeli = corrects[i][:1].view(-1).float().sum(0, keepdim=True)
            res.append(correct_1_modeli.mul_(100.0 / batch_size))
        return res


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST ResNet Example')
    parser.add_argument('--ds', type=int, default=32, 
            help = 'down sample size')
    parser.add_argument('--no_cuda', default=False, 
            help = 'do not use cuda',action='store_true')
    parser.add_argument('--epochs', type=int, default=450, metavar='N',
            help='number of epochs to train (default: 450)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr_epochs', type=int, default=100, metavar='N',
            help='number of epochs to change lr (default: 100)')
    parser.add_argument('--pretrained', default=None, nargs='+',
            help='pretrained model')
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
                    help='evaluate model on validation set')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        #cudnn.deterministic = True
        #warnings.warn('You have chosen to seed training. '
        #              'This will turn on the CUDNN deterministic setting, '
        #              'which can slow down your training considerably! '
        #              'You may see unexpected behavior when restarting '
        #              'from checkpoints.')
    # load cifa-10
    normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=
                                        transforms.Compose([
                                            transforms.RandomCrop(32,padding=2),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize,
                                            ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=
                                       transforms.Compose([
                                           transforms.ToTensor(),
                                           normalize,
                                           ]))

    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=16)

    model = resnet.resnet20(nclass=10,ds=args.ds);
    model2 = resnet.resnet20(nclass=10,ds=args.ds);

    if args.cuda:
        model.cuda()
        model = nn.DataParallel(model, 
                device_ids=range(torch.cuda.device_count()))
        model2.cuda()
        model2 = nn.DataParallel(model2, 
                device_ids=range(torch.cuda.device_count()))
        #model = nn.DataParallel(model, device_ids=args.gpu)

    print(model)


    '''elif not args.combined:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        args.start_epoch = pretrained_model['epoch']
        load_state(model, pretrained_model['state_dict'])'''
    if args.pretrained:
        pretrained_model = torch.load(args.pretrained[0])
        best_acc = pretrained_model['acc']
        print("{} best acc:{}".format(args.pretrained[0],best_acc))
        args.start_epoch = pretrained_model['epoch']
        load_state(model, pretrained_model['state_dict'])
        #model.load_state_dict(pretrained_model['state_dict'])

        pretrained_model2 = torch.load(args.pretrained[1])
        best_acc = max(best_acc,pretrained_model2['acc'])
        print("{} best acc:{}".format(args.pretrained[1],best_acc))
        args.start_epoch = pretrained_model2['epoch']
        load_state(model2, pretrained_model2['state_dict'])

    else:
        bestacc=0

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = optim.SGD(model.parameters(), 
            lr=args.lr, momentum=args.momentum, weight_decay= args.weight_decay)

    ''' evaluate model accuracy and loss only '''
    if args.evaluate:
        testcombined(testloader, model, model2, criterion, args)
        exit()

    ''' train '''
    for epoch in range(args.start_epoch,args.epochs):
        running_loss = 0.0
        adjust_learning_rate(optimizer, epoch, args)
        train(trainloader,optimizer, model, epoch)
        acc = test(testloader, model, epoch)
        if (acc > bestacc):
            bestacc = acc
            save_state(model,acc,epoch,args)
        else:
            save_state(model,bestacc,epoch,args)
