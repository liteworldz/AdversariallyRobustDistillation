import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

#from tvision import models

from models import *
#import preactresnet
#from models import *
import dataset

import torchattacks
import argparse
import numpy as np
from utils import progress_bar
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # setting the visibility only

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS= 8/255
ALPHA= 2/255
STEPS= 10

def test(net, adversary, filename, val_size, batch_size, data_type):
    
    load_path = "./checkpoint/"
    #checkpoint = torch.load(load_path + filename)['net']
    
    #net.load_state_dict(checkpoint)
    net.load_state_dict(torch.load('./checkpoint/CIFAR10/output/epoch=39.t7')['net'])

    trainloader, valloader, testloader = dataset.get_loader(val_size, batch_size)

    if data_type=='test':
        data_loader = testloader
    elif data_type=='val':
        data_loader = valloader
    else:
        data_loader = trainloader
    print('==> Loaded Model data..')
    
    # Data
    print('==> Preparing data..')
    criterion = nn.CrossEntropyLoss()
    print('\n[ Test Start on ', data_type, ' Dataset]')
    start = time.time()
    loss = 0
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0

    AUTOadversary = torchattacks.AutoAttack(
        net, norm='Linf', eps=EPS, version='standard', n_classes=10, seed=None, verbose=False)
    PGDadversary = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS)
    
    net.eval()
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        
        progress_bar(batch_idx, len(data_loader), f'Current batch: {batch_idx}')

        outputs = net(inputs)
        #loss = criterion(outputs, targets)
        #benign_loss += loss.item()

        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()

        '''
        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            #print('Current benign test loss:', loss.item())
            print("Elapsed Time (Min): ", np.floor((time.time() - start)/60))
        '''
        
        if adversary == 'PGD':
            adv = PGDadversary(inputs, targets)
        else:
            adv = AUTOadversary(inputs, targets)
        adv_outputs = net(adv)
        #loss = criterion(adv_outputs, targets)
        #adv_loss += loss.item()

        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

        '''
        if batch_idx % 10 == 0:
            print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            #print('Current adversarial test loss:', loss.item())
            print("Elapsed Time (Min): ", np.floor((time.time() - start)/60))
        '''

    print('\nTotal benign [',data_type,'] accuarcy:', 100. * benign_correct / total)
    print('Total adversarial [',data_type,'] Accuarcy:', 100. * adv_correct / total)
    #print('Total benign test loss:', benign_loss)
    #print('Total adversarial test loss:', adv_loss)
    print("Elapsed Time (Min): ", np.floor((time.time() - start)/60))



def init_test(args):
    filename = args.filename
    net = ResNet18() #preactresnet.PreActResNet18() #models.resnet18() #
    net = net.to(device)
    #net = torch.nn.DataParallel(net) #, device_ids=[0])
    cudnn.benchmark = True
    
    if args.attack != 'PGD':
        print('\n==> Using Testing Dataset.. (AUTOATTACK)')
        test(net, args.attack, filename, args.val_size, args.batch_size, 'test')
    else :
        print('\n==> Using Validation Dataset..')
        test(net, args.attack, filename, args.val_size, args.batch_size, 'val')
        print('\n==> Using Testing Dataset..')
        test(net, args.attack, filename, args.val_size, args.batch_size, 'test')
        print('\n==> Using Training Dataset..')
        test(net, args.attack, filename, args.val_size, args.batch_size, 'train')


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--val_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument(
        '--filename',
        default='teacher.pth',
        type=str,
        help='name of the model to test')
    parser.add_argument(
        '--attack',
        default='PGD',
        type=str,
        help='name of the attack')
    args = parser.parse_args()

    init_test(args)


if __name__ == '__main__':
    main()
