from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from tqdm import tqdm
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'epsilon': 8.0 / 255,
    'num_steps': 10,
    'step_size': 2.0 / 255,
}
KL_loss = nn.KLDivLoss()
XENT_loss = nn.CrossEntropyLoss()

class AttackPGD(nn.Module):
    def __init__(self, attack_model, config):
        super(AttackPGD, self).__init__()
        self.attack_model = attack_model
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']

    def forward(self, inputs, targets):
        x = inputs.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(self.attack_model(x), targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        return self.attack_model(x), x
    

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in args.lr_schedule:
        lr *= args.lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr   
    
def train(basic_net, teacher_net, epoch, optimizer, trainloader):
    print("=> Train Epoch:", epoch)
    basic_net.train()
    attack_model = AttackPGD(basic_net, config)
    train_loss = 0
    iterator = tqdm(trainloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)     
        optimizer.zero_grad()
        outputs, pert_inputs = attack_model(inputs, targets)
        teacher_outputs = teacher_net(inputs)
        basic_outputs = basic_net(inputs)
        loss = args.alpha*args.temp*args.temp*KL_loss(F.log_softmax(basic_outputs/args.temp, dim=1),
                                                      F.softmax(outputs/args.temp, dim=1)) + (1.0-args.alpha)*XENT_loss(basic_outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        iterator.set_description(str(loss.item()))
    if (epoch+1)%args.save_period == 0:
        state = {
            'net': basic_net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint/'+args.dataset+'/'+args.output+'/'):
            os.makedirs('checkpoint/'+args.dataset+'/'+args.output+'/', )
        torch.save(state, './checkpoint/'+args.dataset+'/'+args.output+'/epoch='+str(epoch)+'.t7')
    print('Mean Training Loss:', train_loss/len(iterator))
    return train_loss

def test(basic_net, epoch, optimizer, testloader):
    print("=> Test Epoch:", epoch)
    basic_net.eval()
    attack_model = AttackPGD(basic_net, config)
    adv_correct = 0
    natural_correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_outputs, pert_inputs = attack_model(inputs, targets)
            natural_outputs = basic_net(inputs)
            _, adv_predicted = adv_outputs.max(1)
            _, natural_predicted = natural_outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()
            iterator.set_description(str(adv_predicted.eq(targets).sum().item()/targets.size(0)))
    robust_acc = 100.*adv_correct/total
    natural_acc = 100.*natural_correct/total
    print('Natural acc:', natural_acc)
    print('Robust acc:', robust_acc)
    return natural_acc, robust_acc

def main(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        num_classes = 100

    print('==> Building model..'+args.model)
    if args.model == 'MobileNetV2':
        basic_net = MobileNetV2(num_classes=num_classes)
    elif args.model == 'WideResNet':
        basic_net = WideResNet(num_classes=num_classes)
    elif args.model == 'ResNet18':
        basic_net = ResNet18() #ResNet18(num_classes=num_classes)
    basic_net = basic_net.to(device)
    if args.teacher_path != '':
        if args.teacher_model == 'MobileNetV2':
            teacher_net = MobileNetV2(num_classes=num_classes)
        elif args.teacher_model == 'WideResNet':
            teacher_net = WideResNet(num_classes=num_classes)
        elif args.teacher_model == 'ResNet18':
            teacher_net = ResNet18() #ResNet18(num_classes=num_classes)
        teacher_net = teacher_net.to(device)
        for param in teacher_net.parameters():
            param.requires_grad = False


    print('==> Loading teacher..')
    if device == 'cuda':
        teacher_net = torch.nn.DataParallel(teacher_net, device_ids=[0])
        cudnn.benchmark = True
    load_path = "./checkpoint/"
    checkpoint = torch.load(load_path + 'NT_.pth',
                        map_location=lambda storage, loc: storage.cuda(0))['net']
    teacher_net.load_state_dict(checkpoint)
    teacher_net.eval()
    print("preparing attack with student model..")
    lr = args.lr
    optimizer = optim.SGD(basic_net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    for epoch in range(args.epochs):
        print('epoch:',epoch)
        adjust_learning_rate(optimizer, epoch, lr)
        train_loss = train(basic_net, teacher_net, epoch, optimizer, trainloader)
        if (epoch+1)%args.val_period == 0:
            natural_val, robust_val = test(basic_net, epoch, optimizer, testloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
    parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs for training')
    parser.add_argument('--output', default = './output/', type=str, help='output subdirectory')
    parser.add_argument('--model', default = 'ResNet18', type = str, help = 'student model name')
    parser.add_argument('--teacher_model', default = 'ResNet18', type = str, help = 'teacher network model')
    parser.add_argument('--teacher_path', default = './chekpoint/', type=str, help='path of teacher net being distilled')
    parser.add_argument('--temp', default=30.0, type=float, help='temperature for distillation')
    parser.add_argument('--val_period', default=1, type=int, help='print every __ epoch')
    parser.add_argument('--save_period', default=1, type=int, help='save every __ epoch')
    parser.add_argument('--alpha', default=0.9, type=float, help='weight for sum of losses')
    parser.add_argument('--dataset', default = 'CIFAR10', type=str, help='name of dataset')
    args = parser.parse_args()
    main(args)
