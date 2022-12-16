import torch
from torchvision import datasets
import torchvision.transforms as transforms

MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def get_loader(val_size=5000, batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        
        #transforms.RandomRotation(10),
        #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        #transforms.ColorJitter(brghtness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        #transforms.Normalize(MEAN, STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(MEAN, STD),
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    
    trainset, valset = torch.utils.data.random_split(trainset, [50000 - val_size, val_size],
                                                         torch.Generator().manual_seed(35))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size, shuffle=True, num_workers=0)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size *2, shuffle=False, num_workers=0)
    
    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size, shuffle=False, num_workers=0)
    print("> Datasets ready >")
    return trainloader, valloader, testloader
