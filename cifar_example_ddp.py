import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics
import torch.optim as optim
import argparse
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def init_distributed(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
      args.rank = int(os.environ["RANK"])
      args.world_size = int(os.environ['WORLD_SIZE'])
      args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.gpu = 0
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)

    os.environ['MASTER_ADDR']= '127.0.0.1'
    os.environ['MASTER_PORT']= '29500'
    dist.init_process_group(backend='nccl',rank=args.rank,world_size=args.world_size)
    dist.barrier()

def main(args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    init_distributed(args)
    sampler_train = DistributedSampler(trainset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, sampler=sampler_train)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    sampler_test = DistributedSampler(testset, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size, sampler=sampler_test)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    net = Net()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



    for epoch in range(2):  # loop over the dataset multiple times
        
        sampler_train.set_epoch(epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    accuracy = torchmetrics.Accuracy(dist_sync_on_step=True)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network 
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            accuracy.update()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * accuracy.compute()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("--world_size", help="Total number of nodes")

    args = parser.parse_args()
    main(args)