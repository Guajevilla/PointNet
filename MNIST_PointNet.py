import torch
import torchvision
import torchvision.transforms as transforms
from Mmodule import PointNet, res_block
from Mmodule import pic2point
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

# from tensorboardX import SummaryWriter

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 32
# MNIST很容易达到95%以上准确率
trainset = torchvision.datasets.MNIST(root='./Mdata', train=True,
                                        download=True, transform=transform)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./Mdata', train=False,
                                       download=True, transform=transform)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
num_classes = 10
num_batch = len(trainset) / batch_size

net = PointNet(res_block, k=num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# net = nn.DataParallel(net, device_ids=[0, 1])
net = nn.DataParallel(net)

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print(device)
print('Start training')
for epoch in range(1):
    scheduler.step()
    for i, data in enumerate(trainloader, 0):
        points, target = data
        points = pic2point(points)
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()
        # classifier = net.train()
        pred, trans, trans_feat = net(points)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(batch_size)))

        # if i % 10 == 0:
        #     j, data = next(enumerate(testloader, 0))
        #     points, target = data
        #     target = target[:, 0]
        #     points = points.transpose(2, 1)
        #     points, target = points.to(device), target.to(device)
        #     classifier = classifier.eval()
        #     pred, _, _ = classifier(points)
        #     loss = F.nll_loss(pred, target)
        #     pred_choice = pred.data.max(1)[1]
        #     correct = pred_choice.eq(target.data).cpu().sum()
        #     print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    # torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

total_correct = 0
total_testset = 0
with torch.no_grad():
    # for i, data in tqdm(enumerate(testloader, 0)):
    for i, data in enumerate(testloader, 0):
        points, target = data
        points = pic2point(points)
        points, target = points.to(device), target.to(device)
        classifier = net.eval()
        pred, _, _ = net(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
