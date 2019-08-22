import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 16

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)


class res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.self_conv = nn.Conv2d(in_channels, out_channels, 1, padding=0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.bn(out))
        out = self.relu(self.bn(self.conv2(out)))
        out = self.relu(self.bn(self.conv2(out)))
        x = self.self_conv(x)
        return out + x


class ResNet(nn.Module):
    def __init__(self, block):
        super(ResNet, self).__init__()
        # self.conv1 = block(3, 32)
        # self.conv1_2 = block(32, 32)
        # self.conv2 = block(32, 64)
        # self.conv2_2 = block(64, 64)
        # self.conv3 = block(64, 128)
        # self.conv3_2 = block(128, 128)

        self.conv1 = block(1, 64)
        self.conv1_2 = block(64, 64)
        self.conv2 = block(64, 128)
        self.conv2_2 = block(128, 128)
        self.conv3 = block(128, 256)
        self.conv3_2 = block(256, 256)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv8 = block(256, 256)
        self.global_pool = nn.Conv2d(256, 10, 1)
        self.dropout = nn.Dropout2d()

    # resblock不能写在这里，因为定义在其他函数(非__init__)下的卷积等，权重参数不会跟着移到cuda上
    # def res_block(self, x, in_channels, out_channels, is_pool=False):
    #     out = nn.Conv2d(in_channels, out_channels, 3, padding=1)(x)
    #     out = nn.BatchNorm2d(out_channels)(out)
    #     out = nn.ReLU(out)
    #     out = nn.Conv2d(out_channels, out_channels, 3, padding=1)(out)
    #     out = nn.BatchNorm2d(out_channels)(out)
    #     x = nn.Conv2d(in_channels, out_channels, 1, padding=0)(x)
    #     out = nn.ReLU(out) + x
    #     return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(self.conv1_2(self.conv1(x)))
        x = self.pool(self.conv2_2(self.conv2(x)))
        x = self.pool(self.conv3_2(self.conv3(x)))

        # 全卷积
        x = self.pool(self.conv8(x))
        x = self.dropout(x)
        x = self.global_pool(self.pool(x))
        x = x.view(-1, 10)
        return x


net = ResNet(res_block)
# print(net)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)
net = nn.DataParallel(net)


criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))

print('Start training')
writer = SummaryWriter('runs')
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # inputs, labels = data[0].cuda(), data[1].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # 使用交叉熵损失函数的时候会自动把label转化成onehot
        # 一句话将标签转化为one-hot:
        # label_onehot = torch.eye(10).index_select(0, labels)
        # 实验证明cross entropy输入one-hot报错，但MSE需要输入one-hot
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i*batch_size % 6400 == 6384:    # print every 8192 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            # print(inputs.shape)
            running_loss = 0.0
    writer.add_scalar('loss', loss, epoch)
    writer.add_scalar('running_loss', running_loss, epoch)
writer.add_graph(net, (inputs,))
writer.close()
print('Finished Training')

correct = 0
total = 0   # total = 10000
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
