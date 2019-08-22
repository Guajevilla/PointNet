import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

################################################
#
#
# 用于魔改普通数据集如cifar10等用于PointNet的module
#
#
################################################


# 针对4维B*C*W*H变为B*4*(C*W*H)的三维点云（3表示3维坐标+1维原始数据）
def pic2point(pic_data):
    b, c, w, h = pic_data.size()
    x, y, z = np.where(np.ones([c, w, h]) > 0)
    coordinates = np.vstack((x, y, z))
    coordinates = coordinates / (coordinates.max(axis=1).reshape(-1, 1)+0.0000001)
    coordinates = coordinates.reshape(1, 3, -1).repeat(b, 0)

    point_data = pic_data.view(b, 1, c*w*h)
    point_data = np.concatenate((point_data, coordinates), 1)

    return Variable(torch.from_numpy(point_data.astype(np.float32)))


# TODO: 设想是将下面的尺寸根据输入数据集进行魔改
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.channel = channel
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, channel*channel)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.channel).flatten().astype(np.float32))).view(1, self.channel*self.channel).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.channel, self.channel)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, padding=0)
        self.bn = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1, padding=0)
        self.self_conv = nn.Conv1d(in_channels, out_channels, 1, padding=0)

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
        x = self.self_conv(x)
        return out + x


class PointNet(nn.Module):
    def __init__(self, block, k=2):
        super(PointNet, self).__init__()
        self.k = k
        self.stn3d = STN3d(4)
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        # self.conv1 = block(4, 64)
        # self.conv2 = block(64, 64)
        self.stnkd = STNkd(64)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 512, 1)
        self.conv7 = torch.nn.Conv1d(512, 1024, 1)
        # self.conv3 = block(64, 64)
        # self.conv4 = block(64, 128)
        # self.conv5 = block(128, 256)
        # self.conv6 = block(256, 512)
        # self.conv7 = block(512, 1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(1024)

        self.bn8 = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout2d()

    def forward(self, x):
        # channel = x.size(dim=1)
        trans = self.stn3d(x)
        x = x.transpose(2, 1)       # 变成B*N*3以满足矩阵计算x*trans
        x = torch.bmm(x, trans)     # 实现bath间的矩阵乘法，不改变batch维度,计算结果维度为B*N*3
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        trans_feat = self.stnkd(x)
        x = x.transpose(2, 1)       # 变成B*N*3以满足矩阵计算x*trans
        x = torch.bmm(x, trans_feat)     # 实现bath间的矩阵乘法，不改变batch维度,计算结果维度为B*N*3
        x = x.transpose(2, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        # x = F.relu(self.conv7(x))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)

        x = F.relu(self.bn8(self.fc1(x)))
        x = F.relu(self.bn9(self.fc2(x)))
        x = self.dropout(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1,2)))
    return loss
