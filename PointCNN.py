import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;

    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    # view_shape = [B, 1]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    # repeat_shape = [1, S]
    repeat_shape[0] = 1
    # 其实就是想把[0, B-1]的range一维向量变为与idx维度相同的[0, B-1]重复S次矩阵,这样维度相同切片时才能一一对应
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # 每一次循环找的是batch个最远点: farthest [B, ]
    # distance里存的是每一次操作以后B*N个点相对于所有备选点集的最远距离
    # mask存在是因为点到点集的距离是点到点集中每个点的距离中的最小值
    for i in range(npoint):
        centroids[:, i] = farthest
        # batch_indices [B ], farthest [B ],输出 [B, 1, 3]
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def find_k_neighbor(pts, xyz, K, D):
    """
    Find K nearest neighbor points
    :param pts: represent points(B, P, C)
    :param xyz: original points (B, N, C)
    :param K:
    :param D: Dilation rate
    :return group_pts: K neighbor points(B, P, K, C)
    """
    device = pts.device
    B, P, _ = pts.size()
    sqrdists = square_distance(pts, xyz)    # (B,P,N)
    # k_ind = torch.topk(sqrdists, k=K, dim=-1, largest=False)[1]
    k_ind = sqrdists.sort(dim=-1)[1][:, :, :K*D]  # (B,P,K*D)
    rand_columns = torch.randperm(K*D, dtype=torch.long)[:K].to(device)
    k_ind = k_ind[:, :, rand_columns]
    group_pts = index_points(xyz, k_ind)        # (B,P,K,C)

    return group_pts, k_ind


class xconv(nn.Module):
    def __init__(self, in_channel, lift_channel, out_channel, P, K, D=1, sampling='fps'):
        """
        :param in_channel: Input channel of the points' features
        :param lift_channel: Lifted channel C_delta
        :param out_channel:
        :param P: P represent points
        :param K: K neighbors to operate
        :param D: Dilation rate
        """
        super(xconv, self).__init__()
        self.P = P
        self.K = K
        self.D = D
        self.sampling = sampling
        # Input should be (B, 3, P, K)
        self.MLP_delta = nn.Sequential(
            nn.Conv2d(3, lift_channel, kernel_size=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(lift_channel),
            nn.Conv2d(lift_channel, lift_channel, kernel_size=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(lift_channel)
        )
        # Input should be (B, 3, P, K)
        self.MLP_X = nn.Sequential(
            nn.Conv2d(3, K, kernel_size=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(K),
            nn.Conv2d(K, K, kernel_size=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(K),
            # nn.Conv2d(K, K, kernel_size=1),
            # nn.BatchNorm2d(K)
        )
        nn.init.xavier_uniform_(self.MLP_X[0].weight)
        nn.init.xavier_uniform_(self.MLP_X[3].weight)
        # nn.init.xavier_uniform_(self.MLP_X[6].weight)

        self.MLP_feat0 = nn.Sequential(
            nn.Conv2d(K, K, kernel_size=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(K),
            nn.Conv2d(K, 1, kernel_size=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(1)
        )
        self.MLP_feat1 = nn.Sequential(
            nn.Conv1d(lift_channel+in_channel, out_channel, kernel_size=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, pts, fts):
        """
        :param x: (rep_pt, pts, fts) where
          - pts: Regional point cloud (B, N, 3)
          - fts: Regional features (B, N, C)
        :return: Features aggregated into point rep_pt.
        """
        B, N, _ = pts.size()
        if self.P == -1:
            self.P = N
            represent_pts = pts
            pre_ind = torch.arange(0, N, step=1).unsqueeze(0).repeat((B, 1)).to(pts.device)
        else:
            if self.sampling == 'fps':
                pre_ind = farthest_point_sample(pts, self.P)        # (B, P)
                represent_pts = index_points(pts, pre_ind)      # .view(B, self.P, 1, 3)
            else:
                # idx = np.random.choice(pts.size()[1], self.P, replace=False).tolist() # .to(pts.device)
                # represent_pts = pts[:, idx, :]
                pre_ind = torch.randint(low=0, high=N, size=(B, self.P), dtype=torch.long).to(pts.device)
                represent_pts = index_points(pts, pre_ind)

        group_pts, k_ind = find_k_neighbor(represent_pts, pts, self.K, self.D)  # (B, P, K, 3), (B, P, K)
        center_pts = torch.unsqueeze(represent_pts, dim=2)      # (B, P, 1, 3)
        group_pts = group_pts - center_pts       # (B, P, K, 3)
        # MLP得到fts_lifted
        group_pts = group_pts.permute(0,3,1,2)
        fts_lifted = self.MLP_delta(group_pts)      # (B, C_delta, P, K)
        if fts is not None:
            # TODO: ind会越界
            # center_feat = index_points(fts, pre_ind)  # (B, P, C_in)
            # group_fts = index_points(center_feat, k_ind)  # (B, P, K, C_in)
            group_fts = index_points(fts, k_ind)

            group_fts = group_fts.permute(0,3,1,2)
            feat = torch.cat((fts_lifted, group_fts), 1)  # (B, C_delta + C_in, P, K)
        else:
            feat = fts_lifted
        # X阵
        X = self.MLP_X(group_pts).permute(0,2,3,1)  # (B, P, K, K)

        X = X.contiguous().view(B*self.P, self.K, self.K)
        feat = feat.permute(0,2,3,1).contiguous().view(B*self.P, self.K, -1)
        feat = torch.bmm(X, feat).view(B, self.P, self.K, -1).permute(0,2,1,3)  # (B, K, P, C_delta + C_in)
        feat = self.MLP_feat0(feat).squeeze(1)           # (B, self.P, C_delta + C_in)
        feat = feat.permute(0,2,1)                      # (B, C_delta + C_in, self.P)
        feat = self.MLP_feat1(feat).permute(0,2,1)      # (B, self.P, C_out)

        return represent_pts, feat      # (B, P, 3), (B, P, C_out)


class PointCNN_cls(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        # X_conv1
        C_out = 16*3
        C_delta = C_out // 2
        self.x_conv1 = xconv(0, C_delta, C_out, P=-1, K=8)# , sampling='rand')

        # X_conv2
        C_in = C_out
        C_out = 32*3
        C_delta = C_in // 4
        self.x_conv2 = xconv(C_in, C_delta, C_out, P=384, K=12, D=2)# , sampling='rand')

        # X_conv3
        C_in = C_out
        C_out = 64*3
        C_delta = C_in // 4
        self.x_conv3 = xconv(C_in, C_delta, C_out, P=128, K=16, D=2)# , sampling='rand')

        # X_conv4
        C_in = C_out
        C_out = 128*3
        C_delta = C_in // 4
        self.x_conv4 = xconv(C_in, C_delta, C_out, P=-1, K=16, D=3)# , sampling='rand')

        self.fc = nn.Sequential(
            nn.Conv1d(C_out, 64 * 3, 1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64 * 3),
            nn.Conv1d(64 * 3, num_class, 1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(num_class),
            nn.Dropout(p=0.8)
        )

    def forward(self, x):       # (B, N, 3)
        pts, fts = self.x_conv1(x, None)
        pts, fts = self.x_conv2(pts, fts)
        pts, fts = self.x_conv3(pts, fts)
        pts, fts = self.x_conv4(pts, fts)   # (B, 128, 3), (B, 128, 384)

        fts = fts.permute(0, 2, 1)
        fts = self.fc(fts)          # (B, num_class, 128)
        logits = torch.mean(fts, dim=-1)

        return logits
