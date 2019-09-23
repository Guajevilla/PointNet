# PointNet
Vanilla realization of PointNet
## Mmodule
将常用图片数据集转换为点云的格式用于比较模型

### 一个临时函数
def cosine_similarity(feature):
    """
    Input:
        feature: source points, [B, N, C]
    Output:
        dist: per-point square distance, [B, N, N]
    """
    B, N, C = feature.shape
    feat_ = torch.matmul(feature, feature.permute(0, 2, 1))  # [B, N, N]
    norm = torch.sum(feature ** 2, -1).view(B, N, 1)
    norm = torch.matmul(norm, norm.permute(0, 2, 1))
    res = torch.div(feat_, norm)
    return res
