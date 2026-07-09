import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score
import torch
# 全局默认精度：必须与 DataLoader 中张量精度一致，否则会出现
# "Input type (FloatTensor) and weight type (DoubleTensor) should be the same"
# 训练常用 float32；若改为 float64，请同时把 batch 转为 float64（data = data.to(dtype=TORCH_DTYPE)）
TORCH_DTYPE = torch.float32
torch.set_default_dtype(TORCH_DTYPE)
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
import random
from tqdm import tqdm
from testbench.old.unet import UNet


def set_seed(seed=42):
    """
    设置随机数种子，确保实验可重复
    
    Args:
        seed: 随机数种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA操作的确定性（可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_gaussian_weight_matrix(size=13, sigma=None):
    """
    创建二维高斯权重矩阵
    
    Args:
        size: 矩阵大小（默认13×13）
        sigma: 高斯函数的标准差，如果为None则使用size/6作为默认值
    
    Returns:
        gaussian_matrix: (size, size)的numpy数组，中心值最大，边缘值最小
    """
    if sigma is None:
        sigma = size / 6.0  # 默认sigma，使得边缘值约为中心的1%
    
    # 创建坐标网格
    center = (size - 1) / 2.0
    x = np.arange(size) - center
    y = np.arange(size) - center
    X, Y = np.meshgrid(x, y)
    
    # 计算二维高斯函数
    gaussian_matrix = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # 归一化，使得中心值为1
    gaussian_matrix = gaussian_matrix / gaussian_matrix.max()
    
    return gaussian_matrix


def loadData(i):
    # 读入数据
    data = sio.loadmat(f'datasets/GM/GM0{i}.mat')['img']
    labels = sio.loadmat(f'datasets/GM/GM0{i}.mat')['map']

    return data, labels

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = False):

    # 给 X 做 padding
    # 计算margin：对于偶数windowSize，需要特殊处理
    if windowSize % 2 == 0:
        # 偶数windowSize（如16）：margin = windowSize // 2
        # 提取 [r-margin:r+margin] 会得到 2*margin = windowSize 个元素（因为Python切片是左闭右开）
        margin = windowSize // 2
    else:
        # 奇数windowSize（如15）：margin = (windowSize - 1) // 2
        # 提取 [r-margin:r+margin+1] 会得到 2*margin+1 = windowSize 个元素
        margin = (windowSize - 1) // 2
    
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if windowSize % 2 == 0:
                # 偶数：提取 [r-margin:r+margin]，得到 windowSize 个元素
                patch = zeroPaddedX[r - margin:r + margin, c - margin:c + margin]
            else:
                # 奇数：提取 [r-margin:r+margin+1]，得到 windowSize 个元素
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


def calculate_statistics(values):
    """
    计算均值和标准差（不确定度）
    
    Args:
        values: 数值列表
    
    Returns:
        mean: 均值
        std: 标准差（不确定度）
    """
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # 样本标准差（除以n-1）
    return mean, std

BATCH_SIZE_TRAIN = 64

def create_data_loader(i):
    # 地物类别
    # class_num = 16
    # 读入数据
    X, y = loadData(i)
    # 用于测试样本的比例
    test_ratio = 0.99
    # 每个像素周围提取 patch 的尺寸
    patch_size = 16
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 30

    # 数据预处理（静默处理，不输出详细信息）
    X_pca = applyPCA(X, numComponents=pca_components)
    X_patches, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    
    # 创建高斯权重矩阵 (13, 13)
    gaussian_weight = create_gaussian_weight_matrix(size=patch_size)
    
    # 应用高斯权重：对每个patch的每个波段都乘以高斯权重矩阵
    # X_patches形状: (N, 13, 13, 30)
    # gaussian_weight形状: (13, 13)
    # 使用广播机制：(N, 13, 13, 30) * (13, 13) -> (N, 13, 13, 30)
    X_patches = X_patches * gaussian_weight[:, :, np.newaxis]  # 广播到最后一个维度
    
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_patches, y_all, test_ratio)
    
    # 改变形状并转置
    X = X_patches.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    
    X = torch.FloatTensor(X)
    Xtrain = torch.FloatTensor(Xtrain)
    Xtest = torch.FloatTensor(Xtest)
    
    X = X.permute(0, 4, 3, 1, 2).contiguous()
    Xtrain = Xtrain.permute(0, 4, 3, 1, 2).contiguous()
    Xtest = Xtest.permute(0, 4, 3, 1, 2).contiguous()

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                              )

    return train_loader, test_loader, all_data_loader, y

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len

class UNetClassifier(nn.Module):
    """
    UNet分类器：将UNet特征提取器与分类头结合
    用于二分类任务
    """
    def __init__(self, unet, num_classes=2):
        super(UNetClassifier, self).__init__()
        self.unet = unet
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) -> (B, C, 1, 1)
        # 获取UNet的输出通道数
        # 支持UNet_SmallSize和UNetWithTopKAttention两种类型
        if hasattr(unet, 'unet'):
            # UNetWithTopKAttention类型
            unet_out_channels = unet.unet.stage_out.out_channels
        elif hasattr(unet, 'out_channels'):
            # UNet类型（新版本，直接有out_channels属性）
            unet_out_channels = unet.out_channels
        elif hasattr(unet, 'stage_out') and hasattr(unet.stage_out, 'out_channels'):
            # UNet_SmallSize类型或其他有stage_out.out_channels的类型
            unet_out_channels = unet.stage_out.out_channels
        else:
            # 兜底方案：尝试从deep_unet获取
            unet_out_channels = unet.deep_unet.stage_out.out_channels
        self.norm = nn.LayerNorm(unet_out_channels)
        self.fc = nn.Linear(unet_out_channels, num_classes)
    
    def forward(self, x):
        # 输入: (B, 1, C, H, W)，例如 (B, 1, 25, 13, 13)
        B, _, C, H, W = x.size()
        # 去掉第二维
        x = x.squeeze(1)  # (B, C, H, W)，例如 (B, 25, 13, 13)
        
        # 经过UNet提取特征
        feat = self.unet(x)  # (B, out_channels, H, W)，例如 (B, 25, 13, 13)
        
        # 分类头：全局平均池化 + 全连接
        feat = self.avg_pool(feat).view(B, -1)  # (B, out_channels)
        feat = self.norm(feat)
        out = self.fc(feat)  # (B, num_classes)
        
        return out

def train(train_loader, epochs, pbar=None):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    # 创建UNet模型，用于分类任务
    # 输入通道：30 (PCA后的通道数)
    # 输出通道：30 (特征图)
    # 然后通过分类头映射到2类
    unet = UNet(
        in_channels=30, 
        out_channels=30, 
        num_filters=32  # 保持与原来一致
    ).to(device)
    
    # 选择分类头类型：
    # 1. UNetClassifier: 原始分类头（全局平均池化）
    # 2. MultiScaleClassifier: 多尺度分类头（AvgPool + MaxPool融合）
    # 3. MultiScaleAttentionClassifier: 多尺度注意力分类头（多尺度 + 通道注意力）
    # 4. SimplifiedMultiScaleClassifier: 简化版多尺度分类头（推荐，轻量级，避免过拟合）
    use_multiscale = False  # 关闭多尺度分类头，使用最初的分类头
    use_attention = False
    use_simplified = False
    

    net = UNetClassifier(unet, num_classes=2).to(device)
    
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器（降低学习率，更稳定训练）
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    
    # 开始训练
    best_loss = float('inf')
    
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device=device, dtype=TORCH_DTYPE)
            target = target.to(device)
            
            # 正向传播 + 反向传播 + 优化
            outputs = net(data)
            loss = criterion(outputs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新损失
            epoch_loss += loss.item()
            num_batches += 1
            
            # 更新进度条（显示详细信息）
            if pbar is not None:
                avg_loss = epoch_loss / num_batches
                pbar.set_postfix({
                    '阶段': '训练',
                    'Epoch': f'{epoch+1}/{epochs}',
                    'Loss': f'{loss.item():.4f}',
                    'Best': f'{best_loss:.4f}' if best_loss != float('inf') else 'N/A'
                })
                pbar.update(1)
        
        # 计算epoch平均损失
        epoch_avg_loss = epoch_loss / num_batches
        
        # 更新最佳损失
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss

    return net, device

def test(device, net, test_loader, pbar=None):
    count = 0
    # 模型测试（与 IP_train.py 一致：AUC 使用正类 softmax 概率）
    with torch.no_grad():
        net.eval()
        y_pred_test = 0
        y_test = 0

        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device=device, dtype=TORCH_DTYPE)
            logits = net(inputs)
            prob_pos = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            pred_cls = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels_np = labels.detach().cpu().numpy()

            if count == 0:
                y_pred_test = pred_cls
                y_score_test = prob_pos
                y_test = labels_np
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, pred_cls))
                y_score_test = np.concatenate((y_score_test, prob_pos))
                y_test = np.concatenate((y_test, labels_np))

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({'阶段': '测试', 'Batch': f'{batch_idx+1}/{len(test_loader)}'})

    return y_pred_test, y_test, y_score_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test, y_score=None):
    y_test = np.asarray(y_test).ravel()
    y_pred_test = np.asarray(y_pred_test).ravel()
    recall = recall_score(y_test, y_pred_test, average='binary')
    if y_score is not None:
        auc = roc_auc_score(y_test, np.asarray(y_score).ravel())
    else:
        auc = roc_auc_score(y_test, y_pred_test)

    return recall, auc
def run_single_experiment(i, seed=None, run_idx=None, num_runs=None):
    """
    运行单次实验
    
    Args:
        i: 数据集编号
        seed: 随机数种子（如果为None，则不设置）
        run_idx: 当前运行索引
        num_runs: 总运行次数
    
    Returns:
        recall: Recall值
        auc: AUC值
        train_time: 训练时间
        test_time: 测试时间
    """
    if seed is not None:
        set_seed(seed)
    
    train_loader, test_loader, all_data_loader, y_all = create_data_loader(i)
    
    # 计算总batch数（训练 + 测试）
    epochs = 100
    total_train_batches = epochs * len(train_loader)
    total_test_batches = len(test_loader)
    total_batches = total_train_batches + total_test_batches
    
    # 创建单个进度条（使用leave=True和position=0确保原地更新）
    run_info = f"[{run_idx+1}/{num_runs}]" if run_idx is not None else ""
    pbar = tqdm(total=total_batches, desc=f"GM0{i} {run_info}", unit="batch", ncols=150,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                position=0, leave=True, mininterval=0.5)
    
    # 训练阶段
    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=epochs, pbar=pbar)
    toc1 = time.perf_counter()
    train_time = toc1 - tic1
    
    # 测试阶段
    tic2 = time.perf_counter()
    y_pred_test, y_test, y_score_test = test(device, net, test_loader, pbar=pbar)
    toc2 = time.perf_counter()
    test_time = toc2 - tic2

    pbar.close()

    # 评价指标（AUC 与 IP_train 一致：基于正类概率）
    recall, auc = acc_reports(y_test, y_pred_test, y_score_test)
    
    return recall, auc, train_time, test_time


def run(i, num_runs=5, base_seed=2025):
    """
    运行多次实验并计算统计结果
    
    Args:
        i: 数据集编号
        num_runs: 运行次数（默认5次）
        base_seed: 基础随机数种子
    """
    print(f"\n[GM0{i}] 开始处理（运行{num_runs}次）...")
    
    recalls = []
    aucs = []
    train_times = []
    test_times = []
    
    for run_idx in range(num_runs):
        seed = base_seed + run_idx  # 每次使用不同的种子
        
        recall, auc, train_time, test_time = run_single_experiment(i, seed=seed, run_idx=run_idx, num_runs=num_runs)
        
        recalls.append(recall)
        aucs.append(auc)
        train_times.append(train_time)
        test_times.append(test_time)
        
        # 打印当前运行结果
        print(f"  Run {run_idx+1}: Recall={recall:.4f}, AUC={auc:.4f}")
    
    # 计算统计量
    recall_mean, recall_std = calculate_statistics(recalls)
    auc_mean, auc_std = calculate_statistics(aucs)
    train_time_mean, train_time_std = calculate_statistics(train_times)
    test_time_mean, test_time_std = calculate_statistics(test_times)
    
    # 计算SOTA值（5次训练中的最佳值）
    recall_sota = max(recalls)
    auc_sota = max(aucs)
    recall_sota_idx = recalls.index(recall_sota) + 1  # 找到最佳Recall的运行索引
    auc_sota_idx = aucs.index(auc_sota) + 1  # 找到最佳AUC的运行索引
    
    # 显示结果（简洁版）
    print(f"[GM0{i}] 完成 | {num_runs}次运行结果:")
    print(f"  Recall: {recall_mean:.4f} ± {recall_std:.4f} | SOTA: {recall_sota:.4f} (Run {recall_sota_idx})")
    print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f} | SOTA: {auc_sota:.4f} (Run {auc_sota_idx})")
    print(f"  训练时间: {train_time_mean/60:.1f} ± {train_time_std/60:.1f} min")
    print(f"  测试时间: {test_time_mean:.1f} ± {test_time_std:.1f} s")
    
    # 保存结果到文件
    recall_and_auc = f"cls_result/3_22_float64.txt"
    with open(recall_and_auc, 'a') as f:
        f.write(f"\nGM0{i} ({num_runs}次运行):\n")
        f.write(f"  Recall: {recall_mean:.4f} ± {recall_std:.4f} (范围: [{min(recalls):.4f}, {max(recalls):.4f}])\n")
        f.write(f"  Recall SOTA: {recall_sota:.4f} (Run {recall_sota_idx})\n")
        f.write(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f} (范围: [{min(aucs):.4f}, {max(aucs):.4f}])\n")
        f.write(f"  AUC SOTA: {auc_sota:.4f} (Run {auc_sota_idx})\n")
        f.write(f"  训练时间: {train_time_mean:.1f} ± {train_time_std:.1f} s\n")
        f.write(f"  测试时间: {test_time_mean:.1f} ± {test_time_std:.1f} s\n")
        f.write(f"  详细结果: Recall={recalls}, AUC={aucs}\n")
        f.write("="*50 + "\n")


    # # get_cls_map.get_cls_map(net, device, all_data_loader, y_all)
if __name__ == '__main__':
    print("="*70)
    print("开始处理所有数据集 (GM01-GM08)")
    print("="*70)
    
    # 实验配置
    NUM_RUNS = 5  # 每个数据集运行5次
    BASE_SEED = 2025  # 基础随机数种子
    NUM_DATASETS = 8  # 数据集数量
    
    print(f"实验配置: 每个数据集运行{NUM_RUNS}次，基础种子={BASE_SEED}")
    print("="*70)
    
    total_start_time = time.perf_counter()
    
    # 总体进度条
    datasets_pbar = tqdm(range(1, 8 + 1), desc="总体进度", unit="数据集", ncols=150,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for i in datasets_pbar:
        datasets_pbar.set_description(f"处理数据集 GM0{i}")
        run(i, num_runs=NUM_RUNS, base_seed=BASE_SEED)
        datasets_pbar.set_postfix({'已完成': f'GM0{i}'})
    
    datasets_pbar.close()
    
    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    
    print("="*70)
    print(f"全部完成！总耗时: {total_time/60:.1f} 分钟 ({total_time:.1f} 秒)")
    print("="*70)


