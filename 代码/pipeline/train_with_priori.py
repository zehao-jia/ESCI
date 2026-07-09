import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, \
    roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from operator import truediv
import os
import sys
import time
import importlib
import random
from datetime import datetime
from typing import Optional, Callable, Any
from tqdm import tqdm

# Make sibling modules (get_cls_map, testbench) importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import get_cls_map

TORCH_DTYPE = torch.float32
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# ========== 实验会话标识（同步 TensorBoard / 结果 txt）==========
EXPERIMENT_SESSION_SLUG = os.environ.get("IP_TRAIN_SESSION_SLUG", "5_8_baseline22")
DEFAULT_TRAIN_LOG_DIR = os.path.join("runs", EXPERIMENT_SESSION_SLUG)
# 未传入 per-model 路径时，run() 仍追加到此文件（兼容旧用法）
LEGACY_RESULT_SUMMARY_TXT = os.path.join("cls_result", f"{EXPERIMENT_SESSION_SLUG}.txt")


def _tensorboard_log_torch_graph_relaxed(
        writer: SummaryWriter,
        model: torch.nn.Module,
        example_args: torch.Tensor,
        verbose: bool = False,
) -> None:
    """
    将 PyTorch 模型结构写入 TensorBoard。

    与 ``writer.add_graph`` 等价，但 ``torch.jit.trace(..., check_trace=False)``。
    ``nn.MultiheadAttention`` 在部分 PyTorch 版本下两次校验前向可能分别走
    ``scaled_dot_product_attention`` 与 ``_native_multi_head_attention``，触发
    ``TracingCheckError``；关闭校验仅影响图结构一致性检查，不影响训练。
    """
    import torch.utils.tensorboard._pytorch_graph as tb_pg
    from tensorboard.compat.proto.config_pb2 import RunMetadata
    from tensorboard.compat.proto.graph_pb2 import GraphDef
    from tensorboard.compat.proto.step_stats_pb2 import DeviceStepStats, StepStats
    from tensorboard.compat.proto.versions_pb2 import VersionDef

    with tb_pg._set_model_to_eval(model):
        trace = torch.jit.trace(
            model,
            example_args,
            strict=False,
            check_trace=False,
        )
        graph = trace.graph
        torch._C._jit_pass_inline(graph)
    if verbose:
        print(graph)
    list_of_nodes = tb_pg.parse(graph, trace, example_args)
    stepstats = RunMetadata(
        step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")])
    )
    graph_def = GraphDef(node=list_of_nodes, versions=VersionDef(producer=22))
    writer._get_file_writer().add_graph((graph_def, stepstats))


def result_summary_path_for_model(model_name: str) -> str:
    """
    按模型名生成 cls_result 下的汇总 txt 路径，包含日期和模型名。

    格式: YYYYMMDD_<model_name>.txt
    会对 model_name 做简单文件名净化（避免路径分隔符与 Windows 非法字符）。
    """
    name = (model_name or "model").strip()
    invalid = '<>:"/\\|?*'
    safe = "".join("_" if ch in invalid or ord(ch) < 32 else ch for ch in name.replace(os.sep, "_"))
    safe = safe.strip(" .") or "model"

    # 获取当前日期
    current_date = datetime.now().strftime("%Y%m%d")

    return os.path.join("cls_result", f"{current_date}_{safe}.txt")


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
    # 关闭确定性以启用 cuDNN 最快卷积/注意力内核
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
    gaussian_matrix = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    # 归一化，使得中心值为1
    gaussian_matrix = gaussian_matrix / gaussian_matrix.max()

    return gaussian_matrix


def loadData(i):
    mat = sio.loadmat(f"datasets/GM/GM0{i}.mat")
    return np.asarray(mat["img"], dtype=np.float32), mat["map"]


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2])).astype(np.float32, copy=False)
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX).astype(np.float32, copy=False)
    return np.reshape(newX, (X.shape[0], X.shape[1], numComponents))


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    X = np.asarray(X, dtype=np.float32)
    newX = np.zeros(
        (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]),
        dtype=np.float32,
    )
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=False):
    """
    使用 np.lib.stride_tricks.sliding_window_view 向量化提取 patch，
    替代原有双层 Python for 循环，速度提升 10~100 倍。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # 统一转为 float32：省一半内存，且与 torch float32 默认精度一致
    X = np.asarray(X, dtype=np.float32)

    # 给 X 做 padding
    if windowSize % 2 == 0:
        margin = windowSize // 2
    else:
        margin = (windowSize - 1) // 2

    zeroPaddedX = padWithZeros(X, margin=margin)

    # 向量化滑窗：一次调用提取所有 (ws, ws, C) 窗口
    # sliding_window_view 返回视图，几乎零额外内存
    H, W = X.shape[0], X.shape[1]
    patches_view = sliding_window_view(
        zeroPaddedX,
        (windowSize, windowSize, X.shape[2]),
    )  # 形状: (H_out, W_out, 1, ws, ws, C)
    patches_view = np.squeeze(patches_view, axis=2)  # (H_out, W_out, ws, ws, C)

    # 偶数 windowSize 会产生 H+1 × W+1 个窗口，截取前 H×W
    patches_view = patches_view[:H, :W, :, :, :]
    # 展平为 (H*W, ws, ws, C)；reshape 在连续内存上返回视图，否则返回副本（仍远快于循环）
    patchesData = patches_view.reshape(-1, windowSize, windowSize, X.shape[2])

    # 标签：每个像素在原图上的位置对应 (r-margin, c-margin)
    patchesLabels = y.ravel()[: patchesData.shape[0]]

    if removeZeroLabels:
        mask = patchesLabels > 0
        patchesData = patchesData[mask, :, :, :]
        patchesLabels = patchesLabels[mask]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


def split_indices_by_ratio(y, test_ratio, random_state=345):
    """与 splitTrainTestSet 相同划分，返回 train/test 索引以便多路特征同步切分。"""
    y = np.asarray(y).ravel()
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y,
    )
    return train_idx, test_idx


def split_train_balanced_rest_test_indices(y, n_per_class=400, random_state=345):
    """与 split_train_balanced_rest_test 相同划分，返回 train/test 索引。"""
    y = np.asarray(y).ravel().astype(np.int64)
    n = len(y)
    rng = np.random.RandomState(random_state)
    picks = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        k = min(n_per_class, len(idx))
        if len(idx) < n_per_class:
            print(
                f"[Train 平衡抽样] 类别 {c} 仅有 {len(idx)} 条 patch，少于每类目标 {n_per_class}，"
                f"实际抽取 {k} 条",
                flush=True,
            )
        if k > 0:
            picks.append(rng.choice(idx, size=k, replace=False))
    if not picks:
        raise ValueError("split_train_balanced_rest_test_indices: 未抽到任何训练样本，请检查标签。")
    train_idx = np.concatenate(picks)
    rng.shuffle(train_idx)
    test_idx = np.setdiff1d(np.arange(n, dtype=np.int64), train_idx, assume_unique=False)
    return train_idx, test_idx


def split_train_balanced_rest_test(X, y, n_per_class=400, random_state=345):
    """
    从全体 patch 中，对每个类别不放回各抽至多 n_per_class 条作为训练集，其余为测试集。
    二分类时各类训练条数相同（1:1），即「每类等量」的平衡训练集。
    """
    y = np.asarray(y).ravel().astype(np.int64)
    X = np.asarray(X)
    n = X.shape[0]
    rng = np.random.RandomState(random_state)
    picks = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        k = min(n_per_class, len(idx))
        if len(idx) < n_per_class:
            print(
                f"[Train 平衡抽样] 类别 {c} 仅有 {len(idx)} 条 patch，少于每类目标 {n_per_class}，"
                f"实际抽取 {k} 条",
                flush=True,
            )
        if k > 0:
            picks.append(rng.choice(idx, size=k, replace=False))
    if not picks:
        raise ValueError("split_train_balanced_rest_test: 未抽到任何训练样本，请检查标签。")
    train_idx = np.concatenate(picks)
    rng.shuffle(train_idx)
    test_idx = np.setdiff1d(np.arange(n, dtype=np.int64), train_idx, assume_unique=False)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


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
USE_AMP = True
TENSORBOARD_LOG_GRAPH = False  # jit trace 写图很慢，默认关闭
TENSORBOARD_EVERY_N_BATCHES = 20  # 降低 scalar 写入频率
GENERATE_CLS_MAP_ON_LOW_METRICS = True  # 仅低指标时生成分类图，避免拖慢每轮实验
CLS_MAP_RECALL_THRESHOLD = 0.60
CLS_MAP_AUC_THRESHOLD = 0.80


def _dataloader_num_workers() -> int:
    """Windows 多进程 DataLoader 易出问题，Linux/macOS 可开 worker 加速。"""
    return 0 if os.name == "nt" else min(4, (os.cpu_count() or 4))


# ---------- Dice + 交叉熵组合损失 ----------
DICE_LOSS_WEIGHT = 0.3  # Dice 项权重；CE 权重为 1 - DICE_LOSS_WEIGHT（默认 0.7）

# ---------- 训练/测试集划分（二选一）----------
# 'ratio'：全量 patch 按 TEST_RATIO 分层随机划分（train_test_split + stratify）
# 'balanced_per_class'：每个类别不放回各抽至多 TRAIN_SAMPLES_PER_CLASS 条作训练集，其余作测试（类等量）
TRAIN_TEST_SPLIT_MODE = 'ratio'
# 你要求：训练只取 0.01（即 train=1%，test=99%）
# 在 splitTrainTestSet 中 test_size=testRatio，因此这里固定为 0.99。
TEST_RATIO = 0.99
TRAIN_SPLIT_RANDOM_STATE = 345
TRAIN_SAMPLES_PER_CLASS = 40

# ---- 模块级内存缓存：同一进程内多次 run 复用 PCA patch，避免重复读盘与 PCA ----
_data_loader_cache: dict = {}


def create_data_loader(i):
    # 地物类别
    # class_num = 16
    # 每个像素周围提取 patch 的尺寸（窗口大小）
    patch_size = 16
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 30

    cache_key = (i, patch_size, pca_components)
    if cache_key in _data_loader_cache:
        print(f"  [缓存命中] GM0{i}，跳过重复预处理", flush=True)
        cached = _data_loader_cache[cache_key]
        X_patches = cached["X_patches"]
        y_all = cached["y_all"]
        y = cached["y"]
        priori_patches = cached.get("priori_patches", None)
    else:
        print(f"  [预处理] 加载 GM0{i}.mat ...", flush=True)
        X, y = loadData(i)
        priori = np.mean(X, axis=-1)          # (H, W) 原始波段均值，用作 1 通道先验
        print(f"  [预处理] 图像 {X.shape}, 标签 {y.shape}", flush=True)

        t_pca = time.perf_counter()
        X_pca = applyPCA(X, numComponents=pca_components)
        print(f"  [预处理] PCA →{pca_components} 维 ({time.perf_counter() - t_pca:.1f}s)", flush=True)

        t_patch = time.perf_counter()
        X_patches, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
        gaussian_weight = create_gaussian_weight_matrix(size=patch_size).astype(np.float32)
        X_patches = X_patches * gaussian_weight[:, :, np.newaxis]

        # ── 提取 priori 对应的 patch ──
        priori_3d = priori[:, :, np.newaxis].astype(np.float32)        # (H, W, 1)
        # 复用 padWithZeros 和滑动窗口
        if patch_size % 2 == 0:
            margin = patch_size // 2
        else:
            margin = (patch_size - 1) // 2
        zeroPaddedPriori = padWithZeros(priori_3d, margin=margin)
        H_img, W_img = priori.shape
        priori_view = np.squeeze(
            np.lib.stride_tricks.sliding_window_view(
                zeroPaddedPriori, (patch_size, patch_size, 1)
            ),
            axis=2,
        )  # (H, W, ws, ws, 1)
        priori_view = priori_view[:H_img, :W_img, :, :, :]
        priori_patches = priori_view.reshape(-1, patch_size, patch_size, 1)  # (N, 16, 16, 1)

        print(
            f"  [预处理] patch {X_patches.shape}, priori_patch {priori_patches.shape} "
            f"({time.perf_counter() - t_patch:.1f}s)",
            flush=True,
        )

        _data_loader_cache[cache_key] = {
            "X_patches": X_patches,
            "y_all": y_all,
            "y": y,
            "priori_patches": priori_patches,
        }

    if TRAIN_TEST_SPLIT_MODE == 'ratio':
        train_idx, test_idx = split_indices_by_ratio(
            y_all, TEST_RATIO, random_state=TRAIN_SPLIT_RANDOM_STATE
        )
    elif TRAIN_TEST_SPLIT_MODE == 'balanced_per_class':
        train_idx, test_idx = split_train_balanced_rest_test_indices(
            y_all,
            n_per_class=TRAIN_SAMPLES_PER_CLASS,
            random_state=TRAIN_SPLIT_RANDOM_STATE,
        )
    else:
        raise ValueError(
            "TRAIN_TEST_SPLIT_MODE 须为 'ratio' 或 'balanced_per_class'，"
            f"当前为 {TRAIN_TEST_SPLIT_MODE!r}"
        )

    Xtrain = X_patches[train_idx]
    Xtest = X_patches[test_idx]
    ytrain = y_all[train_idx]
    ytest = y_all[test_idx]

    print(
        f"  [数据集] 总: {len(y_all)}, 训练: {len(train_idx)} ({len(train_idx) / len(y_all) * 100:.1f}%), "
        f"测试: {len(test_idx)} ({len(test_idx) / len(y_all) * 100:.1f}%), "
        f"类别数: {len(np.unique(y_all))}",
        flush=True,
    )

    X = TestDS(X_patches, y_all)  # 全量数据（无 priori，供 get_cls_map 兼容）
    trainset = TrainDS(Xtrain, ytrain, priori_data=priori_patches[train_idx] if priori_patches is not None else None)
    testset = TestDS(Xtest, ytest, priori_data=priori_patches[test_idx] if priori_patches is not None else None)

    # WeightedRandomSampler: batch 内正负样本平衡采样，缓解 99:1 不平衡导致的
    # "全预测负类" 坍缩（不改变训练集总大小，不违反论文 1% 训练集要求）
    class_counts = np.bincount(ytrain)
    sample_weights = 1.0 / class_counts[ytrain]
    train_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.float64),
        num_samples=len(ytrain),
        replacement=True,
    )

    _use_cuda = torch.cuda.is_available()
    _num_workers = _dataloader_num_workers()
    _dl_kwargs = dict(
        batch_size=BATCH_SIZE_TRAIN,
        num_workers=_num_workers,
        pin_memory=_use_cuda,
    )
    if _num_workers > 0:
        _dl_kwargs["persistent_workers"] = True
        _dl_kwargs["prefetch_factor"] = 2

    train_loader = torch.utils.data.DataLoader(dataset=trainset, sampler=train_sampler, **_dl_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=testset, shuffle=False, **_dl_kwargs)
    all_data_loader = torch.utils.data.DataLoader(dataset=X, shuffle=False, **_dl_kwargs)

    return train_loader, test_loader, all_data_loader, y, priori


""" Training dataset — 延迟 tensor 转换，numpy 常驻，__getitem__ 按需转 torch """


class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain, priori_data=None):
        self.len = Xtrain.shape[0]
        self.x_data = np.asarray(Xtrain, dtype=np.float32)
        self.y_data = np.asarray(ytrain, dtype=np.int64)
        self.priori_data = np.asarray(priori_data, dtype=np.float32) if priori_data is not None else None

    def __getitem__(self, index):
        x = torch.from_numpy(self.x_data[index]).permute(2, 0, 1).unsqueeze(0)
        y = torch.tensor(self.y_data[index], dtype=torch.long)
        if self.priori_data is not None:
            # priori patch: (16, 16, 1) → (1, 16, 16)
            prior = torch.from_numpy(self.priori_data[index]).permute(2, 0, 1)
            return x, y, prior
        return x, y

    def __len__(self):
        return self.len


""" Testing dataset — 同上，延迟 tensor 转换 """


class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest, priori_data=None):
        self.len = Xtest.shape[0]
        self.x_data = np.asarray(Xtest, dtype=np.float32)
        self.y_data = np.asarray(ytest, dtype=np.int64)
        self.priori_data = np.asarray(priori_data, dtype=np.float32) if priori_data is not None else None

    def __getitem__(self, index):
        x = torch.from_numpy(self.x_data[index]).permute(2, 0, 1).unsqueeze(0)
        y = torch.tensor(self.y_data[index], dtype=torch.long)
        if self.priori_data is not None:
            prior = torch.from_numpy(self.priori_data[index]).permute(2, 0, 1)
            return x, y, prior
        return x, y

    def __len__(self):
        return self.len


class DiceCELoss(nn.Module):
    """
    多分类（含二分类）：softmax 概率与 one-hot 标签的多类 Dice，
    与 ``nn.CrossEntropyLoss`` 按权重相加。
    total = dice_weight * (1 - mean_class_dice) + (1 - dice_weight) * CE
    """

    def __init__(
            self,
            num_classes: int = 2,
            dice_weight: float = DICE_LOSS_WEIGHT,
            smooth: float = 1e-6,
    ):
        super().__init__()
        if not 0.0 <= dice_weight <= 1.0:
            raise ValueError("dice_weight 应在 [0, 1]")
        self.num_classes = num_classes
        self.dice_weight = float(dice_weight)
        self.ce_weight = 1.0 - self.dice_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        ce_loss = self.ce(logits, target)
        probs = F.softmax(logits, dim=1)
        oh = F.one_hot(target, self.num_classes).to(dtype=logits.dtype, device=logits.device)
        inter = (probs * oh).sum(dim=0)
        pr = probs.sum(dim=0)
        gt = oh.sum(dim=0)
        dice_per_class = (2.0 * inter + self.smooth) / (pr + gt + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


def create_baseline_classifier_net(
        sample_x,
        device,
        branch_dim=128,
        dropout=0.4,
        tri_branch_builder: Optional[Callable[..., Any]] = None,
        **moe_kwargs,
):
    """
    构建三分支分类网络，与 DataLoader 的 (B, 1, C, H, W) 一致。
    """
    if tri_branch_builder is None:
        raise ValueError("tri_branch_builder must be provided")
    return tri_branch_builder(
        sample_x,
        num_classes=2,
        branch_dim=branch_dim,
        dropout=dropout,
        **moe_kwargs,
    ).to(device)


def resolve_tri_branch_builder(model_spec: str) -> Callable[..., Any]:
    """
    加载 testbench 子模块中的 build_tri_branch_net。

    Args:
        model_spec: 不含 ``testbench.`` 前缀，如 ``baseline``、``baseline2``。
    """
    name = (model_spec or "").strip()
    if not name:
        raise ValueError("--model 不能为空")
    mod_name = f"testbench.{name.replace('/', '.')}"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"无法导入 {mod_name!r}。请在 testbench 目录下添加对应模块并实现 build_tri_branch_net。"
        ) from e
    fn = getattr(mod, "build_tri_branch_net", None)
    if not callable(fn):
        raise AttributeError(f"模块 {mod_name!r} 缺少可调用属性 build_tri_branch_net(...)")
    return fn


def train(
        train_loader,
        epochs,
        pbar=None,
        log_dir=DEFAULT_TRAIN_LOG_DIR,
        tri_branch_builder: Optional[Callable[..., Any]] = None,
):
    """tri_branch_builder 由 resolve_tri_branch_builder 提供。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    sample_batch = next(iter(train_loader))
    sample_x = sample_batch[0]  # 兼容 2 元组和 3 元组 (x, y) 或 (x, y, prior)
    net = create_baseline_classifier_net(
        sample_x,
        device,
        branch_dim=128,
        dropout=0.4,
        tri_branch_builder=tri_branch_builder,
    )

    # moe_mod = getattr(net, "moe", None)
    # if moe_mod is not None:
    #     print(
    #         f"[MoE] 已接入训练: dim={moe_mod.dim}, num_experts={moe_mod.num_experts}, "
    #         f"top_k={moe_mod.top_k}, load_balance_coef={moe_mod.load_balance_coef}, "
    #         f"residual={moe_mod.residual}"
    #     )
    # else:
    #     print("[MoE] 未启用（build 时 use_moe=False 或旧版无 moe 子模块）")

    criterion = DiceCELoss(num_classes=2, dice_weight=DICE_LOSS_WEIGHT)
    print(
        f"[Loss] DiceCELoss: dice_weight={DICE_LOSS_WEIGHT}, "
        f"ce_weight={1.0 - DICE_LOSS_WEIGHT:.1f}"
    )
    # 初始化优化器（降低学习率，更稳定训练）
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 初始化TensorBoard写入器
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard日志保存至: {log_dir}")
    print(f"启动命令: tensorboard --logdir=\"{log_dir}\"")
    base = os.path.basename(os.path.normpath(log_dir))
    if base.startswith("GM0"):
        parent = os.path.dirname(os.path.abspath(log_dir))
        print(f"（同一会话内全部 GM: tensorboard --logdir=\"{parent}\")")

    if TENSORBOARD_LOG_GRAPH:
        try:
            x_graph = sample_x.to(device=device, dtype=TORCH_DTYPE).detach()
            _tensorboard_log_torch_graph_relaxed(writer, net, x_graph)
        except Exception as e:
            print(f"模型结构记录到TensorBoard时出错: {e}")

    # 开始训练
    best_loss = float('inf')
    global_step = 0  # 用于TensorBoard的全局步数

    # 混合精度：GPU 时 float16 前向 + float32 梯度累积，CPU 自动退化为普通训练
    amp_enabled = USE_AMP and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if amp_enabled:
        print("[AMP] 混合精度训练已启用 (float16 forward + float32 grad)")
    else:
        print("[AMP] 未启用 (CPU 或 USE_AMP=False)")

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # 解包 2 或 3 元组: (data, target) 或 (data, target, prior)
            data, target = batch[0], batch[1]
            has_prior = len(batch) > 2
            prior = batch[2] if has_prior else None

            data = data.to(device=device, dtype=TORCH_DTYPE, non_blocking=use_cuda)
            target = target.to(device, non_blocking=use_cuda)
            prior = prior.to(device=device, dtype=TORCH_DTYPE, non_blocking=use_cuda) if has_prior else None

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if has_prior:
                    outputs = net(data, priori=prior)
                else:
                    outputs = net(data)
                moe_aux = getattr(net, "last_moe_aux_loss", None)
                if moe_aux is None:
                    moe_aux = outputs.new_zeros(())
                loss = criterion(outputs, target)

            # 反向传播 + 优化（GradScaler 防止 float16 梯度下溢）
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 更新损失
            epoch_loss += loss.item()
            num_batches += 1

            if global_step % TENSORBOARD_EVERY_N_BATCHES == 0:
                writer.add_scalar("Train/Loss_per_batch", loss.item(), global_step)
                writer.add_scalar("Train/moe_aux_loss", float(moe_aux.detach()), global_step)
                writer.add_scalar(
                    "Train/Learning_rate", optimizer.param_groups[0]["lr"], global_step
                )
            global_step += 1

            # 更新进度条（显示详细信息）
            if pbar is not None:
                avg_loss = epoch_loss / num_batches
                pbar.set_postfix({
                    '阶段': '训练',
                    'Epoch': f'{epoch + 1}/{epochs}',
                    'Loss': f'{loss.item():.4f}',
                    'Best': f'{best_loss:.4f}' if best_loss != float('inf') else 'N/A'
                })
                pbar.update(1)

        # 计算epoch平均损失
        epoch_avg_loss = epoch_loss / num_batches

        # 记录每个epoch的平均损失到TensorBoard
        writer.add_scalar('Train/Loss_per_epoch', epoch_avg_loss, epoch)
        writer.add_scalar('Train/Best_loss', best_loss, epoch)

        # 更新最佳损失
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss

        # 学习率调度
        scheduler.step()

    # 训练结束，关闭TensorBoard写入器
    writer.close()
    print(f"TensorBoard日志已保存至: {log_dir}")

    return net, device


def test(device, net, test_loader, pbar=None):
    count = 0
    # 模型测试
    with torch.no_grad():
        net.eval()
        y_pred_test = 0  # argmax 得到的硬标签，用于 Recall
        y_score_test = 0  # class-1 概率，用于 AUC
        y_test = 0

        for batch_idx, batch in enumerate(test_loader):
            # 解包 2 或 3 元组
            inputs, labels = batch[0], batch[1]
            has_prior = len(batch) > 2
            prior = batch[2] if has_prior else None

            inputs = inputs.to(device=device, dtype=TORCH_DTYPE, non_blocking=device.type == "cuda")
            prior = prior.to(device=device, dtype=TORCH_DTYPE, non_blocking=device.type == "cuda") if has_prior else None
            outputs = net(inputs, priori=prior) if has_prior else net(inputs)
            probs = F.softmax(outputs, dim=1)
            # class=1 的概率作为 AUC 的得分
            score_1 = probs[:, 1].detach().cpu().numpy()
            pred_cls = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels_np = labels.detach().cpu().numpy()

            if count == 0:
                y_pred_test = pred_cls
                y_score_test = score_1
                y_test = labels_np
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, pred_cls))
                y_score_test = np.concatenate((y_score_test, score_1))
                y_test = np.concatenate((y_test, labels_np))

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({'阶段': '测试', 'Batch': f'{batch_idx + 1}/{len(test_loader)}'})

    return y_pred_test, y_score_test, y_test


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test, y_score_test):
    recall = recall_score(y_test, y_pred_test, average='binary')
    # 硬标签版 AUC：把预测标签当作 score。
    auc = roc_auc_score(y_test, y_pred_test)
    return recall, auc


def run_single_experiment(
        i,
        seed=None,
        run_idx=None,
        num_runs=None,
        session_log_root=DEFAULT_TRAIN_LOG_DIR,
        tri_branch_builder: Optional[Callable[..., Any]] = None,
        model_name: Optional[str] = None,
):
    """
    运行单次实验

    Args:
        i: 数据集编号
        seed: 随机数种子（如果为None，则不设置）
        run_idx: 当前运行索引
        num_runs: 总运行次数
        session_log_root: TensorBoard 根目录，与 train 默认 log_dir 同名；其下为各 GM 子目录
        tri_branch_builder: 见 create_baseline_classifier_net
        model_name: 模型名称（用于可视化命名）

    Returns:
        recall: Recall值
        auc: AUC值
        train_time: 训练时间
        test_time: 测试时间
    """
    if seed is not None:
        set_seed(seed)

    train_loader, test_loader, all_data_loader, y_all, priori = create_data_loader(i)

    # 计算总batch数（训练 + 测试）
    epochs = 100
    total_train_batches = epochs * len(train_loader)
    total_test_batches = len(test_loader)
    total_batches = total_train_batches + total_test_batches

    # TensorBoard：统一写在 session_log_root 下，按 GM 与 run 分子目录
    os.makedirs(session_log_root, exist_ok=True)
    if run_idx is not None:
        log_dir = os.path.join(session_log_root, f"GM0{i}_run{run_idx + 1}")
    else:
        log_dir = os.path.join(session_log_root, f"GM0{i}")

    # 创建单个进度条（使用leave=True和position=0确保原地更新）
    run_info = f"[{run_idx + 1}/{num_runs}]" if run_idx is not None else ""
    pbar = tqdm(total=total_batches, desc=f"GM0{i} {run_info}", unit="batch", ncols=150,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                position=0, leave=True, mininterval=0.5)

    # 训练阶段
    tic1 = time.perf_counter()
    net, device = train(
        train_loader,
        epochs=epochs,
        pbar=pbar,
        log_dir=log_dir,
        tri_branch_builder=tri_branch_builder,
    )
    toc1 = time.perf_counter()
    train_time = toc1 - tic1

    # 测试阶段
    tic2 = time.perf_counter()
    y_pred_test, y_score_test, y_test = test(device, net, test_loader, pbar=pbar)
    toc2 = time.perf_counter()
    test_time = toc2 - tic2

    pbar.close()

    # 评价指标
    recall, auc = acc_reports(y_test, y_pred_test, y_score_test)

    if GENERATE_CLS_MAP_ON_LOW_METRICS and (
            auc < CLS_MAP_AUC_THRESHOLD or recall < CLS_MAP_RECALL_THRESHOLD
    ):
        os.makedirs("classification_maps", exist_ok=True)
        safe_model = (model_name or "unknown").replace(".", "_").replace("/", "_")
        vis_name = f"{safe_model}_GM0{i}_epoch{epochs}_recall{recall:.4f}_auc{auc:.4f}"
        print(
            f"\n  [可视化] AUC={auc:.4f}, Recall={recall:.4f} 低于阈值，"
            f"生成分类图 -> {vis_name}"
        )
        get_cls_map.get_cls_map(net, device, all_data_loader, y_all, name_prefix=vis_name)

    return recall, auc, train_time, test_time


def run(
        i,
        num_runs=3,
        base_seed=2025,
        session_log_root=DEFAULT_TRAIN_LOG_DIR,
        tri_branch_builder: Optional[Callable[..., Any]] = None,
        result_summary_txt: Optional[str] = None,
        model_name: Optional[str] = None,
):
    """
    运行多次实验并计算统计结果

    Args:
        i: 数据集编号
        num_runs: 运行次数（默认3次）
        base_seed: 基础随机数种子
        session_log_root: 本次会话 TensorBoard 根目录，所有 GM 共用
        tri_branch_builder: 见 run_single_experiment
        result_summary_txt: 本模型/本会话写入的汇总 txt；为 None 时使用 LEGACY_RESULT_SUMMARY_TXT
        model_name: 模型名称（传递给 run_single_experiment 用于可视化命名）
    """
    summary_path = (
        result_summary_txt if result_summary_txt is not None else LEGACY_RESULT_SUMMARY_TXT
    )
    print(f"\n[GM0{i}] 开始处理（运行{num_runs}次）...")

    recalls = []
    aucs = []
    train_times = []
    test_times = []

    for run_idx in range(num_runs):
        seed = base_seed + run_idx  # 每次使用不同的种子

        recall, auc, train_time, test_time = run_single_experiment(
            i,
            seed=seed,
            run_idx=run_idx,
            num_runs=num_runs,
            session_log_root=session_log_root,
            tri_branch_builder=tri_branch_builder,
            model_name=model_name,
        )

        recalls.append(recall)
        aucs.append(auc)
        train_times.append(train_time)
        test_times.append(test_time)

        # 打印当前运行结果
        print(f"  Run {run_idx + 1}: Recall={recall:.4f}, AUC={auc:.4f}")

    # 计算统计量
    recall_mean, recall_std = calculate_statistics(recalls)
    auc_mean, auc_std = calculate_statistics(aucs)
    train_time_mean, train_time_std = calculate_statistics(train_times)
    test_time_mean, test_time_std = calculate_statistics(test_times)

    # 计算SOTA值（各次训练中的最佳值）
    recall_sota = max(recalls)
    auc_sota = max(aucs)
    recall_sota_idx = recalls.index(recall_sota) + 1  # 找到最佳Recall的运行索引
    auc_sota_idx = aucs.index(auc_sota) + 1  # 找到最佳AUC的运行索引

    # 显示结果（简洁版）
    print(f"[GM0{i}] 完成 | {num_runs}次运行结果:")
    print(f"  Recall: {recall_mean:.4f} ± {recall_std:.4f} | SOTA: {recall_sota:.4f} (Run {recall_sota_idx})")
    print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f} | SOTA: {auc_sota:.4f} (Run {auc_sota_idx})")
    print(f"  训练时间: {train_time_mean / 60:.1f} ± {train_time_std / 60:.1f} min")
    print(f"  测试时间: {test_time_mean:.1f} ± {test_time_std:.1f} s")

    # 保存结果到文件
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"\nGM0{i} ({num_runs}次运行):\n")
        f.write(f"  Recall: {recall_mean:.4f} ± {recall_std:.4f} (范围: [{min(recalls):.4f}, {max(recalls):.4f}])\n")
        f.write(f"  Recall SOTA: {recall_sota:.4f} (Run {recall_sota_idx})\n")
        f.write(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f} (范围: [{min(aucs):.4f}, {max(aucs):.4f}])\n")
        f.write(f"  AUC SOTA: {auc_sota:.4f} (Run {auc_sota_idx})\n")
        f.write(f"  训练时间: {train_time_mean:.1f} ± {train_time_std:.1f} s\n")
        f.write(f"  测试时间: {test_time_mean:.1f} ± {test_time_std:.1f} s\n")
        f.write(f"  详细结果: Recall={recalls}, AUC={aucs}\n")
        f.write("=" * 50 + "\n")

    # # get_cls_map.get_cls_map(net, device, all_data_loader, y_all)


if __name__ == '__main__':
    # 在此指定本次要训练的模型（testbench 子模块名，不含 "testbench." 前缀）
    # "baseline,baseline2",baseline3","baseline4","baseline5""6""7"
    TRAIN_MODELS = [
        "bbl"

    ]

    print("=" * 70)
    print("开始处理所有数据集 (GM01-GM08)")
    print("=" * 70)

    # 实验配置
    NUM_RUNS = 5  # 每个数据集运行3次
    BASE_SEED = 2025  # 基础随机数种子
    NUM_DATASETS = 8  # 数据集数量
    START_DATASET = 1  # 断点续跑起始数据集编号（1-8）；从 GM04 开始则设为 4
    END_DATASET = NUM_DATASETS  # 结束数据集编号（1-8）

    if not (1 <= START_DATASET <= NUM_DATASETS):
        raise ValueError(
            f"START_DATASET 应在 [1, {NUM_DATASETS}]，当前为 {START_DATASET}"
        )
    if not (1 <= END_DATASET <= NUM_DATASETS):
        raise ValueError(
            f"END_DATASET 应在 [1, {NUM_DATASETS}]，当前为 {END_DATASET}"
        )
    if START_DATASET > END_DATASET:
        raise ValueError(
            f"START_DATASET({START_DATASET}) 不能大于 END_DATASET({END_DATASET})"
        )

    print(f"本次训练模型列表: {TRAIN_MODELS}")
    print(f"会话标识: {EXPERIMENT_SESSION_SLUG}")
    print(f"实验配置: 每个数据集运行{NUM_RUNS}次，基础种子={BASE_SEED}")
    print(f"数据集范围: GM0{START_DATASET} ~ GM0{END_DATASET}")
    print(f"TensorBoard 会话根目录: {DEFAULT_TRAIN_LOG_DIR}（各模型与 GM 在其子目录下）")
    print("=" * 70)

    total_start_time = time.perf_counter()
    os.makedirs("cls_result", exist_ok=True)

    for model_name in TRAIN_MODELS:
        result_txt = result_summary_path_for_model(model_name)
        print("\n" + "#" * 70)
        print(f"开始训练模型: testbench.{model_name}")
        print(f"结果汇总将写入: {result_txt}")
        print("#" * 70)
        tri_builder = resolve_tri_branch_builder(model_name)
        model_log_root = os.path.join(DEFAULT_TRAIN_LOG_DIR, model_name)

        # 断点续跑时保留既有结果并追加；全量跑时覆盖旧结果
        resume_mode = START_DATASET > 1
        result_open_mode = "a" if resume_mode else "w"
        with open(result_txt, result_open_mode, encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"实验日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"会话标识: {EXPERIMENT_SESSION_SLUG}\n")
            f.write(f"当前模型: testbench.{model_name}\n")
            f.write(f"本次 TRAIN_MODELS 全列表: {TRAIN_MODELS}\n")
            f.write(f"实验配置: 每个数据集运行{NUM_RUNS}次，基础种子={BASE_SEED}\n")
            f.write(f"数据集范围: GM0{START_DATASET} ~ GM0{END_DATASET}\n")
            if resume_mode:
                f.write("运行模式: 断点续跑（append）\n")
            f.write("=" * 70 + "\n\n")

        # 当前模型的总体进度条
        datasets_pbar = tqdm(
            range(START_DATASET, END_DATASET + 1),
            desc=f"{model_name} 总体进度",
            unit="数据集",
            ncols=150,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        )

        for i in datasets_pbar:
            datasets_pbar.set_description(f"{model_name} -> 处理数据集 GM0{i}")
            run(
                i,
                num_runs=NUM_RUNS,
                base_seed=BASE_SEED,
                session_log_root=model_log_root,
                tri_branch_builder=tri_builder,
                result_summary_txt=result_txt,
                model_name=model_name,
            )
            datasets_pbar.set_postfix({'已完成': f'GM0{i}'})

        datasets_pbar.close()

    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time

    print("=" * 70)
    print(f"全部完成！总耗时: {total_time / 60:.1f} 分钟 ({total_time:.1f} 秒)")
    print("=" * 70)
