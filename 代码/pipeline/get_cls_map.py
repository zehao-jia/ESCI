import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


def get_classification_map(y_pred, y):
    """
    兼容旧接口：仅在 gt!=0 的像素上填入预测（+1 作为显示类别）。
    全图二分类请优先使用 y_pred_to_raster + build_binary_analysis_map。
    """
    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k] + 1
                k += 1

    return cls_labels


def y_pred_to_raster(y_pred, height, width):
    """将按行优先排列的 patch 预测展成 (H, W)，与 createImageCubes 顺序一致。"""
    y_pred = np.asarray(y_pred).ravel()
    if y_pred.size != height * width:
        raise ValueError(
            f"预测长度 {y_pred.size} 与 H×W={height * width} 不一致，无法对齐原图。"
        )
    return y_pred.reshape(height, width)


def build_binary_analysis_map(gt, pred):
    """
    二分类 gt/pred ∈ {0,1}：1=TN, 2=TP, 3=FP, 4=FN，0 保留给不参与分析的像素。
    """
    gt = np.asarray(gt).astype(np.int64)
    pred = np.asarray(pred).astype(np.int64)
    out = np.zeros(gt.shape, dtype=np.int8)
    tn = (gt == 0) & (pred == 0)
    tp = (gt == 1) & (pred == 1)
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)
    out[tn] = 1
    out[tp] = 2
    out[fp] = 3
    out[fn] = 4
    return out


def analysis_category_to_rgb(cat_map):
    """类别 0 黑；1 TN 蓝；2 TP 绿；3 FP 橙；4 FN 红。"""
    h, w = cat_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    palette = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([0.25, 0.45, 0.95]),  # TN 正确负类
        2: np.array([0.25, 0.85, 0.35]),  # TP 正确正类
        3: np.array([1.0, 0.55, 0.1]),   # FP
        4: np.array([0.92, 0.25, 0.28]), # FN
    }
    for k, col in palette.items():
        rgb[cat_map == k] = col
    return rgb


def emphasis_single_category_rgb(cat_map, show_codes, dim_others=0.12):
    """只高亮 show_codes 中的类别，其余置为暗灰。"""
    rgb = analysis_category_to_rgb(cat_map)
    mask = np.zeros(cat_map.shape, dtype=bool)
    for c in show_codes:
        mask |= cat_map == c
    dim = np.array([dim_others, dim_others, dim_others], dtype=np.float32)
    rgb[~mask] = dim
    return rgb


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.0
        if item == 1:
            y[index] = np.array([147, 67, 46]) / 255.0

    return y


def classification_map(map_rgb, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map_rgb)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

    return 0


def save_analysis_map_with_legend(cat_map, dpi, save_path, title="二分类结果分析"):
    """保存四色分析图并在下方附加图例。"""
    rgb = analysis_category_to_rgb(cat_map)
    h, w = cat_map.shape
    fig_h = max(5.0, h / max(w, 1) * 6.0) + 1.2
    fig_w = max(6.0, w / max(h, 1) * 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgb)
    ax.set_axis_off()

    patches = [
        mpatches.Patch(color=(0.25, 0.45, 0.95), label="TN 正确负类"),
        mpatches.Patch(color=(0.25, 0.85, 0.35), label="TP 正确正类"),
        mpatches.Patch(color=(1.0, 0.55, 0.1), label="FP 假正"),
        mpatches.Patch(color=(0.92, 0.25, 0.28), label="FN 假负"),
    ]
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=True,
        fontsize=9,
    )
    ax.set_title(title, fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return 0


def test(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            lab = labels.detach().cpu().numpy()

            if count == 0:
                y_pred_test = pred
                y_test = lab
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, pred))
                y_test = np.concatenate((y_test, lab))

    return y_pred_test, y_test


def get_cls_map(net, device, all_data_loader, y, name_prefix="IP"):
    y_pred, _y_new = test(device, net, all_data_loader)
    y = np.asarray(y)
    height, width = y.shape[0], y.shape[1]

    # ---------- 全图栅格预测 + 二分类 TN/TP/FP/FN 分析 ----------
    pred_raster = y_pred_to_raster(y_pred, height, width)
    uniq = np.unique(y)
    if uniq.size > 2 or np.setdiff1d(uniq, [0, 1]).size > 0:
        print(
            "[get_cls_map] 警告: 真值标签并非仅含 0/1，分析图按二分类 TN/TP/FP/FN 可能不适用。"
        )
    cat_map = build_binary_analysis_map(y, pred_raster)

    # ---------- 渲染各子图 RGB 数据 ----------
    # 真值
    gt_flat = y.flatten()
    y_gt = list_to_colormap(gt_flat)
    gt_rgb = np.reshape(y_gt, (height, width, 3))

    # 预测
    cls_labels = pred_raster.astype(np.float64)
    x = np.ravel(cls_labels)
    y_pred_rgb_list = list_to_colormap(x)
    pred_rgb = np.reshape(y_pred_rgb_list, (height, width, 3))

    # 二分类四色分析 + 分项高亮
    binary_rgb = analysis_category_to_rgb(cat_map)
    tn_rgb = emphasis_single_category_rgb(cat_map, [1])
    tp_rgb = emphasis_single_category_rgb(cat_map, [2])
    fnfp_rgb = emphasis_single_category_rgb(cat_map, [3, 4])

    # ---------- 合成一张大图（3×2 子图矩阵）----------
    os.makedirs("classification_maps", exist_ok=True)
    save_path = f"classification_maps/{name_prefix}_composite.png"

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle(name_prefix, fontsize=13, fontweight="bold", y=0.98)

    # (0,0) GT
    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title("Ground Truth", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    # (0,1) Predictions
    axes[0, 1].imshow(pred_rgb)
    axes[0, 1].set_title("Predictions", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    # (1,0) TN
    axes[1, 0].imshow(tn_rgb)
    axes[1, 0].set_title("TN (Correct Negative)", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")

    # (1,1) TP
    axes[1, 1].imshow(tp_rgb)
    axes[1, 1].set_title("TP (Correct Positive)", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")

    # (2,0) FP+FN
    axes[2, 0].imshow(fnfp_rgb)
    axes[2, 0].set_title("Misclassified (FP + FN)", fontsize=10, fontweight="bold")
    axes[2, 0].axis("off")

    # (2,1) Binary Analysis 四色 + 图例
    axes[2, 1].imshow(binary_rgb)
    axes[2, 1].set_title("Binary Analysis (TN/TP/FP/FN)", fontsize=10, fontweight="bold")
    axes[2, 1].axis("off")
    legend_patches = [
        mpatches.Patch(color=(0.25, 0.45, 0.95), label="TN"),
        mpatches.Patch(color=(0.25, 0.85, 0.35), label="TP"),
        mpatches.Patch(color=(1.0, 0.55, 0.1), label="FP"),
        mpatches.Patch(color=(0.92, 0.25, 0.28), label="FN"),
    ]
    axes[2, 1].legend(
        handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.85
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("------Get classification maps successful-------")
    print(f"  合成图: {name_prefix}_composite.png （含 GT / Predictions / TN / TP / FP+FN / 四色分析）")
