import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# 工具函数：连通域分析 → 得到前景框、背景区域
# ======================
def get_foreground_backbone_mask(gt_mask):
    """
    输入：gt_mask [B, 1, H, W] 二值显著图（1=前景，0=背景）
    输出：fore_masks list[B], back_mask [B,1,H,W]
    每个前景是最小外接矩形的mask
    """
    B, _, H, W = gt_mask.shape
    fore_masks_batch = []
    back_mask_batch = torch.ones_like(gt_mask)

    for b in range(B):
        mask = gt_mask[b, 0].cpu().numpy()
        # 连通域分析
        from skimage.measure import label, regionprops
        labeled, num = label(mask, connectivity=2, return_num=True)
        props = regionprops(labeled)

        fore_masks = []
        for p in props:
            ymin, xmin, ymax, xmax = p.bbox
            box_mask = torch.zeros((H, W), device=gt_mask.device)
            box_mask[ymin:ymax, xmin:xmax] = 1.0
            fore_masks.append(box_mask)
            back_mask_batch[b, 0] *= (1 - box_mask)

        fore_masks_batch.append(fore_masks)
    return fore_masks_batch, back_mask_batch

# ======================
# 1. 尺寸不变 BCE 损失 SIBCE
# ======================
class SIBCE(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, fore_masks_batch, back_mask, alpha_SI):
        B = pred.shape[0]
        total_loss = 0.0

        for b in range(B):
            fore_msk = fore_masks_batch[b]
            M = len(fore_msk)
            if M == 0:
                continue

            # 前景损失平均
            fore_loss = 0.0
            for msk in fore_msk:
                msk = msk.unsqueeze(0).unsqueeze(0)
                p = pred[b:b+1] * msk
                g = gt[b:b+1] * msk
                loss = F.binary_cross_entropy_with_logits(p, g, reduction='mean')
                fore_loss += loss
            fore_loss /= M

            # 背景损失
            b_msk = back_mask[b:b+1]
            p_b = pred[b:b+1] * b_msk
            g_b = gt[b:b+1] * b_msk
            back_loss = F.binary_cross_entropy_with_logits(p_b, g_b, reduction='mean')

            # 总损失
            loss_img = (fore_loss + alpha_SI * back_loss) / (M + alpha_SI)
            total_loss += loss_img
        return total_loss / B

# ======================
# 2. 尺寸不变 Dice 损失 SIDice
# ======================
class SIDice(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, gt, fore_masks_batch):
        B = pred.shape[0]
        total_loss = 0.0
        pred = torch.sigmoid(pred)

        for b in range(B):
            fore_msk = fore_masks_batch[b]
            M = len(fore_msk)
            if M == 0:
                continue

            loss = 0.0
            for msk in fore_msk:
                msk = msk.unsqueeze(0).unsqueeze(0)
                p = pred[b:b+1] * msk
                g = gt[b:b+1] * msk
                inter = (p * g).sum()
                union = p.sum() + g.sum()
                loss += 1 - (2 * inter + self.smooth) / (union + self.smooth)
            loss /= M
            total_loss += loss
        return total_loss / B

# ======================
# 3. 尺寸不变 AUC 损失 SIAUC + PBAcc 加速
# ======================
class SIAUC(nn.Module):
    def forward(self, pred, gt, fore_masks_batch):
        B, _, H, W = pred.shape
        total_loss = 0.0
        pred = torch.sigmoid(pred).view(B, -1)
        gt = gt.view(B, -1)

        for b in range(B):
            fore_msk = fore_masks_batch[b]
            M = len(fore_msk)
            if M == 0:
                continue

            loss = 0.0
            for msk in fore_msk:
                msk = msk.view(-1)
                pos_idx = torch.where((gt[b] == 1) & (msk == 1))[0]
                neg_idx = torch.where(gt[b] == 0)[0]
                if len(pos_idx) == 0 or len(neg_idx) == 0:
                    continue

                pos = pred[b, pos_idx]
                neg = pred[b, neg_idx]
                # PBAcc 简化：矩阵快速计算
                diff = 1.0 - (pos.unsqueeze(1) - neg.unsqueeze(0))
                diff = torch.clamp(diff, min=0) ** 2
                loss += diff.mean()
            loss /= M
            total_loss += loss
        return total_loss / B

# ======================
# SIOpt2 总损失（最终版）
# ======================
class SIOpt2Loss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0, weight_auc=0.1):
        super().__init__()
        self.sibce = SIBCE()
        self.sidice = SIDice()
        self.siauc = SIAUC()
        self.w_bce = weight_bce
        self.w_dice = weight_dice
        self.w_auc = weight_auc

    def forward(self, pred, gt):
        # 1. 得到前景、背景mask
        fore_masks_batch, back_mask = get_foreground_backbone_mask(gt)

        # 2. 自适应平衡系数 alpha_SI
        B = pred.shape[0]
        fore_pixel_num = sum([len(m.view(-1).nonzero()) for m in fore_masks_batch[0]])
        back_pixel_num = back_mask[0].sum().item()
        alpha_SI = back_pixel_num / (fore_pixel_num + 1e-8)

        # 3. 三个损失
        loss_bce = self.sibce(pred, gt, fore_masks_batch, back_mask, alpha_SI)
        loss_dice = self.sidice(pred, gt, fore_masks_batch)
        loss_auc = self.siauc(pred, gt, fore_masks_batch)

        # 加权和
        total_loss = (self.w_bce * loss_bce
                     + self.w_dice * loss_dice
                     + self.w_auc * loss_auc)
        return total_loss

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    criterion = SIOpt2Loss()
    pred = torch.randn(2, 1, 384, 384)    # 模型输出
    gt = torch.randint(0, 2, (2, 1, 384, 384)).float()  # 真值
    loss = criterion(pred, gt)
    print("SIOpt2 Loss =", loss.item())