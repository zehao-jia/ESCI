import torch
import torch.nn as nn
import torch.nn.functional as F

def connected_components_bbox(gt):
    """
    从GT得到每个连通前景的最小外接矩形 mask
    输入: gt [B,1,H,W]  0/1
    输出: fore_list [B个list], 每个元素是 [H,W] 的bbox mask
    """
    B, _, H, W = gt.shape
    fore_list = []
    for b in range(B):
        mask = gt[b, 0].cpu().numpy()
        from skimage.measure import label, regionprops
        lbl, n = label(mask, connectivity=2, return_num=True)
        props = regionprops(lbl)
        boxes = []
        for p in props:
            y0, x0, y1, x1 = p.bbox
            m = torch.zeros((H, W), device=gt.device)
            m[y0:y1, x0:x1] = 1.0
            boxes.append(m)
        fore_list.append(boxes)
    return fore_list

class SIAUC_Loss(nn.Module):
    """
    论文 SI-AUC 损失 + PBAcc 加速
    对应论文 Eq.28, 29, 30, 31, 32
    """
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, pred, gt, fore_list):
        """
        pred: [B,1,H,W]  logits
        gt:   [B,1,H,W]  0/1
        fore_list: 每个前景的bbox mask，来自 connected_components_bbox
        """
        pred = torch.sigmoid(pred)
        B = pred.shape[0]
        total_loss = 0.0

        for b in range(B):
            fore_msks = fore_list[b]
            M = len(fore_msks)
            if M == 0:
                continue

            # 全图正负像素
            gt_flat = gt[b].view(-1)
            pred_flat = pred[b].view(-1)
            S = gt_flat.numel()
            pos_mask = (gt_flat == 1.0).float()
            neg_mask = (gt_flat == 0.0).float()
            S_neg = neg_mask.sum() + self.eps

            loss_k = 0.0
            for msk in fore_msks:
                # 当前前景的正样本权重：1/S_k^+
                fore_msk_flat = msk.view(-1)
                fore_pos = pos_mask * fore_msk_flat
                S_k_pos = fore_pos.sum() + self.eps

                # 论文 PBAcc 邻接矩阵 A = (y (1-y)^T + (1-y) y^T) / S_neg
                y = gt_flat
                y1my = torch.outer(y, 1 - y)
                my1y = torch.outer(1 - y, y)
                A = (y1my + my1y) / S_neg

                # 权重 c: 属于第k前景的正像素设为 1/S_k_pos
                c = torch.zeros_like(y)
                c[fore_pos.bool()] = 1.0 / S_k_pos
                y_tilde = y * c

                # 度矩阵 D
                c_plus = y_tilde.sum()
                D = torch.diag(y_tilde + c_plus * (1 - y))

                # 拉普拉斯矩阵 P = D - A
                P = D - A

                # 最终 PBAcc 损失
                f = pred_flat
                loss = (f - y).unsqueeze(0) @ P @ (f - y).unsqueeze(1)
                loss = loss.squeeze()
                loss_k += loss

            loss_k = loss_k / M
            total_loss += loss_k

        return total_loss / B

# =========================
# 完整 SIAUC 使用示例
# =========================
if __name__ == "__main__":
    pred = torch.randn(2, 1, 384, 384).cuda()
    gt = torch.randint(0, 2, (2, 1, 384, 384)).float().cuda()

    fore_list = connected_components_bbox(gt)
    siauc = SIAUC_Loss()
    loss = siauc(pred, gt, fore_list)
    print("SIAUC Loss =", loss.item())