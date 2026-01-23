import torch
import torch.nn as nn
import math


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.strides = [8, 16, 32]

        # Gains (Trọng số cho từng loại loss)
        self.box_gain = 0.05
        self.obj_gain = 1.0
        self.cls_gain = 0.5

        # Balance Objectness cho các tầng P3, P4, P5
        self.balance = [4.0, 1.0, 0.4]

        # Hàm Loss cơ bản
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, targets):
        """
        preds: List[Tensor] -> [[B, 85, 80, 80], [B, 85, 40, 40], [B, 85, 20, 20]]
        targets: Tensor [N, 6] -> [batch_idx, class, cx, cy, w, h] (Normalized 0-1)
        """
        device = preds[0].device
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)

        # 1. Xây dựng Target cho từng Feature Map (Grid)
        # targets_maps: List 3 tensors [B, H, W, 6]
        # Format channel cuối: [tx_offset, ty_offset, tw_grid, th_grid, 1.0, class_id]
        targets_maps = self._build_targets(preds, targets)

        # 2. Duyệt qua từng tầng (P3, P4, P5)
        for i, pred in enumerate(preds):
            # pred: [B, 4+1+Nc, H, W] -> chuyển về [B, H, W, 4+1+Nc]
            pred = pred.permute(0, 2, 3, 1)
            target = targets_maps[i]  # [B, H, W, 6]

            # Mask xác định ô nào có vật thể
            obj_mask = target[..., 4] == 1

            # --- A. Objectness Loss (Tính trên TOÀN BỘ Grid) ---
            # Target objectness: 1 ở ô có vật, 0 ở ô nền
            t_obj = torch.zeros_like(pred[..., 4])
            t_obj[obj_mask] = 1.0

            loss_obj += self.bce(pred[..., 4], t_obj).mean() * self.balance[i]

            # Nếu không có object nào trong batch ở tầng này thì bỏ qua
            if obj_mask.sum() == 0:
                continue

            # --- B. Lọc ra các mẫu dương (Positive Samples) ---
            # Chỉ tính Box và Class loss tại những ô có vật thể
            pred_pos = pred[obj_mask]  # [N_pos, 85]
            target_pos = target[obj_mask]  # [N_pos, 6]

            # --- C. Classification Loss ---
            # pred_pos[:, 5:] là logits của các class
            # Tạo one-hot target cho class
            t_cls = torch.zeros_like(pred_pos[:, 5:])
            class_ids = target_pos[:, 5].long()
            t_cls[torch.arange(t_cls.size(0)), class_ids] = (
                1.0  # Label smoothing nếu cần: 1.0 -> 0.9something
            )

            loss_cls += self.bce(pred_pos[:, 5:], t_cls).mean()

            # --- D. Box Regression Loss (CIoU) ---
            # 1. Decode Prediction (Grid -> Pixel)
            # Sigmoid cho offset x, y (0-1 trong ô lưới)
            px = torch.sigmoid(pred_pos[:, 0])
            py = torch.sigmoid(pred_pos[:, 1])
            # Exp cho width, height (tương đối so với stride)
            pw = torch.exp(pred_pos[:, 2])
            ph = torch.exp(pred_pos[:, 3])

            # Lấy toạ độ grid (gx, gy)
            # obj_mask là [B, H, W], nonzero sẽ trả về indices
            b_idx, gy, gx = torch.where(obj_mask)
            stride = self.strides[i]

            # Chuyển về Pixel Space
            pred_cx = (px + gx) * stride
            pred_cy = (py + gy) * stride
            pred_w = pw * stride
            pred_h = ph * stride
            pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)

            # 2. Decode Target (Grid -> Pixel)
            # target_pos đã chứa [tx_offset, ty_offset, tw_grid, th_grid, ...]
            # tx_offset = cx_grid - gx
            # => cx_grid = tx_offset + gx
            # => cx_pixel = (tx_offset + gx) * stride
            t_cx = (target_pos[:, 0] + gx) * stride
            t_cy = (target_pos[:, 1] + gy) * stride
            t_w = target_pos[:, 2] * stride
            t_h = target_pos[:, 3] * stride
            target_boxes = torch.stack([t_cx, t_cy, t_w, t_h], dim=1)

            # 3. Tính CIoU
            iou = self.bbox_ciou(pred_boxes, target_boxes)
            loss_box += (1.0 - iou).mean()

        # Tổng hợp Loss
        total_loss = (
            loss_box * self.box_gain
            + loss_obj * self.obj_gain
            + loss_cls * self.cls_gain
        )

        return total_loss, {
            "box_loss": loss_box.item(),
            "obj_loss": loss_obj.item(),
            "cls_loss": loss_cls.item(),
        }

    def _build_targets(self, preds, targets):
        """
        Assign targets to grid cells.
        Input: targets [batch_idx, class, cx, cy, w, h] (Normalized 0-1)
        Output: List 3 tensors [B, H, W, 6]
                Channel 6: [tx, ty, tw, th, obj, class] (Grid units)
        """
        target_list = []
        batch_size = preds[0].shape[0]
        device = preds[0].device

        for i, stride in enumerate(self.strides):
            # Lấy kích thước feature map hiện tại (H, W)
            h_fs, w_fs = preds[i].shape[2], preds[i].shape[3]

            # Tạo target tensor rỗng
            t_tensor = torch.zeros((batch_size, h_fs, w_fs, 6), device=device)

            if targets.numel() > 0:
                # Convert Normalized Coordinates -> Grid Coordinates
                # cx (0-1) * w_fs = cx (grid units, e.g., 15.5)
                gx_all = targets[:, 2] * w_fs
                gy_all = targets[:, 3] * h_fs
                gw_all = targets[:, 4] * w_fs
                gh_all = targets[:, 5] * h_fs

                # Lấy phần nguyên để biết vật thuộc ô nào
                gi = gx_all.long()
                gj = gy_all.long()

                # Kẹp giá trị trong phạm vi feature map (an toàn)
                gi = gi.clamp(0, w_fs - 1)
                gj = gj.clamp(0, h_fs - 1)

                batch_idx = targets[:, 0].long()
                class_ids = targets[:, 1]

                # --- Assign Values ---
                # Offset tương đối trong ô lưới (0 -> 1)
                tx = gx_all - gi.float()
                ty = gy_all - gj.float()

                # Gán vào tensor
                # Lưu ý: Cách gán này ghi đè nếu 1 ô có nhiều vật thể (First-come-last-serve)
                # Đây là cách đơn giản nhất. Các phiên bản nâng cao dùng SimOTA.
                t_tensor[batch_idx, gj, gi, 0] = tx
                t_tensor[batch_idx, gj, gi, 1] = ty
                t_tensor[batch_idx, gj, gi, 2] = gw_all
                t_tensor[batch_idx, gj, gi, 3] = gh_all
                t_tensor[batch_idx, gj, gi, 4] = 1.0  # Objectness
                t_tensor[batch_idx, gj, gi, 5] = class_ids  # Class ID

            target_list.append(t_tensor)

        return target_list

    @staticmethod
    def bbox_ciou(box1, box2, eps=1e-7):
        """
        Tính CIoU Loss.
        Input: box1, box2 dạng (cx, cy, w, h) - Pixel Space
        Output: CIoU score (Scalar)
        """
        # 1. Chuyển cx,cy,w,h -> x1,y1,x2,y2
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

        # 2. Intersection Area
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        # 3. Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union_area = w1 * h1 + w2 * h2 - inter_area + eps

        # 4. IoU
        iou = inter_area / union_area

        # 5. Distance center (rho^2) / Diagonal (c^2)
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw**2 + ch**2 + eps

        rho2 = (box2[..., 0] - box1[..., 0]) ** 2 + (box2[..., 1] - box1[..., 1]) ** 2

        # 6. Aspect Ratio Penalty (v)
        v = (4 / (math.pi**2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))

        ciou = iou - (rho2 / c2) - (alpha * v)
        return ciou
