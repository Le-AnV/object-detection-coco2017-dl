import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def decode_outputs(preds, conf_thres=0.01):
    """
    preds: list tensor [B, 4 + 1 + Nc, H, W] từ các head (logits thô)
    trả về: list length B, mỗi phần là [N, 6] = (x1, y1, x2, y2, score, class_id)
    """
    strides = [8, 16, 32]
    batch_size = preds[0].shape[0]
    device = preds[0].device

    # Mỗi ảnh trong batch có 1 tensor detections riêng
    batch_outputs = [torch.zeros((0, 6), device=device) for _ in range(batch_size)]

    for i, pred in enumerate(preds):
        stride = strides[i]

        # [B, C, H, W] -> [B, H, W, C]
        pred = pred.permute(0, 2, 3, 1)
        B, H, W, C = pred.shape

        # Objectness và class probability
        obj = torch.sigmoid(pred[..., 4])
        cls = torch.sigmoid(pred[..., 5:])

        # Lọc thô theo objectness để giảm số box cần xử lý
        mask = obj > conf_thres
        if not mask.any():
            continue

        # Decode toạ độ bbox (center + size)
        px = torch.sigmoid(pred[..., 0])
        py = torch.sigmoid(pred[..., 1])
        pw = torch.exp(pred[..., 2])
        ph = torch.exp(pred[..., 3])

        # Grid toạ độ cho feature map
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        # Đổi từ grid space sang pixel space
        cx = (px + grid_x) * stride
        cy = (py + grid_y) * stride
        w = pw * stride
        h = ph * stride

        # (cx, cy, w, h) -> (x1, y1, x2, y2)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Gom kết quả theo từng ảnh trong batch
        for b in range(batch_size):
            m = mask[b]
            if not m.any():
                continue

            # Lấy các box thỏa mask
            b_x1 = x1[b][m]
            b_y1 = y1[b][m]
            b_x2 = x2[b][m]
            b_y2 = y2[b][m]
            b_obj = obj[b][m]
            b_cls = cls[b][m]

            # Score cuối = objectness * class prob (lấy class tốt nhất)
            scores, class_ids = b_cls.max(dim=1)
            final_scores = b_obj * scores

            # Lọc lại theo score cuối
            keep = final_scores > conf_thres

            detections = torch.stack(
                [
                    b_x1[keep],
                    b_y1[keep],
                    b_x2[keep],
                    b_y2[keep],
                    final_scores[keep],
                    class_ids[keep].float(),
                ],
                dim=1,
            )

            batch_outputs[b] = torch.cat([batch_outputs[b], detections], dim=0)

    return batch_outputs
