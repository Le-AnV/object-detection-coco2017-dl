import torch
from torchvision.ops import nms


def decode_predictions(preds, strides=[8, 16, 32], conf_thres=0.001):
    """
    Decode output của YOLO anchor-free
    Trả về: list[B] mỗi phần tử Tensor [N,6] = xyxy, score, cls
    """
    device = preds[0].device
    B = preds[0].shape[0]
    outputs = [[] for _ in range(B)]

    for pred, stride in zip(preds, strides):
        B, _, H, W = pred.shape
        pred = pred.permute(0, 2, 3, 1)  # [B,H,W,5+C]

        # tạo grid
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        grid = torch.stack((x, y), -1).float()

        for b in range(B):
            p = pred[b]

            # decode bbox
            xy = (p[..., :2].sigmoid() * 2 - 0.5 + grid) * stride
            wh = (p[..., 2:4].sigmoid() * 2).pow(2) * stride

            box1 = xy - wh / 2
            box2 = xy + wh / 2

            obj = p[..., 4].sigmoid()
            cls_prob = p[..., 5:].sigmoid()
            cls_score, cls_id = cls_prob.max(-1)

            score = obj * cls_score
            mask = score > conf_thres
            if mask.sum() == 0:
                continue

            boxes = torch.cat([box1[mask], box2[mask]], dim=1)
            scores = score[mask]
            labels = cls_id[mask].float()

            outputs[b].append(
                torch.cat([boxes, scores[:, None], labels[:, None]], dim=1)
            )

    return [
        torch.cat(o, dim=0) if len(o) else torch.empty((0, 6), device=device)
        for o in outputs
    ]


def apply_nms(preds, iou_thres=0.6):
    """
    NMS cho từng ảnh
    preds: list[T(N,6)]
    """
    results = []

    for p in preds:
        if p.numel() == 0:
            results.append(p)
            continue

        keep = nms(p[:, :4], p[:, 4], iou_thres)
        results.append(p[keep])

    return results
