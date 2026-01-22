from tqdm import tqdm
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from utils.nms import non_max_suppression
from utils.box_ops import decode_outputs


@torch.no_grad()
def validate_one_epoch(model, loss_fn, data_loader, device, epoch):
    model.eval()

    # Metric mAP (0.5:0.95)
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=False
    )
    metric.to(device)

    # Biến theo dõi loss
    running_loss = 0.0
    running_box = 0.0
    running_obj = 0.0
    running_cls = 0.0
    num_batches = len(data_loader)

    pbar = tqdm(data_loader, desc=f"Val epoch {epoch}")

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        B, C, H, W = images.shape

        # Forward
        preds = model(images)

        # Tính loss
        loss, loss_items = loss_fn(preds, targets)

        running_loss += loss.item()
        running_box += loss_items["box_loss"]
        running_obj += loss_items["obj_loss"]
        running_cls += loss_items["cls_loss"]

        # Decode + NMS
        raw_proposals = decode_outputs(preds, conf_thres=0.01)
        final_preds = non_max_suppression(raw_proposals, iou_thres=0.5)

        pred_list = []
        target_list = []

        for i in range(B):
            # Predictions của ảnh i
            p = final_preds[i]
            pred_list.append(
                {
                    "boxes": p[:, :4],
                    "scores": p[:, 4],
                    "labels": p[:, 5].long(),
                }
            )

            # Targets của ảnh i
            mask = targets[:, 0] == i
            t_img = targets[mask]

            if len(t_img) > 0:
                # cx,cy,w,h (0–1) -> x1,y1,x2,y2 (pixel)
                cx = t_img[:, 2] * W
                cy = t_img[:, 3] * H
                w = t_img[:, 4] * W
                h = t_img[:, 5] * H

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                t_boxes = torch.stack([x1, y1, x2, y2], dim=1)
                t_labels = t_img[:, 1].long()
            else:
                t_boxes = torch.empty((0, 4), device=device)
                t_labels = torch.empty((0,), device=device, dtype=torch.long)

            target_list.append({"boxes": t_boxes, "labels": t_labels})

        # Update mAP cho batch
        metric.update(pred_list, target_list)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Tổng hợp metric cả epoch
    metrics_result = metric.compute()
    metric.reset()

    return {
        "total_loss": running_loss / num_batches,
        "box_loss": running_box / num_batches,
        "obj_loss": running_obj / num_batches,
        "cls_loss": running_cls / num_batches,
        "map_50": metrics_result["map_50"].item(),
        "map": metrics_result["map"].item(),
    }
