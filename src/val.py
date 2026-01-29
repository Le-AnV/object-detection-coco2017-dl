import torch
import torchmetrics
from tqdm import tqdm

from utils.post_process import post_process


def val_one_epoch(model, loader, criterion, device, epoch):
    model.eval()

    # 1. Setup Metric Calculator (torchmetrics)
    map_metric = torchmetrics.detection.mean_ap.MeanAveragePrecision(  # type: ignore
        class_metrics=False
    )

    # 2. Setup Loss Accumulator
    running_metrics = {"total_loss": 0, "box_loss": 0, "obj_loss": 0, "cls_loss": 0}
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")

    with torch.no_grad():
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets_tensor = targets.to(device)

            # --- A. Tính Loss (để vẽ biểu đồ so sánh train/val) ---
            preds = model(imgs)
            loss, loss_items = criterion(preds, targets_tensor)

            for k, v in loss_items.items():
                running_metrics[k] += v

            # --- B. Tính mAP ---
            # 1. Post-process predictions (Raw Tensor -> Box Coordinates)
            # Output: List[Dict{'boxes', 'scores', 'labels'}] (Pixel coords)
            preds_processed = post_process(preds, conf_thres=0.001, iou_thres=0.6)

            # 2. Prepare Targets for Metric (Convert Normalized Tensor -> Pixel Dict)
            # Targets tensor: [idx, cls, x, y, w, h] -> List[Dict]
            target_formatted = []
            batch_size = imgs.shape[0]
            _, _, H, W = imgs.shape

            for b in range(batch_size):
                t = targets_tensor[targets_tensor[:, 0] == b]

                if len(t) > 0:
                    # Normalized xywh -> Pixel xyxy
                    boxes_norm = t[:, 2:6]
                    cls_labels = t[:, 1].long()

                    # cx,cy,w,h -> x1,y1,x2,y2 pixel
                    boxes_pixel = boxes_norm.clone()
                    boxes_pixel[:, 0] = (boxes_norm[:, 0] - boxes_norm[:, 2] / 2) * W
                    boxes_pixel[:, 1] = (boxes_norm[:, 1] - boxes_norm[:, 3] / 2) * H
                    boxes_pixel[:, 2] = (boxes_norm[:, 0] + boxes_norm[:, 2] / 2) * W
                    boxes_pixel[:, 3] = (boxes_norm[:, 1] + boxes_norm[:, 3] / 2) * H

                    target_formatted.append(dict(boxes=boxes_pixel, labels=cls_labels))
                else:
                    target_formatted.append(
                        dict(
                            boxes=torch.tensor([]).to(device),
                            labels=torch.tensor([]).to(device),
                        )
                    )

            # 3. Update Metric state
            map_metric.update(preds_processed, target_formatted)

    # Tổng hợp kết quả
    avg_loss_metrics = {k: v / num_batches for k, v in running_metrics.items()}

    # Tính mAP
    map_results = map_metric.compute()

    # Merge Loss dict và Map dict lại thành 1
    final_metrics = {**avg_loss_metrics}
    final_metrics["map_50"] = map_results["map_50"].item()
    final_metrics["map"] = map_results["map"].item()  # mAP 0.5:0.95

    print(
        f"Val Epoch {epoch} | Loss: {final_metrics['total_loss']:.4f} | mAP50: {final_metrics['map_50']:.4f}"
    )

    return final_metrics
