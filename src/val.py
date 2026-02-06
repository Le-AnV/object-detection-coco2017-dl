from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

from utils.postprocess import decode_predictions, apply_nms


@torch.no_grad()
def validate_one_epoch(
    model,
    loss_fn,
    data_loader,
    device,
    epoch,
):
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=False,
    ).to(device)

    loss_sum = box_sum = obj_sum = cls_sum = 0.0
    num_batches = len(data_loader)

    pbar = tqdm(data_loader, desc=f"Val epoch {epoch}")

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        B, _, H, W = images.shape

        # --- Forward + Loss ---
        preds = model(images)
        loss, loss_items = loss_fn(preds, targets)

        loss_sum += loss.item()
        box_sum += loss_items["box"].item()
        obj_sum += loss_items["obj"].item()
        cls_sum += loss_items["cls"].item()

        # --- Decode + NMS ---
        decoded = decode_predictions(
            preds,
            conf_thres=0.25,
        )
        final_preds = apply_nms(decoded, iou_thres=0.45)

        # --- Chuẩn bị dữ liệu cho metric ---
        pred_list = []
        target_list = []

        for i in range(B):
            # Predictions
            p = final_preds[i]
            pred_list.append(
                {
                    "boxes": p[:, :4],
                    "scores": p[:, 4],
                    "labels": p[:, 5].long(),
                }
            )

            # Targets
            mask = targets[:, 0] == i
            t = targets[mask]

            if len(t) > 0:
                cxcywh = t[:, 2:6] * torch.tensor([W, H, W, H], device=device)
                xyxy = torch.zeros_like(cxcywh)
                xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
                xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
                xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
                xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2

                target_list.append(
                    {
                        "boxes": xyxy,
                        "labels": t[:, 1].long(),
                    }
                )
            else:
                target_list.append(
                    {
                        "boxes": torch.empty((0, 4), device=device),
                        "labels": torch.empty((0,), device=device, dtype=torch.long),
                    }
                )

        metric.update(pred_list, target_list)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    results = metric.compute()
    metric.reset()

    return {
        "val_loss": loss_sum / num_batches,
        "val_box": box_sum / num_batches,
        "val_obj": obj_sum / num_batches,
        "val_cls": cls_sum / num_batches,
        "map_50": results["map_50"].item(),
        "map": results["map"].item(),
    }
