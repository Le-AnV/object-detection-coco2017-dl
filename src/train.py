from tqdm import tqdm
import torch


def train_one_epoch(
    model,
    optimizer,
    loss_fn,
    data_loader,
    device,
    epoch,
    scaler,
):
    model.train()

    loss_sum = box_sum = obj_sum = cls_sum = 0.0
    num_batches = len(data_loader)

    pbar = tqdm(data_loader, desc=f"Train epoch {epoch}")

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # -------- AMP FORWARD --------
        with torch.autocast(device_type="cuda"):  # type: ignore
            preds = model(images)
            loss, loss_items = loss_fn(preds, targets)

        # -------- AMP BACKWARD --------
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # -------- LOG --------
        loss_val = loss.item()
        box_val = loss_items["box"].item()
        obj_val = loss_items["obj"].item()
        cls_val = loss_items["cls"].item()

        loss_sum += loss_val
        box_sum += box_val
        obj_sum += obj_val
        cls_sum += cls_val

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            box=f"{box_val:.4f}",
            obj=f"{obj_val:.4f}",
            cls=f"{cls_val:.4f}",
            lr=f"{lr:.6f}",
        )

    return {
        "total_loss": loss_sum / num_batches,
        "box_loss": box_sum / num_batches,
        "obj_loss": obj_sum / num_batches,
        "cls_loss": cls_sum / num_batches,
    }
