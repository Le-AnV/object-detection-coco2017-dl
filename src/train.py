from tqdm import tqdm
import torch


def train_one_epoch(model, optimizer, loss_fn, data_loader, device, epoch):
    model.train()

    running_loss = 0.0
    running_box = 0.0
    running_obj = 0.0
    running_cls = 0.0
    num_batches = len(data_loader)

    pbar = tqdm(data_loader, desc=f"Train epoch {epoch}")

    for images, targets in pbar:
        # 1. Chuyển dữ liệu sang device
        images = images.to(device)
        targets = targets.to(device)

        # 2. Forward pass
        preds = model(images)
        # preds là list 3 tensor [B, 85, H, W] chưa qua sigmoid/softmax

        # 3. Tính Loss
        loss, loss_items = loss_fn(preds, targets)

        # 4. Backward pass & Update weight
        optimizer.zero_grad()
        loss.backward()

        # Clip gradient để tránh nổ (Gradient Explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # type: ignore

        optimizer.step()

        # 5. Logging
        loss_val = loss.item()
        running_loss += loss_val
        running_box += loss_items["box_loss"]
        running_obj += loss_items["obj_loss"]
        running_cls += loss_items["cls_loss"]

        pbar.set_postfix(
            {
                "loss": f"{loss_val:.4f}",
                "box": f"{loss_items['box_loss']:.4f}",
                "obj": f"{loss_items['obj_loss']:.4f}",
            }
        )
    return {
        "total_loss": running_loss / num_batches,
        "box_loss": running_box / num_batches,
        "obj_loss": running_obj / num_batches,
        "cls_loss": running_cls / num_batches,
    }
