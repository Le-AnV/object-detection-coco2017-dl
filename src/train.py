import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()

    # Khởi tạo bộ đếm
    running_metrics = {"total_loss": 0, "box_loss": 0, "obj_loss": 0, "cls_loss": 0}
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")

    for imgs, targets in pbar:
        imgs = imgs.to(device)
        targets = targets.to(device)  # [idx, cls, x, y, w, h] normalized

        optimizer.zero_grad()

        preds = model(imgs)

        # Nhận về total_loss và dictionary chi tiết
        loss, loss_items = criterion(preds, targets)

        loss.backward()

        # Gradient Clipping (Quan trọng để tránh nổ gradient với ResNet)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # type: ignore

        optimizer.step()

        # Cộng dồn metrics
        for k, v in loss_items.items():
            running_metrics[k] += v

        # Update thanh progress bar
        pbar.set_postfix(loss=loss.item())

    # Tính trung bình cho cả epoch
    avg_metrics = {k: v / num_batches for k, v in running_metrics.items()}
    return avg_metrics
