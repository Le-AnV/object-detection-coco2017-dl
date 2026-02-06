import os
import torch


def save_model(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    val_metrics,
    best_metric,
    save_dir="checkpoints",
    monitor="map_50",
    mode="max",
):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Xử lý DataParallel (Unwrap)
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    # 2. So sánh metric
    current_metric = val_metrics[monitor]
    is_best = (
        current_metric > best_metric if mode == "max" else current_metric < best_metric
    )

    # 3. Tạo Checkpoint Dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler else None
        ),  # Lưu tiến trình LR
        "scaler_state_dict": (
            scaler.state_dict() if scaler else None
        ),  # Lưu scale factor của FP16
        "val_metrics": val_metrics,
        "best_metric": best_metric,
    }

    # 4. Lưu Last Model
    last_path = os.path.join(save_dir, "last.pth")
    torch.save(checkpoint, last_path)

    # 5. Lưu Best Model
    if is_best:
        best_path = os.path.join(save_dir, "best.pth")
        torch.save(checkpoint, best_path)
        print(f"Saved BEST model at epoch {epoch} | {monitor}: {current_metric:.4f}")
        best_metric = current_metric

    return best_metric, is_best


# Load model from PATH
def load_model(model, path="checkpoints/best.pth", map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model
