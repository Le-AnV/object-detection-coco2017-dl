import os
import torch


def save_model(
    model,
    optimizer,
    epoch,
    val_metrics,
    best_metric,
    save_dir="checkpoints",
    monitor="map_50",
    mode="max",
):
    os.makedirs(save_dir, exist_ok=True)

    model_state = model.state_dict()

    # So sánh metric hiện tại với best
    current_metric = val_metrics[monitor]
    is_best = (
        current_metric > best_metric if mode == "max" else current_metric < best_metric
    )

    # Lưu checkpoint mới nhất (last)
    last_path = os.path.join(save_dir, "last.pth")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "best_metric": best_metric,
    }
    torch.save(checkpoint, last_path)

    # Nếu tốt hơn thì lưu thêm best
    if is_best:
        best_path = os.path.join(save_dir, "best.pth")
        torch.save(checkpoint, best_path)
        print(f"Saved best model at epoch {epoch} with {monitor}: {current_metric:.4f}")
        best_metric = current_metric

    return best_metric, is_best


def load_model(model, path="checkpoints/best_model.pth", map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model
