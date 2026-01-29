def log_to_tensorboard(writer, epoch, train_metrics, val_metrics):
    """
    Ghi log metrics lên Tensorboard.
    Lưu ý: Key phải khớp với return của train_one_epoch/validate_one_epoch
    """
    # 1. Ghi Loss
    writer.add_scalars(
        "Loss/Total",
        {"train": train_metrics["total_loss"], "val": val_metrics["total_loss"]},
        epoch,
    )

    writer.add_scalars(
        "Loss/Box",
        {"train": train_metrics["box_loss"], "v al": val_metrics["box_loss"]},
        epoch,
    )

    writer.add_scalars(
        "Loss/Objectness",
        {"train": train_metrics["obj_loss"], "val": val_metrics["obj_loss"]},
        epoch,
    )

    writer.add_scalars(
        "Loss/Class",
        {"train": train_metrics["cls_loss"], "val": val_metrics["cls_loss"]},
        epoch,
    )

    # 2. Ghi Metrics (mAP)
    writer.add_scalar("Metrics/mAP_50", val_metrics["map_50"], epoch)
    writer.add_scalar("Metrics/mAP_0.5:0.95", val_metrics["map"], epoch)
