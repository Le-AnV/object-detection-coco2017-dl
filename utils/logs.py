def log_to_tensorboard(writer, epoch, train_metrics, val_metrics, optimizer):
    """
    Ghi log metrics lên Tensorboard.
    """
    # 1. Ghi Loss (Group Train vs Val)
    writer.add_scalars(
        "Loss/Total",
        {
            "train": train_metrics["total_loss"],
            "val": val_metrics["val_loss"],
        },
        epoch,
    )

    writer.add_scalars(
        "Loss/Box",
        {
            "train": train_metrics["box_loss"],
            "val": val_metrics["val_box"],
        },
        epoch,
    )

    writer.add_scalars(
        "Loss/Objectness",
        {
            "train": train_metrics["obj_loss"],
            "val": val_metrics["val_obj"],
        },
        epoch,
    )

    writer.add_scalars(
        "Loss/Class",
        {
            "train": train_metrics["cls_loss"],
            "val": val_metrics["val_cls"],
        },
        epoch,
    )

    # 2. Ghi Metrics (mAP)
    writer.add_scalar("Metrics/mAP_50", val_metrics["map_50"], epoch)
    writer.add_scalar("Metrics/mAP_0.5:0.95", val_metrics["map"], epoch)

    # 3. Ghi Learning Rate
    current_lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar("Hyperparameters/Learning_Rate", current_lr, epoch)
