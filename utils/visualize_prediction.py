import cv2
from matplotlib import pyplot as plt

import torch

from .preprocess import preprocess_image
from .postprocess import decode_predictions, apply_nms
import matplotlib.patches as patches
from .show_image import denormalize


# Detect image
def detect_and_visualize(
    model,
    image_path,
    device,
    class_names,
    img_size=224,
    conf_thres=0.25,
    iou_thres=0.5,
):
    # 1. Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type: ignore

    # 2. Preprocess
    img_tensor, meta = preprocess_image(img, img_size=img_size)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 3. Inference
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)

    # 4. Decode + NMS
    preds = decode_predictions(outputs, conf_thres=conf_thres)
    final_boxes = apply_nms(preds, iou_thres=iou_thres)

    boxes = final_boxes[0]

    # 5. Scale boxes về ảnh gốc
    if boxes is not None and boxes.numel() > 0:
        scale = meta["scale"]
        pad_x, pad_y = meta["pad_x"], meta["pad_y"]

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes[:, :4] /= scale

    # 6. Visualize
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")

    if boxes is not None and boxes.numel() > 0:
        for x1, y1, x2, y2, score, cls in boxes:
            cls = int(cls)
            label = f"{class_names[cls]}: {score:.2f}"

            plt.gca().add_patch(
                plt.Rectangle(  # type: ignore
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )
            plt.text(
                x1,
                y1 - 3,
                label,
                color="lime",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.5),
            )

    plt.show()

    return boxes


# Display gt and detect
def display_gt_and_pred(
    img,
    gt_target,
    pred_target,
    class_names=None,
):
    """
    img: Tensor (C,H,W)
    gt_target: dict {"boxes","labels"}
    pred_target: dict {"boxes","labels","scores"}
    """

    # dùng lại denormalize của bạn
    image = denormalize(img)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    # -------- Ground Truth (RED)
    for box, label in zip(gt_target["boxes"], gt_target["labels"]):
        box = box.cpu().numpy()
        label = int(label)

        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        name = class_names[label] if class_names else str(label)
        ax.text(
            x1,
            y1 - 2,
            f"GT: {name}",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="red", alpha=0.6),
        )

    # -------- Prediction (GREEN)
    for i, (box, label) in enumerate(zip(pred_target["boxes"], pred_target["labels"])):
        box = box.cpu().numpy()
        label = int(label)

        score = pred_target["scores"][i]

        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

        name = class_names[label] if class_names else str(label)
        ax.text(
            x1,
            y1,
            f"Pred: {name} {score:.2f}",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="green", alpha=0.6),
        )

    plt.axis("off")
    plt.show()


def preds_to_target(boxes):
    """
    boxes: Tensor [N,6] = x1,y1,x2,y2,score,cls
    return: dict cho display_image()
    """
    if boxes is None or boxes.numel() == 0:
        return {"boxes": [], "labels": [], "scores": []}

    return {
        "boxes": boxes[:, :4],
        "scores": boxes[:, 4],
        "labels": boxes[:, 5].long(),
    }
