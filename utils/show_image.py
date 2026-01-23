import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def denormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize ảnh đã được normalize về dải [0, 1] cho matplotlib

    Args:
        img_tensor: Tensor (C, H, W) đã qua Normalize
        mean: mean values đã dùng khi normalize
        std: std values đã dùng khi normalize

    Returns:
        image: numpy array (H, W, C) trong dải [0, 1]
    """
    # Chuyển từ Tensor (C, H, W) sang Numpy (H, W, C)
    image = img_tensor.permute(1, 2, 0).cpu().numpy()

    # Thực hiện phép tính ngược: image * std + mean
    image = (image * np.array(std)) + np.array(mean)

    # Chốt chặn dải giá trị về [0, 1] để plt.imshow không báo lỗi clipping
    image = np.clip(image, 0, 1)
    return image


def display_image(img, target, israw=True, class_names=None):
    """
    Hiển thị ảnh với bounding boxes và labels (tương thích với Letterboxing)

    Args:
        img: Tensor từ Dataset (C, H, W) - đã qua ToTensor() và Normalize()
        target: dict chứa 'boxes' và 'labels', có thể có 'scores'
        israw: True = denormalize về màu gốc, False = chỉ permute không denormalize
        class_names: list/dict tên các class theo index liên tục
        show_confidence: có hiển thị confidence score không (cho inference)
        conf_threshold: ngưỡng confidence để lọc boxes
    """
    # Xử lý ảnh
    if israw:
        # Ảnh đã qua normalize, cần denormalize để hiển thị màu gốc
        image = denormalize(img)
    else:
        # Ảnh đã qua normalize, chỉ permute (C,H,W) -> (H,W,C), KHÔNG denormalize
        image = img.permute(1, 2, 0).cpu().numpy()

    # Tạo figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    # Lấy boxes và labels
    boxes = target.get("boxes", [])
    labels = target.get("labels", [])
    scores = target.get("scores", None)

    for idx, (box, label_id) in enumerate(zip(boxes, labels)):
        # Chuyển box sang numpy nếu nó đang là tensor
        if hasattr(box, "numpy"):
            box = box.cpu().numpy()

        if hasattr(label_id, "item"):
            label_id = label_id.item()

        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1

        # Bỏ qua box không hợp lệ
        if width <= 0 or height <= 0:
            continue

        # Vẽ khung hình chữ nhật
        rect = patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)

        # Tạo label text
        if class_names is not None:
            label_idx = int(label_id)
            if isinstance(class_names, dict):
                name = class_names.get(label_idx, f"Class {label_idx}")
            elif isinstance(class_names, list):
                name = (
                    class_names[label_idx]
                    if label_idx < len(class_names)
                    else f"Class {label_idx}"
                )
            else:
                name = str(label_idx)
            label_text = name
        else:
            label_text = f"Class {int(label_id)}"

        # Hiển thị label
        ax.text(
            x1,
            y1 - 2,
            label_text,
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="blue", alpha=0.6, pad=2),
        )

    plt.axis("off")
    plt.show()
