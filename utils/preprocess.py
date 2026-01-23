import cv2
import numpy as np
import torch
import torchvision.transforms as T


def letterbox(image, img_size=640, color=(114, 114, 114)):
    # Resize ảnh giữ nguyên tỉ lệ, pad về hình vuông
    h, w = image.shape[:2]
    scale = min(img_size / w, img_size / h)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    # Tạo canvas và pad ảnh vào giữa
    canvas = np.full((img_size, img_size, 3), color, dtype=np.uint8)
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # Lưu thông tin để scale bbox ngược lại
    meta = {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "orig_size": (h, w)}

    return canvas, meta


def get_normalize_transform():
    # Normalize theo ImageNet
    return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def preprocess_image(image, img_size=640):
    # Tiền xử lý ảnh cho inference
    image, meta = letterbox(image, img_size)
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image = get_normalize_transform()(image)

    return image, meta
