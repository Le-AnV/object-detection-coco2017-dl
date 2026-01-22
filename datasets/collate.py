import torch


def collate_fn(batch):
    """
    Collate function chuẩn cho YOLO Style Training.

    Input:
        batch: List các tuple (image, target_dict) từ Dataset.
               - image: Tensor [3, H, W] (đã qua ToTensor)
               - target_dict: {'boxes': [N, 4] (xyxy), 'labels': [N]}

    Output:
        images: Tensor [Batch_Size, 3, H, W]
        targets: Tensor [Total_Objects, 6]
                 Format: [batch_index, class_id, cx, cy, w, h] (Normalized 0-1)
    """
    images = []
    targets_list = []

    for i, (img, target_dict) in enumerate(batch):
        images.append(img)

        # 1. Lấy kích thước ảnh an toàn để Normalize
        # Giả định img đã qua transforms.ToTensor() -> Shape [3, H, W]
        if isinstance(img, torch.Tensor):
            # Shape [C, H, W] -> Lấy H tại index 1, W tại index 2
            h_img, w_img = img.shape[1], img.shape[2]
        else:
            # Fallback nếu là numpy array [H, W, C]
            h_img, w_img = img.shape[0], img.shape[1]

        # 2. Xử lý targets
        boxes = target_dict.get("boxes", torch.empty(0, 4))  # xyxy format
        labels = target_dict.get("labels", torch.empty(0))

        if boxes.numel() > 0:
            # Chuyển đổi xyxy (pixel) -> xywh (pixel)
            # boxes: [x_min, y_min, x_max, y_max]
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            # Tính center x, center y, width, height
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # 3. NORMALIZE về khoảng [0, 1] (BẮT BUỘC CHO YOLO)
            cx = cx / w_img
            cy = cy / h_img
            w = w / w_img
            h = h / h_img

            # 4. Tạo target tensor [batch_idx, class, cx, cy, w, h]
            num_objs = len(labels)

            # Index của ảnh trong batch (0, 1, 2...)
            batch_indices = torch.full((num_objs, 1), i, dtype=torch.float32)

            combined_targets = torch.cat(
                [
                    batch_indices,  # Cột 0: batch_idx
                    labels.float().unsqueeze(1),  # Cột 1: class_id
                    cx.unsqueeze(1),  # Cột 2: cx (norm)
                    cy.unsqueeze(1),  # Cột 3: cy (norm)
                    w.unsqueeze(1),  # Cột 4: w (norm)
                    h.unsqueeze(1),  # Cột 5: h (norm)
                ],
                dim=1,
            )
            targets_list.append(combined_targets)

    # 5. Stack ảnh và nối targets
    # images -> [Batch, 3, H, W]
    images_tensor = torch.stack(images, 0)

    if len(targets_list) > 0:
        # targets -> [Total_Objects, 6]
        targets_tensor = torch.cat(targets_list, 0)
    else:
        # Trả về tensor rỗng đúng shape [0, 6] nếu batch không có vật thể nào
        targets_tensor = torch.empty((0, 6), dtype=torch.float32)

    return images_tensor, targets_tensor
