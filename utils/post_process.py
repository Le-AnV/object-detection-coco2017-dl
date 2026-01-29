import torch


def post_process(outputs, conf_thres=0.25, iou_thres=0.45):
    """
    Chuyển output raw của model thành box cuối cùng.
    Input: List of [B, 85, H, W]
    Output: List of dict {'boxes', 'scores', 'labels'}
    """
    results = []
    # 1. Ghép tất cả layer lại: [B, 8400, 85]
    # 2. Lọc theo confidence threshold
    # 3. Non-Max Suppression (NMS)

    # Giả sử sau khi xử lý ta có:
    keep_boxes = torch.tensor([[100, 100, 200, 200]])  # x1, y1, x2, y2
    keep_scores = torch.tensor([0.95])
    keep_labels = torch.tensor([1])

    results.append({"boxes": keep_boxes, "scores": keep_scores, "labels": keep_labels})
    return results
