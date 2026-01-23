import torchvision


def non_max_suppression(prediction, iou_thres=0.45):
    output = []
    for pred in prediction:
        # pred: [N, 6] -> [x1, y1, x2, y2, score, class]
        if pred.shape[0] == 0:
            output.append(pred)
            continue

        boxes = pred[:, :4]
        scores = pred[:, 4]
        class_ids = pred[:, 5]

        # Batched NMS: NMS độc lập cho từng class
        keep = torchvision.ops.batched_nms(boxes, scores, class_ids, iou_thres)

        # Giới hạn số lượng box tối đa, tránh tràn bộ nhớ lúc eval
        if keep.shape[0] > 300:
            keep = keep[:300]

        output.append(pred[keep])

    return output
