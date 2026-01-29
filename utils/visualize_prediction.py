import cv2
from matplotlib import pyplot as plt


def visualize_prediction(image, prediction, class_names):
    """
    Vẽ box lên ảnh.
    image: numpy array [H, W, 3] (RGB)
    prediction: dict {'boxes', 'scores', 'labels'}
    """
    img = image.copy()
    for box, score, label in zip(
        prediction["boxes"], prediction["scores"], prediction["labels"]
    ):
        x1, y1, x2, y2 = box.int().tolist()
        cls_name = class_names[label]

        # Vẽ Box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Viết Label + Score
        text = f"{cls_name}: {score:.2f}"
        cv2.putText(
            img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
