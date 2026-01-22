import os
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from utils.preprocess import letterbox


class COCODataset(Dataset):
    """
    Dataset COCO
    - Đọc ảnh
    - Resize + pad bằng letterbox
    - Chuyển đổi bbox theo letterbox
    - Trả dữ liệu đúng format cho model [3, H, W]
    """

    def __init__(
        self, img_dir: str, ann_file: str, img_size: int = 640, transform=None
    ):
        # Load annotation COCO
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform

        # Danh sách image id
        self.img_ids = list(self.coco.imgs.keys())

        # Map category_id COCO -> label liên tục (0..N-1)
        self.coco_cat_ids = sorted(self.coco.getCatIds())
        self.coco_id_to_idx = {
            coco_id: idx for idx, coco_id in enumerate(self.coco_cat_ids)
        }

    # Số lượng ảnh trong dataset
    def __len__(self):

        return len(self.img_ids)

    # Load image
    def _load_image(self, img_id):
        # Lấy thông tin ảnh từ COCO
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info["file_name"])

        # Đọc ảnh
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, info

    # Load & transform annotations
    def _load_target(self, img_id, meta):
        # Lấy annotation của ảnh
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        # Thông tin scale & padding từ letterbox
        scale = meta["scale"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]

        for ann in anns:
            # Bỏ qua crowd
            if ann.get("iscrowd", 0) == 1:
                continue

            x, y, w, h = ann["bbox"]

            # Chuyển bbox theo letterbox
            x1 = x * scale + pad_x
            y1 = y * scale + pad_y
            x2 = (x + w) * scale + pad_x
            y2 = (y + h) * scale + pad_y

            # Giới hạn bbox trong ảnh
            x1 = max(0, min(x1, self.img_size))
            y1 = max(0, min(y1, self.img_size))
            x2 = max(0, min(x2, self.img_size))
            y2 = max(0, min(y2, self.img_size))

            # Chỉ giữ bbox hợp lệ
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.coco_id_to_idx[ann["category_id"]])

        # Trả tensor rỗng nếu ảnh không có object
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        return {"boxes": boxes, "labels": labels}

    # Get item
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # Load ảnh gốc
        image, _ = self._load_image(img_id)

        # Resize + pad
        image, meta = letterbox(image, self.img_size)

        # Load và transform bbox
        target = self._load_target(img_id, meta)

        # Chuyển ảnh sang tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if self.transform:
            image = self.transform(image)

        return image, target
