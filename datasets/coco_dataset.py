import os
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from utils.preprocess import letterbox


class COCODataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        img_size: int = 224,
        transform=None,
        selected_cat_ids=None,
    ):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform

        if selected_cat_ids is None:
            raise ValueError("selected_cat_ids must be provided")

        self.selected_cat_ids = sorted(selected_cat_ids)

        # Tạo map ID -> Index (0..N-1)
        self.coco_id_to_idx = {
            coco_id: idx for idx, coco_id in enumerate(self.selected_cat_ids)
        }
        self.idx_to_coco_id = {idx: id for id, idx in self.coco_id_to_idx.items()}
        self.num_classes = len(self.selected_cat_ids)

        # getImgIds trả về list ảnh chứa các catIds đó
        valid_img_ids = set()
        for cat_id in self.selected_cat_ids:
            valid_img_ids.update(self.coco.getImgIds(catIds=[cat_id]))

        self.img_ids = sorted(list(valid_img_ids))

        print(
            f"Dataset created with {len(self.img_ids)} images containing {self.num_classes} classes."
        )

    def __len__(self):
        return len(self.img_ids)

    def _load_image(self, img_id):
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info["file_name"])
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, info

    def _load_target(self, img_id, meta):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        scale = meta["scale"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]

        for ann in anns:
            if ann["category_id"] not in self.coco_id_to_idx:
                continue

            if ann.get("iscrowd", 0) == 1:
                continue

            x, y, w, h = ann["bbox"]

            # Transform bbox
            x1 = x * scale + pad_x
            y1 = y * scale + pad_y
            x2 = (x + w) * scale + pad_x
            y2 = (y + h) * scale + pad_y

            # Clip bbox to image bounds
            x1 = max(0, min(x1, self.img_size))
            y1 = max(0, min(y1, self.img_size))
            x2 = max(0, min(x2, self.img_size))
            y2 = max(0, min(y2, self.img_size))

            # Validate bbox
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.coco_id_to_idx[ann["category_id"]])

        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        return {"boxes": boxes, "labels": labels}

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image, _ = self._load_image(img_id)

        # Sử dụng kỹ thuật letterbox để resize image
        image, meta = letterbox(image, img_size=self.img_size)
        target = self._load_target(img_id, meta)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = image / 255.0
        image = torch.clamp(image, 0.0, 1.0)
        if self.transform:
            image = self.transform(image)
        return image, target
