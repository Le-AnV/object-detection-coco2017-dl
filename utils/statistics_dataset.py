import pandas as pd
from pycocotools.coco import COCO


def load_coco(ann_file):
    """Load file annotation COCO từ đường dẫn json"""
    return COCO(ann_file)


def count_instances_per_category(coco):
    """Thống kê số lượng instance (object) theo từng category"""
    categories = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    data = []
    for cat_id, name in cat_id_to_name.items():
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        data.append({"category_name": name, "num_instances": len(ann_ids)})

    return pd.DataFrame(data).sort_values("num_instances", ascending=True)


def count_images_per_category(coco):
    """Thống kê số lượng image theo từng category"""
    categories = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    data = []
    for cat_id, name in cat_id_to_name.items():
        img_ids = coco.getImgIds(catIds=[cat_id])
        data.append({"category_name": name, "num_images": len(img_ids)})

    return pd.DataFrame(data).sort_values("num_images", ascending=True)


def bbox_size_distribution(coco):
    """Thống kê bbox theo kích thước small / medium / large"""
    anns = coco.loadAnns(coco.getAnnIds())

    sizes = {"small": 0, "medium": 0, "large": 0}

    for ann in anns:
        area = ann["area"]
        if area < 32**2:
            sizes["small"] += 1
        elif area < 96**2:
            sizes["medium"] += 1
        else:
            sizes["large"] += 1

    return pd.DataFrame(sizes.items(), columns=["size", "count"])
