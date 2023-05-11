import json
from pycocotools.coco import COCO

file_path = '../../../dataset/train.json'
new_file_path = '../dataset/train_area_10.json'

coco = COCO(file_path)

new_annotations = []
new_images = []

for img_id in coco.imgs:
    img_info = coco.loadImgs(ids=[img_id])[0]

    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ids=ann_ids)

    for ann in anns:
        if ann['area'] >= 1000:
            new_annotations.append(ann)

    new_images.append(img_info)

coco.dataset['annotations'] = new_annotations
coco.dataset['images'] = new_images

with open(new_file_path, 'w') as f:
    json.dump(coco.dataset, f)