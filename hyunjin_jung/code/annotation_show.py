import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import cv2
from pycocotools.coco import COCO

# 경로 설정
data = COCO("./dataset/train.json")
img_path = "./dataset/"

colors = ["Red", "Orange", "Green", "Yellow", "Brown", "Blue", "Pink", "Turquoise", "Violet", "White"] 
category_name = ["General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

# annotation 정렬
anns = data.loadAnns(data.getAnnIds())
anns.sort(key=lambda x : int(x['area']))

key = ' '
idx = 700
while key != 'q':

    img_id = anns[idx]['image_id']
    img_file = data.loadImgs(img_id)[0]['file_name']
    box = anns[idx]['bbox']
    category = anns[idx]['category_id']

    img = Image.open(img_path + img_file)
    draw = ImageDraw.Draw(img)
    draw.rectangle((box[0], box[1], box[0] + box[2], box[1] + box[3]), outline = ImageColor.getrgb('Red'), width=3)
    draw.text((box[0], box[1] - 10 ), category_name[category], "Red")
    trim = ""
    if box[0] <= 0 or box[1] <= 0 or box[0] + box[2] >= 1024 or box[1] + box[3] >= 1024:
        trim = "trim!"

    img = img.resize((819,819))
    cv2.imshow('image', np.array(img))
    print(idx," : ", anns[idx]['area'], trim)

    key = cv2.waitKey(0)
    if key == 39: # 오른쪽 방향키
        idx = idx + 1 if idx < len(anns)-1 else len(anns)-1
    elif key == 37: # 왼쪽 방향키
        idx = idx -1 if idx > 0 else 0

cv2.destroyAllWindows()