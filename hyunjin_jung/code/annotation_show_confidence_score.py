import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFont
import cv2

# 경로 설정
data = pd.read_csv("./ensemble15.csv")
img_path = "./dataset/"
img_idx = 700

colors = ["Red", "Orange", "Green", "Yellow", "Brown", "Blue", "Pink", "Turquoise", "Violet", "White"] 
category_name = ["General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
font = ImageFont.truetype("arial.ttf", 40) # arial.ttf 글씨체, font_size=15

# annotation 정렬
anns = []
pred_str = data.loc[img_idx]['PredictionString']
img_file = data.loc[img_idx]['image_id']

preds = np.array(pred_str.strip().split(" ")).reshape(-1, 6)
bboxes = preds[:, 2:].astype(np.float64)
category = preds[:, 0].flatten().astype(np.int64)
confidence = preds[:, 1].flatten().astype(np.float64)

for i in range(len(bboxes)):
    anns.append({"bbox" : bboxes[i], "category_id" : category[i], "confidence" : confidence[i]})
anns.sort(key=lambda x : float(x['confidence']), reverse=True)

key = ' '
idx = 0
while key != 'q':

    img = Image.open(img_path + img_file)
    draw = ImageDraw.Draw(img)

    ann = anns[idx]
    box = ann['bbox']
    category_id = ann['category_id']

    draw.rectangle((box[0], box[1], box[2], box[3]), outline = ImageColor.getrgb("Red"), width=3)
    draw.text((box[0], box[1] - 40 ), category_name[category_id], "Red", font=font)
    draw.text((box[0], box[1]), str(ann['confidence']), "Red", font=font)

    img = np.array(img.resize((819,819)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)

    key = cv2.waitKey(0)
    if key == 39: # 오른쪽 방향키
        idx = idx + 1 if idx < len(anns)-1 else len(anns)-1
    elif key == 37: # 왼쪽 방향키
        idx = idx -1 if idx > 0 else 0

cv2.destroyAllWindows()