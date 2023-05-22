import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import cv2

# 경로 설정
data = pd.read_csv("./sandy3.csv")
img_path = "./dataset/"

colors = ["Red", "Orange", "Green", "Yellow", "Brown", "Blue", "Pink", "Turquoise", "Violet", "White"] 
print(len(data))

key = ' '
idx = 0
while key != 'q':
    low = data.loc[idx]
    pred_str = low['PredictionString']
    img_id = low['image_id']

    preds = np.array(pred_str.strip().split(" ")).reshape(-1, 6)
    bboxes = preds[:, 2:].astype(np.float64)
    category = preds[:, 0].flatten().astype(np.int64)

    img = Image.open(img_path + img_id)
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(bboxes):
        draw.rectangle((box[0], box[1], box[2], box[3]), outline = ImageColor.getrgb(colors[category[i]]), width=3)

    cv2.imshow('img',np.array(img))
    key = cv2.waitKey(0)
    if key == 39: # 오른쪽 방향키
        idx = idx + 1 if idx < len(data)-1 else len(data)-1
    elif key == 37: # 왼쪽 방향키
        idx = idx -1 if idx > 0 else 0

cv2.destroyAllWindows()