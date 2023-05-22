import json
import numpy as np
import pandas as pd
import math
import ensemble_boxes

# 이 이상의 score를 갖는 box들을 레이블로 만듦
THRESHOLD = 0.22
'''
read submission csv file and create COCO style label (for test data)
'''

# 레이블로 쓰고 싶은 inference
pseudo_df = pd.read_csv("ellie.csv")
# print(pseudo_df.head())

# 새로운 json
pseudo_coco = {"info": {}, "licenses": [], "images": [], "categories": [], "annotations": []}

"""
PredictionString = (label, score, xmin, ymin, xmax, ymax) 형식

format per annotation
{
      "image_id": 0,            --> 이미지마다 고유
      "category_id": 0,         --> class
      "area": 257301.66,        --> 계산해서 넣기
      "bbox": [                 --> (xmin, ymin, w, h)
        197.6,
        193.7,
        547.8,
        469.7
      ],
      "iscrowd": 0,             --> 다 0
      "id": 0                   --> annotation마다 고유, 1씩 증가
    }
"""

# pseudo_df으로 pseudo_coco 만들기

id = 23144      # train anno 개수
for i in range(len(pseudo_df['PredictionString'])):

    preds = pseudo_df['PredictionString'][i].split()

    # 사진 당 annotation 6개 이하만 뽑기(train data는 평균 5였음) -> THRESHOLD 보다 큰 score만 담기로 변경
    first = True
    scores = []
    annotations = []
    for j in range(0, len(preds), 6):
        # assertion 에러 넣어도 좋을 것 같음
        if float(preds[j+1]) > THRESHOLD or first:
          first = False
          scores.append(float(preds[j+1]))
          anno = {}
          anno["image_id"] = i + 4883
          anno["category_id"] = int(preds[j])
          w = math.floor((float(preds[j+4]) - float(preds[j+2]))*10)/10   # 반올림 해야 할까
          h = math.floor((float(preds[j+5]) - float(preds[j+3]))*10)/10
          anno["area"] = w * h
          anno["bbox"] = [float(preds[j+2]), float(preds[j+3]), w, h]
          anno["iscrowd"] = 0
          anno["id"] = id
          annotations.append(anno)
          id += 1
          if w+float(preds[j+2]) > 1024 or h+float(preds[j+3]) > 1024:
            print("error")
    boxes = [(np.array(anno['bbox'])/(1023)).tolist() for anno in annotations]
    scores = np.array(scores)
    labels = [np.ones(scores.shape[0]).astype(int).tolist() for anno in annotations]
    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores.tolist(), labels, weights=None, iou_thr=0.43, skip_box_thr=0.43)
    boxes = boxes*(1023)
    pseudo_coco["annotations"].extend(annotations)

print(id-23144)   # 추가된 annotations 수 (test 사진은 총 4870장)


with open("train_pseudo.json", "w", encoding="utf-8") as f:
    train_file = open("train.json", "r")
    train_json = json.load(train_file)

    test_file = open("test.json", "r")
    test_json = json.load(test_file)

    train_file.close()
    test_file.close()

    for i in range(len(test_json["images"])):
        test_json["images"][i]["id"] += 4883

    pseudo_coco["info"] = train_json["info"]
    pseudo_coco["licenses"] = train_json["licenses"]
    pseudo_coco["images"] = train_json["images"]
    pseudo_coco["images"].extend(test_json["images"])
    pseudo_coco["categories"] = train_json["categories"]
    train_json["annotations"].extend(pseudo_coco["annotations"])
    pseudo_coco["annotations"] = train_json["annotations"]

    json.dump(pseudo_coco, f, indent="\t")