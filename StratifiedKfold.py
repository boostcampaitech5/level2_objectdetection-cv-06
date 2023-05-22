import os
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# image
data_path = '/opt/ml/dataset'
annotations_path = os.path.join('/opt/ml/dataset/train.json')
kfold = 5

path = './stratified_kfold'

def main():
    with open(annotations_path, 'r') as f:
        train_json = json.loads(f.read())
        images = train_json['images']
        categories = train_json['categories']
        annotations = train_json['annotations']
    annotations_df = pd.DataFrame.from_dict(annotations)

    x = images
    y = [[0] * len(categories) for _ in range(len(images))]

    for anno in annotations:
        y[anno['image_id']][anno['category_id']] += 1
    mskf = MultilabelStratifiedKFold(n_splits=kfold, shuffle=True)

    if not os.path.exists(path):
        os.mkdir(path)

    for idx, (train_index, val_index) in tqdm(enumerate(mskf.split(x, y)), total=kfold):
        train_dict = dict()
        val_dict = dict()
        for i in ['info', 'licenses', 'categories']:
            train_dict[i] = train_json[i]
            val_dict[i] = train_json[i]
        train_dict['images'] = np.array(images)[train_index].tolist()
        val_dict['images'] = np.array(images)[val_index].tolist()
        train_dict['annotations'] = annotations_df[annotations_df['image_id'].isin(train_index)].to_dict('records')
        val_dict['annotations'] = annotations_df[annotations_df['image_id'].isin(val_index)].to_dict('records')
        train_dir = os.path.join(path, f'train_{idx + 1}.json')
        val_dir = os.path.join(path, f'val_{idx + 1}.json')
        with open(train_dir, 'w') as train_file:
            json.dump(train_dict, train_file)
        with open(val_dir, 'w') as val_file:
            json.dump(val_dict, val_file)
            
main()