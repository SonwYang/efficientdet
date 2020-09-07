# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import os
import cv2
import re
import argparse
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import xml.etree.ElementTree as ET
import numpy as np
from random import sample
import pandas as pd
from albumentations.pytorch import ToTensorV2
import albumentations as A
from albumentations import (HorizontalFlip, ShiftScaleRotate, VerticalFlip, Normalize, Flip, Compose,
                            Resize, GaussNoise)
import glob

# class_dict = {'playground': 1}
class_dict = {'aircraft': 1, 'RBC': 2, 'Platelets': 3}

# 自定义获取数据的方式
class Wheatset(torch.utils.data.Dataset):
    def __init__(self, imglist, phase='train'):
        self.imglist = imglist
        self.transforms = get_transforms(phase)

    def load_image_and_boxes(self, index):
        imagePath = self.imglist[index]
        image_id = os.path.split(imagePath)[-1][:-3]
        # Read image
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Read objects in this image (bounding boxes, labels, difficulties)
        xmlPath = imagePath[:-3] + 'xml'
        labels, boxes, difficulties = self.parse_xml(xmlPath.replace('JPEGImages', 'Annotation/xml'))
        return image, boxes, labels, image_id

    def __getitem__(self, index):
        image, boxes, labels, image_id = self.load_image_and_boxes(index)
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        labels = torch.LongTensor(labels)  # (n_objects)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        #suppose all instances are not crowd
        iscrowd = torch.zeros_like(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self):
        return self.imglist

    def parse_xml(self, path):
        tree = ET.parse(path)
        root = tree.findall('object')
        class_list = []
        boxes_list = []
        difficult_list = []
        for sub in root:
            xmin = float(sub.find('bndbox').find('xmin').text)
            xmax = float(sub.find('bndbox').find('xmax').text)
            ymin = float(sub.find('bndbox').find('ymin').text)
            ymax = float(sub.find('bndbox').find('ymax').text)
            # if ymax > 915 or xmax > 1044 or xmin < 0 or ymin < 0:
            #     print(xmin, ymin, xmax, ymax, path)
            boxes_list.append([xmin, ymin, xmax, ymax])
            class_list.append(class_dict[sub.find('name').text])
            difficult_list.append(int(sub.find('difficult').text))
        return np.array(class_list), np.array(boxes_list), np.array(difficult_list)



def collate_fn(batch):
    return tuple(zip(*batch))


def get_transforms(phase):
    list_transforms = []
    if phase == 'train':
        list_transforms.extend( [
            A.Resize(height=512, width=512, p=1),
            A.RandomSizedCrop(min_max_height=(500, 500), height=512, width=512, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
        ],
       )
    else:
        list_transforms.extend([
            A.Resize(height=512, width=512, p=1),
        ])
    list_transforms.extend(
            [
                 ToTensorV2(),
            ])
    list_trfms = Compose(list_transforms,
                         bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    return list_trfms


# this function will take the dataframe and vertically stack the image ids
# with no bounding boxes
def process_bbox(df, train_dir):
    ids = []
    values = []
    imd = np.unique(df['image_id'])
    df['bbox'] = df['bbox'].apply(lambda x: eval(x))
    for image_id in os.listdir(train_dir):
        image_id = image_id.split('.')[0]
        if image_id not in imd :
            ids.append(image_id)
            values.append(str([-1,-1,-1,-1]))
    new_df = {'image_id':ids, 'bbox':values}
    new_df = pd.DataFrame(new_df)
    df = df[['image_id','bbox']]
    df.append(new_df)
    df = df.sample(frac=1).reset_index(drop=True)
    df['x'] = df['bbox'].apply(lambda x: x[0])
    df['y'] = df['bbox'].apply(lambda x: x[1])
    df['w'] = df['bbox'].apply(lambda x: x[2])
    df['h'] = df['bbox'].apply(lambda x: x[3])

    df.drop(columns=['bbox'],inplace=True)
    return df


def build_loader(cfg):
    imglist = glob.glob(f'{cfg.root}/*.jpg')
    indices = list(range(len(imglist)))
    indices = sample(indices, len(indices))
    split = int(np.floor(0.15 * len(imglist)))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    print(f'Total images {len(imglist)}')
    print(f'No of train images {len(train_idx)}')
    print(f'No of validation images {len(valid_idx)}')

    train_data = Wheatset(imglist, phase='train')
    val_data = Wheatset(imglist, phase='validation')

    train_loader = DataLoader(train_data,
                                  batch_size=cfg.batch_size,
                                  sampler=train_sampler,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_loader = DataLoader(val_data,
                                  batch_size=cfg.batch_size,
                                  sampler=valid_sampler,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    return train_loader, val_loader


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from config import Config
    cfg = Config()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader, val_loader = build_loader(cfg)
    # for images, targets, images_id in train_loader:
    #     print('')
    images, targets, images_id = next(iter(train_loader))
    images = torch.stack(images)
    print(images.size())
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
    print(f'boxes.size:{boxes.shape}')
    image = images[0].permute(1, 2, 0).cpu().numpy()
    sample = image.copy()
    for box in boxes:
        print(box, box.dtype)
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (1, 0, 0), 1)

    plt.imshow(sample)
    plt.show()
