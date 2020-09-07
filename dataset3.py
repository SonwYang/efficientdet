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
import numpy as np
import pandas as pd
import random
from albumentations.pytorch import ToTensor, ToTensorV2
import albumentations as A
import glob
import xml.etree.ElementTree as ET
from random import sample


# 自定义获取数据的方式
class Wheatset(torch.utils.data.Dataset):
    def __init__(self, image_dir, image_size, transforms=None, phase='train', class_dict=None):
        self.image_dir = image_dir
        self.image_list = glob.glob(f'{image_dir}/*.jpg')
        self.phase = phase
        self.transforms = transforms
        self.image_size = image_size
        self.class_dict = class_dict

    def __getitem__(self, index):
        image_id = self.image_list[index]
        if self.phase == 'valid' or random.random() > 0.5:
            image, boxes = self.load_image_and_boxes(index)
        else:
            image, boxes = self.load_cutmix_image_and_boxes(index)

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]  # yxyx: be warning
                    break

        return image, target, image_id

    def __len__(self):
        return len(self.image_list)

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
            class_list.append(self.class_dict[sub.find('name').text])
            difficult_list.append(int(sub.find('difficult').text))
        return np.array(class_list), np.array(boxes_list), np.array(difficult_list)

    def resize_img_boxes(self, img, boxes, input_shape):
        # 调整图片大小
        w, h = img.shape[:2]
        tw, th = input_shape
        image = cv2.resize(img, (tw, th))
        dw = tw / w
        dh = th / h
        new_boxes = []
        for box in boxes.astype(np.float):
            new_box = [box[0] * dh, box[1] * dw, box[2] * dh, box[3] * dw]
            new_boxes.append(new_box)
        return image, np.array(new_boxes).astype(np.int32)

    def load_image_and_boxes(self, index):
        image_path = self.image_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # Read objects in this image (bounding boxes, labels, difficulties)
        xmlPath = image_path[:-3] + 'xml'
        labels, boxes, difficulties = self.parse_xml(xmlPath.replace('JPEGImages', 'Annotation/xml'))
        new_image, new_boxes = self.resize_img_boxes(image, boxes, (self.image_size, self.image_size))
        return new_image, new_boxes

    def load_cutmix_image_and_boxes(self, index, imsize=512):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, len(self.image_list) - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image, result_boxes


def collate_fn(batch):
    return tuple(zip(*batch))


def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(400, 400), height=512, width=512, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_valid_transforms():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

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

    train_data = Wheatset(cfg.root,
                          image_size=cfg.image_size,
                          transforms=get_train_transforms(),
                          phase='train',
                          class_dict=cfg.class_dict)
    val_data = Wheatset(cfg.root,
                        image_size=cfg.image_size,
                        transforms=get_valid_transforms(),
                        phase='validation',
                        class_dict=cfg.class_dict)

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
    # images, targets, images_id = next(iter(train_loader))
    # images = torch.stack(images)
    # print(images.size())
    # images = list(image.to(device) for image in images)
    # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #
    # boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
    # print(f'boxes.size:{boxes.shape}')
    # sample = images[1].permute(1, 2, 0).cpu().numpy()
    #
    # for box in boxes:
    #     cv2.rectangle(sample,
    #                   (box[0], box[1]),
    #                   (box[2], box[3]),
    #                   (0, 1, 0), 2)
    #
    # plt.imshow(sample)
    # plt.show()

    # image, target, image_id = train_data[1]
    # boxes = target['boxes'].cpu().numpy().astype(np.int32)
    #
    # numpy_image = image.permute(1, 2, 0).cpu().numpy()
    #
    # fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    #
    # for box in boxes:
    #     cv2.rectangle(numpy_image, (box[1], box[0]), (box[3], box[2]), (0, 1, 0), 2)
    #
    # ax.set_axis_off()
    # ax.imshow(numpy_image)
    # plt.show()
    for i in range(10):
        print(f'epoch:{i}')
        for images, targets, image_ids in train_loader:
            try:
                 images = torch.stack(images)
            except:
                 print('error', image_ids)
                 continue
        for images, targets, image_ids in val_loader:
            try:
                images = torch.stack(images)
            except:
                print('error', image_ids)
                continue