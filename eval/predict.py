import cv2
import gc
import torch
import numpy as np
from matplotlib import pyplot as plt
from dataset_eval import build_dataloader
from effdet import EfficientDet, DetBenchEval
from effdet.config import get_efficientdet_config
from effdet.efficientdet import HeadNet
from wbf import *


def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1
    config.image_size = 512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    gc.collect()

    net = DetBenchEval(net, config)
    net.eval()
    return net.cuda()


def make_predictions(images, net, score_threshold=0.22):
    images = torch.stack(images).cuda().float()
    predictions = []
    with torch.no_grad():
        det = net(images, torch.tensor([1]*images.shape[0]).float().cuda())
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:, :4]
            scores = det[i].detach().cpu().numpy()[:, 4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]


def run_wbf(predictions, image_index, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from config_eval import Config

    cfg = Config()
    loader = build_dataloader(cfg)
    net = load_net('J:/dl_code/object_detection/wheat_detection/effdet_d5-cutmix-augmix/best-checkpoint-025epoch.bin')
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
    font_size = 1

    for j, (images, image_ids) in enumerate(loader):

        predictions = make_predictions(images, net)

        i = 0
        sample = images[i].permute(1, 2, 0).cpu().numpy()

        boxes, scores, labels = run_wbf(predictions, image_index=i)
        print(labels)
        boxes = boxes.astype(np.int32).clip(min=0, max=511)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        for box, score in zip(boxes, scores):
            cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 0, 0), 1)
            text_location = (box[0] + 2, box[1] - font_size)
            sample = cv2.putText(sample, f'playground {score:.2f}%', text_location, font, fontScale=0.5, color=(0, 255, 0))

        plt.imshow(sample)
        plt.show()
