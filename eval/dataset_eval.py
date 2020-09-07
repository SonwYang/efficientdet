import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from config_eval import Config



def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)


class DatasetRetriever(Dataset):

    def __init__(self, image_ids, cfg, transforms=None):
        super().__init__()
        self.cfg = cfg
        self.image_ids = image_ids
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        return image, image_id

    def __len__(self) -> int:
        return len(self.image_ids)


def build_dataloader(cfg):
    dataset = DatasetRetriever(
        image_ids=glob(f'{cfg.DATA_ROOT_PATH}/*.jpg'),
        cfg=cfg,
        transforms=get_valid_transforms()
    )

    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn
    )

    return data_loader


if __name__ == '__main__':
    cfg = Config()
    loader = build_dataloader(cfg)
    for image, image_id in loader:
        print('...')