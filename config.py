import torch
import glob

class Config(object):
    root = 'J:/dl_dataset/object_detection/playground/JPEGImages'
    CSV_PATH = f'{root}/train.csv'
    IMG_PATH = f'{root}/train'

    class_dict = {'playground': 1}
    model_name = 'tf_efficientdet_d5'
    phi = model_name.split("_")[-1]
    bench_task = 'train'
    image_size = 512
    initial_checkpoint = glob.glob(f'./model_data/tf_efficientdet_{phi}*.pth')[0]
    batch_size = 2
    num_workers = 0
    n_epochs = 30
    lr = 0.001
    num_classes = 1

    folder = f'effdet_{phi}-cutmix-augmix'

    verbose = True
    verbose_step = 1

    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=3,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )