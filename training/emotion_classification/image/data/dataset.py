from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils import load_augs, load_obj


def get_dataloaders(cfg: DictConfig):
    train_dataset = load_obj(cfg.data.image_classification.dataset_name)(
        cfg.data.image_classification.datapath,
        "train",
        transform=load_augs(cfg.augmentations.train.augs),
    )
    valid_dataset = load_obj(cfg.data.image_classification.dataset_name)(
        cfg.data.image_classification.datapath,
        "test",
        transform=load_augs(cfg.augmentations.valid.augs),
    )

    return (
        DataLoader(
            train_dataset,
            batch_size=cfg.general.batch_size,
            shuffle=True,
            num_workers=cfg.general.n_workers,
            pin_memory=True,
        ),
        DataLoader(
            valid_dataset,
            batch_size=cfg.general.batch_size,
            shuffle=False,
            num_workers=cfg.general.n_workers,
            pin_memory=True,
        ),
    )
