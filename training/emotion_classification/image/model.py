import os
import logging
from copy import deepcopy
from typing import Any, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch import autocast
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from omegaconf import DictConfig
from tqdm.auto import tqdm

from utils import load_obj
from training.logger import log_meta


def initialize_model(model_cfg: DictConfig) -> Any:
    model = load_obj(model_cfg._target_)(model_cfg.params)
    if model_cfg.freeze:
        for mp in model.parameters():
            mp.requires_grad = False
    if "resnet" in model_cfg._target_.lower():
        model.fc = nn.Linear(model.fc.in_features, model_cfg.n_classes)
        return model
    elif "vit" in model_cfg._target_.lower():
        feature_extractor = load_obj(model_cfg.feature_extractor._target_)(
            model_cfg.feature_extractor.params
        )
        return model, feature_extractor


def _train_epoch(
    cfg: DictConfig,
    model: Any,
    train_dataloader: DataLoader,
    epoch: int,
    criterion: Any,
    optimizer: Any,
    scaler: GradScaler,
    metric: Any,
) -> Tuple[float, float]:
    model.train()

    train_loss = []
    train_score = []
    for batch, targets in tqdm(train_dataloader, desc=f"Epoch: {epoch}"):
        optimizer.zero_grad()

        batch = batch.to(cfg.general.device)
        targets = targets.to(cfg.general.device)

        with autocast(cfg.general.device):
            pred = model(batch)
            loss = criterion(pred, targets)
            score = metric(
                pred.argmax(axis=1).cpu().detach().numpy(),
                targets.cpu().numpy(),
                **cfg.metric.params,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss.append(loss.item())
        train_score.append(score.item())

    train_loss = np.mean(train_loss)
    train_score = np.mean(train_score)

    return train_loss, train_score


def _val_epoch(
    cfg: DictConfig,
    model: Any,
    val_dataloader: DataLoader,
    epoch: int,
    criterion: Any,
    metric: Any,
) -> Tuple[float, float]:
    model.eval()

    val_loss = []
    val_score = []
    for batch, targets in tqdm(val_dataloader, desc=f"Epoch: {epoch}"):
        with torch.no_grad():
            batch = batch.to(cfg.general.device)
            targets = targets.to(cfg.general.device)
            with autocast(cfg.general.device):
                pred = model(batch)
                loss = criterion(pred, targets)
                score = metric(
                    pred.argmax(axis=1).cpu().numpy(),
                    targets.cpu().numpy(),
                    **cfg.metric.params,
                )

            val_loss.append(loss.item())
            val_score.append(metric.item())

    val_loss = np.mean(val_loss)
    val_score = np.mean(score)

    return val_loss, val_score


def train_model(
    cfg: DictConfig,
    logger: Any,
    model: Any,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
) -> None:
    scaler = GradScaler()
    optimizer = load_obj(cfg.optimizer._target_)(
        model.parameters(), **cfg.optimizer.params
    )
    criterion = load_obj(cfg.criterion._target_)(**cfg.criterion.params)
    metric = load_obj(cfg.metric._target_)

    best_score = 0
    best_model = model

    for epoch in range(cfg.general.n_epochs):
        train_loss, train_score = _train_epoch(
            cfg, model, train_dataloader, epoch, criterion, optimizer, scaler, metric
        )
        print("Train loss:", train_loss, "Train score:", train_score)
        val_loss, val_score = _val_epoch(
            cfg, model, valid_dataloader, epoch, criterion, scaler, metric
        )
        print("Val loss:", val_loss, "Val score:", val_score)

        # log best model
        if val_score > best_score:
            best_model = deepcopy(model)
            best_score = val_score

        train_meta = {
            "n_iter": epoch,
            "train": {"loss": train_loss, "metric": train_score},
            "val": {"loss": val_loss, "metric": val_score},
        }
        log_meta(logger, train_meta)

    # Save models
    torch.save(
        best_model.state_dict(),
        os.path.join(cfg.general.checkpoints_folder, "best_vision_classifier.pth"),
    )
    torch.save(
        model.state_dict(),
        os.path.join(cfg.general.checkpoints_folder, "last_vision_classifier.pth"),
    )
    logging.info(f"Model saved to {cfg.general.checkpoints_folder}")
