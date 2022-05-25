import os
import logging
from copy import deepcopy
from typing import Any, Tuple

import torch
import numpy as np
from torch import autocast
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from omegaconf import DictConfig
from tqdm.auto import tqdm

from training.utils import load_obj, log_meta


class Trainer:
    def __init__(self, cfg: DictConfig, logger: Any) -> None:
        self.cfg = cfg
        self._logger = logger
        self._model = self._initialize_model()
        self._model.to(self.cfg.general.device)
        self._optimizer = load_obj(cfg.optimizer._target_)(
            self._model.parameters(), **cfg.optimizer.params
        )
        if cfg.criterion.params.weight:
            weight = torch.as_tensor(
                cfg.criterion.params.weight, device=cfg.general.device
            )
        else:
            weight = None
        del cfg.criterion.params.weight
        self._criterion = load_obj(cfg.criterion._target_)(
            weight=weight, **cfg.criterion.params
        )
        self._metric = load_obj(cfg.metric._target_)
        self._scaler = GradScaler()

    def _initialize_model(self) -> Any:
        raise NotImplementedError()

    def _save_model(self, model: Any, name: str, data_loader: DataLoader) -> None:
        torch.save(
            model.state_dict(),
            os.path.join(self.cfg.general.checkpoint_path, name + ".pt"),
        )
        if self.cfg.general.save_scripted_model:
            model_scripted = torch.jit.trace(
                model,
                data_loader.dataset[0][0].unsqueeze(0).to(self.cfg.general.device),
            )
            model_scripted.save(
                os.path.join(self.cfg.general.checkpoint_path, name + ".torchscript.pt")
            )

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        self._model.train()

        train_loss = []
        train_score = []
        for batch, targets in tqdm(train_loader, desc=f"Epoch: {epoch}"):
            self._optimizer.zero_grad()

            batch = batch.to(self.cfg.general.device)
            targets = targets.to(self.cfg.general.device)

            if self.cfg.general.amp:
                with autocast(self.cfg.general.device):
                    pred = self._model(batch)
                    loss = self._criterion(pred, targets)
                    score = self._metric(
                        pred.argmax(axis=1).cpu().detach().numpy(),
                        targets.cpu().numpy(),
                        **self.cfg.metric.params,
                    )
                self._scaler.scale(loss).backward()
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                pred = self._model(batch)
                loss = self._criterion(pred, targets)
                score = self._metric(
                    pred.argmax(axis=1).cpu().detach().numpy(),
                    targets.cpu().numpy(),
                    **self.cfg.metric.params,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.5)
                self._optimizer.step()

            train_loss.append(loss.item())
            train_score.append(score.item())

        train_loss = np.mean(train_loss)
        train_score = np.mean(train_score)

        return train_loss, train_score

    def _val_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        self._model.eval()

        val_loss = []
        val_score = []
        for batch, targets in tqdm(val_loader, desc=f"Epoch: {epoch}"):
            with torch.no_grad():
                batch = batch.to(self.cfg.general.device)
                targets = targets.to(self.cfg.general.device)

                if self.cfg.general.amp:
                    with autocast(self.cfg.general.device):
                        pred = self._model(batch)
                        loss = self._criterion(pred, targets)
                        score = self._metric(
                            pred.argmax(axis=1).cpu().numpy(),
                            targets.cpu().numpy(),
                            **self.cfg.metric.params,
                        )
                else:
                    pred = self._model(batch)
                    loss = self._criterion(pred, targets)
                    score = self._metric(
                        pred.argmax(axis=1).cpu().numpy(),
                        targets.cpu().numpy(),
                        **self.cfg.metric.params,
                    )

                val_loss.append(loss.item())
                val_score.append(score.item())

        val_loss = np.mean(val_loss)
        val_score = np.mean(val_score)

        return val_loss, val_score

    def train(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
    ) -> None:
        best_score = 0
        best_model = self._model

        for epoch in range(1, self.cfg.general.n_epochs + 1):
            train_loss, train_score = self._train_epoch(train_dataloader, epoch)
            logging.info(f"Train loss: {train_loss:.4f} Train score: {train_score:.4f}")
            val_loss, val_score = self._val_epoch(valid_dataloader, epoch)
            logging.info(f"Valid loss: {val_loss:.4f} Valid score: {val_score:.4f}")

            # log best model
            if val_score > best_score:
                best_model = deepcopy(self._model)
                best_score = val_score
                self._save_model(best_model, "best", valid_dataloader)

            train_meta = {
                "n_iter": epoch,
                "train": {"loss": train_loss, "metric": train_score},
                "val": {"loss": val_loss, "metric": val_score},
            }
            log_meta(self._logger, train_meta)

        self._save_model(self._model, "last", valid_dataloader)
        logging.info(
            "Model saved to "
            f"{os.path.join(self.cfg.general.artifacts_dir, self.cfg.general.run_name, self.cfg.general.checkpoint_path)}"
        )
        logging.info(f"Best score: {best_score:.4f}")
