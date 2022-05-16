from typing import Any
import torch.nn as nn
from training.utils import load_obj
from training.trainers.base_trainer import Trainer


class VisionTrainer(Trainer):
    def _initialize_model(self) -> Any:
        model_cfg = self.cfg.model
        model = load_obj(model_cfg._target_)(model_cfg.params)
        if model_cfg.freeze:
            for mp in model.parameters():
                mp.requires_grad = False
        if "resnet" in model_cfg._target_.lower():
            model.fc = nn.Linear(model.fc.in_features, model_cfg.n_classes)
            return model
        elif "mobilenet" in model_cfg._target_.lower():
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, model_cfg.n_classes
            )
            return model
        raise ValueError("No such model!")
