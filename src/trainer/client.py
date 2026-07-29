from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import flwr as fl
import torch

from torch.utils.data import DataLoader

from .local_trainer import LocalTrainer


class FlowerClient(fl.client.NumPyClient):

    def __init__(
        self,
        cid: int,
        model: torch.nn.Module,
        train_dataset,
        val_dataset,
        optimizer,
        criterion,
        device,
        config,
        scheduler=None,
    ):

        self.cid = cid

        self.model = model
        self.device = device

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.config = config

        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers

        self.local_epochs = config.federated.local_epochs

        self.trainer = LocalTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            num_classes=config.model.num_classes,
            aggregation=config.evaluation.aggregation,
        )

    ####################################################################
    # DATALOADERS
    ####################################################################

    def _train_loader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def _val_loader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    ####################################################################
    # PARAMETERS
    ####################################################################

    def get_parameters(self, config):

        return [
            val.detach().cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):

        params_dict = zip(
            self.model.state_dict().keys(),
            parameters,
        )

        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in params_dict
            }
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

    ####################################################################
    # FIT
    ####################################################################

    def fit(
        self,
        parameters,
        config,
    ):

        self.set_parameters(parameters)

        history = []

        for epoch in range(self.local_epochs):

            if hasattr(
                self.train_dataset,
                "set_epoch",
            ):
                self.train_dataset.set_epoch(epoch)

            metrics = self.trainer.train_epoch(
                self._train_loader()
            )

            metrics["epoch"] = epoch + 1

            history.append(metrics)

        last_metrics = history[-1]

        last_metrics["client_id"] = self.cid

        return (
            self.get_parameters({}),
            len(self.train_dataset),
            last_metrics,
        )

    ####################################################################
    # EVALUATE
    ####################################################################

    def evaluate(
        self,
        parameters,
        config,
    ):

        self.set_parameters(parameters)

        metrics = self.trainer.validate(
            self._val_loader()
        )

        metrics["client_id"] = self.cid

        return (
            float(metrics["loss"]),
            len(self.val_dataset),
            metrics,
        )

    ####################################################################
    # OPTIONAL
    ####################################################################

    def get_history(self):

        return getattr(
            self,
            "_history",
            [],
        )
