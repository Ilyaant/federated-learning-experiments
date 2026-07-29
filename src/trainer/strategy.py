from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from trainer.metrics import merge_metrics


class TextureFedAvg(FedAvg):
    """
    Extended FedAvg strategy.

    Responsibilities
    ----------------
    * FedAvg aggregation
    * Aggregation of client train metrics
    * Centralized validation
    * Best model checkpoint
    * Last checkpoint
    * Experiment history
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        trainer,
        val_loader,
        test_loader,
        experiment,
        device,
        initial_parameters: Optional[Parameters] = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
    ):

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )

        self.model = model
        self.trainer = trainer

        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = device

        self.experiment = experiment

        self.best_f1 = -1.0
        self.best_round = -1

        self.history: List[dict] = []

    ####################################################################
    # Utilities
    ####################################################################

    def parameters_to_model(
        self,
        parameters: Parameters,
    ):

        arrays = parameters_to_ndarrays(parameters)

        state_dict = OrderedDict()

        for key, value in zip(
            self.model.state_dict().keys(),
            arrays,
        ):

            state_dict[key] = torch.tensor(value)

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

    def model_to_parameters(self):

        arrays = [
            tensor.detach().cpu().numpy()
            for tensor in self.model.state_dict().values()
        ]

        return ndarrays_to_parameters(arrays)

    ####################################################################
    # Fit
    ####################################################################

    def aggregate_fit(
        self,
        server_round: int,
        results: List[
            Tuple[
                ClientProxy,
                FitRes,
            ]
        ],
        failures,
    ):

        aggregated = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated is None:

            return None

        parameters_aggregated, _ = aggregated

        self.parameters_to_model(
            parameters_aggregated
        )

        train_metrics = []

        for _, fit_res in results:

            train_metrics.append(
                (
                    fit_res.metrics,
                    fit_res.num_examples,
                )
            )

        train = merge_metrics(train_metrics)

        history = {
            "round": server_round,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "train_precision": train["precision"],
            "train_recall": train["recall"],
            "train_f1": train["f1"],
        }

        self.history.append(history)

        self.experiment.save_last_model(
            self.model
        )

        return (
            parameters_aggregated,
            {},
        )
        ####################################################################
    # Centralized validation
    ####################################################################

    def evaluate(
        self,
        server_round: int,
        parameters,
    ):

        self.parameters_to_model(parameters)

        metrics = self.trainer.validate(
            self.model,
            self.val_loader
        )

        self.history[-1].update(
            {
                "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
                "patch_accuracy": metrics["patch_accuracy"],
                "patch_precision": metrics["patch_precision"],
                "patch_recall": metrics["patch_recall"],
                "patch_f1": metrics["patch_f1"],
            }
        )

        self.experiment.append_history(
            self.history[-1]
        )

        self.experiment.save_history(
            self.history
        )

        self.experiment.save_last_model(
            self.model
        )

        if metrics["f1"] > self.best_f1:

            self.best_f1 = metrics["f1"]
            self.best_round = server_round

            self.experiment.save_best_model(
                self.model
            )

            self.experiment.save_confusion_matrix(
                metrics["confusion_matrix"],
                filename="best_confusion_matrix.png",
            )

        return (
            float(metrics["loss"]),
            {
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            },
        )

    ####################################################################
    # Flower evaluate aggregation
    ####################################################################

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):

        return super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

    ####################################################################
    # Finish experiment
    ####################################################################

    def finish(self):

        self.experiment.load_best_model(
            self.model
        )

        metrics = self.trainer.test(
            self.model,
            self.test_loader
        )

        self.experiment.save_test_metrics(
            metrics
        )

        self.experiment.save_confusion_matrix(
            metrics["confusion_matrix"],
            filename="test_confusion_matrix.png",
        )

        self.experiment.finish(
            best_round=self.best_round,
            best_f1=self.best_f1,
        )

    ####################################################################
    # Convenience
    ####################################################################

    @property
    def best_metrics(self):

        return {
            "round": self.best_round,
            "f1": self.best_f1,
        }

    @property
    def current_round(self):

        if len(self.history) == 0:
            return 0

        return self.history[-1]["round"]

    @property
    def last_metrics(self):

        if len(self.history) == 0:
            return None

        return self.history[-1]