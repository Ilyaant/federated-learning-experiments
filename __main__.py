from dataclasses import dataclass
import pickle
import pyrallis
import torch

from src.run_experiment import run_experiment


@dataclass
class Config:
    shared_strategies = ["dirichlet", "pathological", "quantity_skew", "iid"]
    shared_ratio: float = 0.0
    num_clients: int = 15
    rounds: int = 100
    device: str = 'cpu'
    metrics_log_file: str = '../logs/metrics_0.txt'
    model: str = 'CIFARMediumCNN'  # or efficientnet_v2_s or resnet18
    num_classes: int = 100
    batch_size: int = 64
    epochs: int = 3

    def __post_init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.metrics_log_file = '../logs/metrics_ds.txt' if self.shared_ratio else '../logs/metrics_0.txt'


@pyrallis.wrap()
def main(config: Config):
    for share_strategy in config.shared_strategies:
        history_exp2 = run_experiment(
            share_strategy, 
            shared_ratio=config.shared_ratio,
            num_clients=config.num_clients,
            num_rounds=config.rounds,
            epochs=config.epochs,
            device=config.device,
            model_name=config.model,
            num_classes=config.num_classes,
            batch_size=config.batch_size,
            metrics_log_file=config.metrics_log_file
        )

        print("\n--- ИТОГОВЫЕ РЕЗУЛЬТАТЫ (ПОСЛЕДНИЙ РАУНД) ---")
        print("Accuracy:", history_exp2.metrics_distributed['accuracy'][-1][1])
        print("F1-score:", history_exp2.metrics_distributed['f1'][-1][1])

        with open(f'../metrics/history_{share_strategy}_{config.shared_ratio}.pkl', 'wb') as f:
            pickle.dump(history_exp2, f)


if __name__ == "__main__":
    main()
