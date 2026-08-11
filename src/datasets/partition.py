from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

Sample = Tuple[str, int]


def _empty_partitions(num_clients: int) -> Dict[int, List[Sample]]:
    return {i: [] for i in range(num_clients)}


def _round_robin(
    samples: Sequence[Sample],
    num_clients: int,
    rng: random.Random,
    offset: int = 0,
) -> Dict[int, List[Sample]]:
    shuffled = list(samples)
    rng.shuffle(shuffled)

    partitions = _empty_partitions(num_clients)
    for idx, sample in enumerate(shuffled):
        partitions[(idx + offset) % num_clients].append(sample)

    return partitions


def _group_by_label(samples: Sequence[Sample]) -> Dict[int, List[Sample]]:
    grouped: Dict[int, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)
    return grouped


class FederatedPartitioner:
    """Assigns source images to federated clients; patches are extracted locally."""

    def __init__(
        self,
        samples: Sequence[Sample],
        num_clients: int,
        seed: int = 42,
    ):
        self.samples = list(samples)
        self.num_clients = num_clients
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def iid(self) -> Dict[int, List[Sample]]:
        return _round_robin(self.samples, self.num_clients, self.rng)

    def stratified(self) -> Dict[int, List[Sample]]:
        partitions = _empty_partitions(self.num_clients)

        # Rotate the starting client per class so that leftover
        # samples do not all pile up on client 0.
        for class_idx, class_samples in enumerate(
            _group_by_label(self.samples).values()
        ):
            for client, chunk in _round_robin(
                class_samples,
                self.num_clients,
                self.rng,
                offset=class_idx,
            ).items():
                partitions[client].extend(chunk)

        return partitions

    def dirichlet(
        self,
        alpha: float = 0.5,
        min_size: int = 10,
        max_attempts: int = 1000,
    ) -> Dict[int, List[Sample]]:
        grouped = _group_by_label(self.samples)
        labels = sorted(grouped)

        for _ in range(max_attempts):
            partitions = _empty_partitions(self.num_clients)

            for label in labels:
                class_samples = grouped[label].copy()
                self.rng.shuffle(class_samples)

                proportions = self.np_rng.dirichlet(
                    np.full(self.num_clients, alpha)
                )
                split_points = (
                    np.cumsum(proportions) * len(class_samples)
                ).astype(int)[:-1]

                for client, chunk in enumerate(
                    np.split(
                        np.array(class_samples, dtype=object),
                        split_points,
                    )
                ):
                    partitions[client].extend(chunk.tolist())

            if min(len(p) for p in partitions.values()) >= min_size:
                for client in partitions:
                    self.rng.shuffle(partitions[client])
                return partitions

        raise ValueError(
            f"Could not satisfy min_size={min_size} "
            f"for {self.num_clients} clients after {max_attempts} attempts"
        )

    @staticmethod
    def statistics(partitions: Dict[int, List[Sample]]) -> Dict[int, dict]:
        stats = {}

        for client, samples in partitions.items():
            distribution: Dict[int, int] = defaultdict(int)
            for _, label in samples:
                distribution[label] += 1

            stats[client] = {
                "num_images": len(samples),
                "class_distribution": dict(distribution),
            }

        return stats

    @staticmethod
    def print_statistics(partitions: Dict[int, List[Sample]]) -> None:
        stats = FederatedPartitioner.statistics(partitions)
        print("-" * 70)

        for client, info in stats.items():
            print(f"Client {client}: {info['num_images']} images")
            for cls, count in sorted(info["class_distribution"].items()):
                print(f"  class {cls}: {count}")
            print()

        print("-" * 70)
