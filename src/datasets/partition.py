from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


Sample = Tuple[str, int]


class FederatedPartitioner:
    """
    Creates image-level partitions for federated learning.

    Each client receives its own set of source images.
    Patch extraction is performed locally on each client.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        num_clients: int,
        seed: int = 42,
    ):
        self.samples = list(samples)
        self.num_clients = num_clients
        self.random = random.Random(seed)
        self.rng = np.random.default_rng(seed)

    def iid(self) -> Dict[int, List[Sample]]:
        samples = self.samples.copy()

        self.random.shuffle(samples)

        partitions = {
            i: []
            for i in range(self.num_clients)
        }

        for idx, sample in enumerate(samples):
            partitions[idx % self.num_clients].append(sample)

        return partitions

    def stratified(self) -> Dict[int, List[Sample]]:
        grouped = defaultdict(list)

        for sample in self.samples:
            grouped[sample[1]].append(sample)

        partitions = {
            i: []
            for i in range(self.num_clients)
        }

        for class_samples in grouped.values():
            self.random.shuffle(class_samples)

            for idx, sample in enumerate(class_samples):
                partitions[idx % self.num_clients].append(sample)

        for client in partitions:
            self.random.shuffle(partitions[client])

        return partitions

    def dirichlet(
        self,
        alpha: float = 0.5,
        min_size: int = 10,
    ) -> Dict[int, List[Sample]]:
        labels = sorted(
            {
                label
                for _, label in self.samples
            }
        )

        grouped = defaultdict(list)

        for sample in self.samples:
            grouped[sample[1]].append(sample)

        while True:
            partitions = {
                i: []
                for i in range(self.num_clients)
            }

            for label in labels:
                class_samples = grouped[label].copy()

                self.random.shuffle(class_samples)

                proportions = self.rng.dirichlet(
                    np.repeat(
                        alpha,
                        self.num_clients,
                    )
                )

                split_points = (
                    np.cumsum(proportions)
                    * len(class_samples)
                ).astype(int)[:-1]

                chunks = np.split(
                    np.array(class_samples, dtype=object),
                    split_points,
                )

                for client, chunk in enumerate(chunks):
                    partitions[client].extend(
                        chunk.tolist()
                    )

            sizes = [
                len(v)
                for v in partitions.values()
            ]

            if min(sizes) >= min_size:
                break

        for client in partitions:
            self.random.shuffle(
                partitions[client]
            )

        return partitions

    @staticmethod
    def statistics(
        partitions: Dict[int, List[Sample]],
    ):
        stats = {}

        for client, samples in partitions.items():
            distribution = defaultdict(int)

            for _, label in samples:
                distribution[label] += 1

            stats[client] = {
                "num_images": len(samples),
                "class_distribution": dict(distribution),
            }

        return stats

    @staticmethod
    def print_statistics(
        partitions: Dict[int, List[Sample]],
    ):
        stats = FederatedPartitioner.statistics(
            partitions
        )

        print("-" * 70)

        for client, info in stats.items():
            print(
                f"Client {client}: "
                f"{info['num_images']} images"
            )

            for cls, count in sorted(
                info["class_distribution"].items()
            ):
                print(
                    f"  class {cls}: {count}"
                )

            print()

        print("-" * 70)
