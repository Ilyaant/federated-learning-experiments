from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


class DatasetPartitioner:
    """
    Partition training dataset between federated clients.
    Supports:
        - IID
        - Dirichlet Non-IID
    """

    def __init__(
        self,
        samples: List[dict],
        num_clients: int,
        seed: int = 42,
    ):
        self.samples = samples
        self.num_clients = num_clients
        self.seed = seed

        self.rng = np.random.default_rng(seed)

    ####################################################################
    # IID
    ####################################################################

    def iid(self) -> Dict[int, List[dict]]:

        indices = np.arange(len(self.samples))
        self.rng.shuffle(indices)

        splits = np.array_split(indices, self.num_clients)

        clients = {}

        for cid, split in enumerate(splits):

            clients[cid] = [
                self.samples[i]
                for i in split
            ]

        return clients

    ####################################################################
    # DIRICHLET
    ####################################################################

    def dirichlet(
        self,
        alpha: float = 0.5,
        min_samples: int = 1,
    ) -> Dict[int, List[dict]]:

        labels = np.array(
            [s["label"] for s in self.samples]
        )

        num_classes = len(np.unique(labels))

        while True:

            client_indices = [
                []
                for _ in range(self.num_clients)
            ]

            for cls in range(num_classes):

                cls_idx = np.where(labels == cls)[0]

                self.rng.shuffle(cls_idx)

                proportions = self.rng.dirichlet(
                    np.repeat(alpha, self.num_clients)
                )

                split_points = (
                    np.cumsum(proportions)[:-1]
                    * len(cls_idx)
                ).astype(int)

                split = np.split(
                    cls_idx,
                    split_points,
                )

                for cid in range(self.num_clients):

                    client_indices[cid].extend(
                        split[cid].tolist()
                    )

            sizes = [
                len(x)
                for x in client_indices
            ]

            if min(sizes) >= min_samples:
                break

        clients = {}

        for cid in range(self.num_clients):

            self.rng.shuffle(client_indices[cid])

            clients[cid] = [
                self.samples[i]
                for i in client_indices[cid]
            ]

        return clients

    ####################################################################
    # SAVE / LOAD
    ####################################################################

    @staticmethod
    def save(
        clients: Dict[int, List[dict]],
        filename: str | Path,
    ):

        filename = Path(filename)

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        obj = {}

        for cid, samples in clients.items():

            obj[str(cid)] = [
                str(s["path"])
                for s in samples
            ]

        with open(filename, "w") as f:
            json.dump(obj, f, indent=4)

    @staticmethod
    def load(
        filename: str | Path,
        all_samples: List[dict],
    ) -> Dict[int, List[dict]]:

        with open(filename) as f:
            obj = json.load(f)

        mapping = {
            str(s["path"]): s
            for s in all_samples
        }

        clients = {}

        for cid, paths in obj.items():

            clients[int(cid)] = [
                mapping[p]
                for p in paths
            ]

        return clients

    ####################################################################
    # INFO
    ####################################################################

    @staticmethod
    def distribution(
        samples: List[dict],
    ):

        d = defaultdict(int)

        for sample in samples:

            d[sample["class_name"]] += 1

        return dict(d)

    @staticmethod
    def summary(
        clients: Dict[int, List[dict]]
    ):

        print()

        print("=" * 70)
        print("CLIENT PARTITIONS")
        print("=" * 70)

        for cid in sorted(clients):

            print()

            print(
                f"Client {cid:02d} "
                f"({len(clients[cid])} images)"
            )

            print(
                DatasetPartitioner.distribution(
                    clients[cid]
                )
            )