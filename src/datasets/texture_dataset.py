from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


CLASS_TO_IDX = {
    "clear": 0,
    "G": 1,
    "GP": 2,
    "M": 3,
    "T": 4,
}


class TextureDataset:
    """
    Reads texture images and performs train/val/test split.

    Dataset structure:

    root/
        clear/
        G/
        GP/
        M/
        T/
    """

    IMG_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    def __init__(
        self,
        root: str | Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        split_file: Optional[str | Path] = None,
    ):

        self.root = Path(root)
        self.seed = seed

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.")

        self.samples = self._scan()

        if split_file is not None and Path(split_file).exists():
            self._load_split(split_file)
        else:
            self._split()

            if split_file is not None:
                self.save_split(split_file)

    def _scan(self):

        samples = []

        for cls_name, label in CLASS_TO_IDX.items():

            cls_dir = self.root / cls_name

            if not cls_dir.exists():
                raise FileNotFoundError(cls_dir)

            for img in sorted(cls_dir.rglob("*")):

                if img.suffix.lower() not in self.IMG_EXTENSIONS:
                    continue

                samples.append(
                    {
                        "path": img,
                        "label": label,
                        "class_name": cls_name,
                    }
                )

        return samples

    def _split(self):

        labels = np.array([x["label"] for x in self.samples])

        idx = np.arange(len(self.samples))

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self.train_ratio,
            random_state=self.seed,
        )

        train_idx, tmp_idx = next(splitter.split(idx, labels))

        tmp_labels = labels[tmp_idx]

        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=val_size,
            random_state=self.seed,
        )

        val_local, test_local = next(
            splitter.split(tmp_idx, tmp_labels)
        )

        self.train_samples = [
            self.samples[i]
            for i in train_idx
        ]

        self.val_samples = [
            self.samples[tmp_idx[i]]
            for i in val_local
        ]

        self.test_samples = [
            self.samples[tmp_idx[i]]
            for i in test_local
        ]

    def save_split(self, filename):

        filename = Path(filename)

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        split = {
            "train": [
                str(x["path"])
                for x in self.train_samples
            ],
            "val": [
                str(x["path"])
                for x in self.val_samples
            ],
            "test": [
                str(x["path"])
                for x in self.test_samples
            ],
        }

        with open(filename, "w") as f:
            json.dump(split, f, indent=4)

    def _load_split(self, filename):

        with open(filename) as f:
            split = json.load(f)

        mapping = {
            str(x["path"]): x
            for x in self.samples
        }

        self.train_samples = [
            mapping[p]
            for p in split["train"]
        ]

        self.val_samples = [
            mapping[p]
            for p in split["val"]
        ]

        self.test_samples = [
            mapping[p]
            for p in split["test"]
        ]

    def class_distribution(self, samples):

        result = {}

        for cls in CLASS_TO_IDX:

            result[cls] = 0

        for sample in samples:

            result[sample["class_name"]] += 1

        return result

    def summary(self):

        print()

        print("========== DATASET ==========")

        print("Total:", len(self.samples))
        print("Train:", len(self.train_samples))
        print("Validation:", len(self.val_samples))
        print("Test:", len(self.test_samples))

        print()

        print("Train distribution")

        print(
            self.class_distribution(
                self.train_samples
            )
        )

        print()

        print("Validation distribution")

        print(
            self.class_distribution(
                self.val_samples
            )
        )

        print()

        print("Test distribution")

        print(
            self.class_distribution(
                self.test_samples
            )
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        return self.samples[index]