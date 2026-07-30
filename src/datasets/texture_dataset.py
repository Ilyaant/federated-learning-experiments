from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


class TextureDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        grayscale: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.grayscale = grayscale

        self.split_dir = self.root / split

        if not self.split_dir.exists():
            raise FileNotFoundError(self.split_dir)

        self.classes = sorted(
            [
                directory.name
                for directory in self.split_dir.iterdir()
                if directory.is_dir()
            ]
        )

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }

        self.samples: List[Tuple[Path, int]] = []

        for cls in self.classes:
            class_dir = self.split_dir / cls

            files = [
                file
                for file in class_dir.rglob("*")
                if file.suffix.lower() in VALID_EXTENSIONS
            ]

            files.sort()

            for file in files:
                self.samples.append(
                    (
                        file,
                        self.class_to_idx[cls],
                    )
                )

        if transform is None:
            transform_list = []

            if grayscale:
                transform_list.append(
                    transforms.Grayscale(
                        num_output_channels=1
                    )
                )

            transform_list.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5],
                        std=[0.5],
                    ),
                ]
            )

            self.transform = transforms.Compose(
                transform_list
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        image = Image.open(path)

        if self.grayscale:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        image = self.transform(image)

        return image, label

    @property
    def targets(self):
        return [
            label
            for _, label in self.samples
        ]

    @property
    def image_paths(self):
        return [
            path
            for path, _ in self.samples
        ]

    @property
    def num_classes(self):
        return len(self.classes)

    def class_distribution(self):
        distribution = {
            cls: 0
            for cls in self.classes
        }

        for _, label in self.samples:
            distribution[
                self.classes[label]
            ] += 1

        return distribution

    def subset(
        self,
        indices: Sequence[int],
    ):
        dataset = TextureDataset.__new__(
            TextureDataset
        )

        dataset.root = self.root
        dataset.split = self.split
        dataset.split_dir = self.split_dir
        dataset.grayscale = self.grayscale
        dataset.transform = self.transform
        dataset.classes = self.classes
        dataset.class_to_idx = self.class_to_idx

        dataset.samples = [
            self.samples[idx]
            for idx in indices
        ]

        return dataset

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"split='{self.split}', "
            f"images={len(self.samples)}, "
            f"classes={len(self.classes)})"
        )
