import os
import cv2
import random
from pathlib import Path
from dataclasses import dataclass
import pyrallis


@dataclass
class Config:
    source_dir: str = "data/brodatz"
    output_dir: str = "data/brodatz_exp"
    block_size: int = 128
    patch_size: int = 64
    stride: int = 32
    random_seed: int = 42

    def __post_init__(self):
        random.seed(self.random_seed)


def save_patches(config, block, class_name, split, block_id):
    """Сохранение патчей одного блока."""

    out_dir = os.path.join(
        config.output_dir,
        split,
        class_name
    )
    os.makedirs(out_dir, exist_ok=True)
    h, w = block.shape[:2]
    idx = 0
    for y in range(0, h - config.patch_size + 1, config.stride):
        for x in range(0, w - config.patch_size + 1, config.stride):
            patch = block[
                y:y + config.patch_size,
                x:x + config.patch_size
            ]
            filename = os.path.join(
                out_dir,
                f"block{block_id:02d}_{idx:03d}.png"
            )
            cv2.imwrite(filename, patch)
            idx += 1


@pyrallis.wrap()
def main(config: Config):

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(config.output_dir, split), exist_ok=True)

    for image_path in sorted(Path(config.source_dir).glob("*")):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        h, w = image.shape
        if h != 512 or w != 512:
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

        class_name = image_path.stem
        blocks = []
        block_id = 0
        for by in range(0, 512, config.block_size):
            for bx in range(0, 512, config.block_size):
                block = image[
                    by:by + config.block_size,
                    bx:bx + config.block_size
                ]
                blocks.append((block_id, block))
                block_id += 1

        random.shuffle(blocks)

        train_blocks = blocks[:11]
        val_blocks = blocks[11:13]
        test_blocks = blocks[13:]

        for bid, block in train_blocks:
            save_patches(config, block, class_name, "train", bid)

        for bid, block in val_blocks:
            save_patches(config, block, class_name, "val", bid)

        for bid, block in test_blocks:
            save_patches(config, block, class_name, "test", bid)

        print(f"{class_name}: OK")

    print("Dataset successfully created.")


if __name__ == "__main__":
    main()
