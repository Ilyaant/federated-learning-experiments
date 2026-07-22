import os
import cv2
import random
from pathlib import Path


SOURCE_DIR = "data/brodatz"
OUTPUT_DIR = "data/brodatz_exp"

# Размер блока
BLOCK_SIZE = 128

# Размер патча
PATCH_SIZE = 64

# Шаг окна
STRIDE = 32

# Для воспроизводимости
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


for split in ("train", "val", "test"):
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)


def save_patches(block, class_name, split, block_id):
    """Сохранение патчей одного блока."""

    out_dir = os.path.join(
        OUTPUT_DIR,
        split,
        class_name
    )

    os.makedirs(out_dir, exist_ok=True)

    h, w = block.shape[:2]

    idx = 0

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):

            patch = block[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            filename = os.path.join(
                out_dir,
                f"block{block_id:02d}_{idx:03d}.png"
            )

            cv2.imwrite(filename, patch)

            idx += 1

for image_path in sorted(Path(SOURCE_DIR).glob("*")):

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    h, w = image.shape

    # assert h == 512 and w == 512, \
    #     f"{image_path.name}: ожидалось 512×512, получено {w}×{h}"
    if h != 512 or w != 512:
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    class_name = image_path.stem

    blocks = []

    block_id = 0

    for by in range(0, 512, BLOCK_SIZE):
        for bx in range(0, 512, BLOCK_SIZE):

            block = image[
                by:by + BLOCK_SIZE,
                bx:bx + BLOCK_SIZE
            ]

            blocks.append((block_id, block))

            block_id += 1

    random.shuffle(blocks)

    train_blocks = blocks[:11]
    val_blocks = blocks[11:13]
    test_blocks = blocks[13:]

    for bid, block in train_blocks:
        save_patches(block, class_name, "train", bid)

    for bid, block in val_blocks:
        save_patches(block, class_name, "val", bid)

    for bid, block in test_blocks:
        save_patches(block, class_name, "test", bid)

    print(f"{class_name}: OK")

print("Dataset successfully created.")