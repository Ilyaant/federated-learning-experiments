"""Texture classification with FastViT-T8 on 224x224 patches.

Usage:
    python main.py --config configs/texture.yaml
    python main.py --override train.epochs=5 --override patches.train_overlap=0.5
    python main.py --run-name smoke --override dataset.max_images_per_class=2 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import apply_overrides, load_config, set_path  # noqa: E402
from src.experiment import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/texture.yaml", help="YAML config path")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value by dotted path, e.g. train.lr=3e-5 (repeatable)",
    )
    parser.add_argument("--run-name", help="Shortcut for logging.run_name")
    parser.add_argument("--save-dir", help="Shortcut for logging.save_dir")
    parser.add_argument("--epochs", type=int, help="Shortcut for train.epochs")
    parser.add_argument("--dry-run", action="store_true", help="Build data and model, do not train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    if args.run_name:
        set_path(config, "logging.run_name", args.run_name)
    if args.save_dir:
        set_path(config, "logging.save_dir", args.save_dir)
    if args.epochs is not None:
        set_path(config, "train.epochs", args.epochs)

    run_experiment(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
