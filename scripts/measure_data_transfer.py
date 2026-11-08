from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def mb(num_bytes: float) -> float:
    """Convert bytes to MiB (2^20 bytes)."""
    return num_bytes / (1024 * 1024)


def mb_dec(num_bytes: float) -> float:
    """Convert bytes to decimal MB (10^6 bytes)."""
    return num_bytes / 1_000_000


def _positions(size: int, patch_size: int, stride: int) -> List[int]:
    if size <= patch_size:
        return [0]
    positions = list(range(0, size - patch_size + 1, stride))
    if positions[-1] != size - patch_size:
        positions.append(size - patch_size)
    return positions


def _measure_single_image(args: Tuple[str, int, float]) -> Tuple[str, Dict[str, int]]:
    path_str, patch_size, overlap = args
    image_path = Path(path_str)
    file_bytes = image_path.stat().st_size
    stride = max(1, int(patch_size * (1 - overlap)))

    with Image.open(image_path) as img:
        img_gray = img.convert("L")
        w, h = img_gray.size
        arr = np.asarray(img_gray, dtype=np.uint8)

    xs = _positions(w, patch_size, stride)
    ys = _positions(h, patch_size, stride)
    coords = [(x, y) for y in ys for x in xs]

    png_c1_bytes = 0
    jpeg_98_bytes = 0
    raw_uint8_bytes = len(coords) * patch_size * patch_size
    raw_float32_bytes = raw_uint8_bytes * 4

    buf = io.BytesIO()
    for x, y in coords:
        patch_arr = arr[y : y + patch_size, x : x + patch_size]
        patch_img = Image.fromarray(patch_arr, mode="L")

        buf.seek(0)
        buf.truncate(0)
        patch_img.save(buf, format="PNG", compress_level=1)
        png_c1_bytes += buf.tell()

        buf.seek(0)
        buf.truncate(0)
        patch_img.save(buf, format="JPEG", quality=98, subsampling=0)
        jpeg_98_bytes += buf.tell()

    # Also measure source image in JPEG quality=98
    buf.seek(0)
    buf.truncate(0)
    Image.fromarray(arr, mode="L").save(buf, format="JPEG", quality=98, subsampling=0)
    img_jpeg_98_bytes = buf.tell()

    return path_str, {
        "width": w,
        "height": h,
        "num_patches": len(coords),
        "img_file_bytes": file_bytes,
        "img_jpeg_98_bytes": img_jpeg_98_bytes,
        "img_raw_uint8_bytes": w * h,
        "img_raw_float32_bytes": w * h * 4,
        "patch_png_c1_bytes": png_c1_bytes,
        "patch_jpeg_98_bytes": jpeg_98_bytes,
        "patch_raw_uint8_bytes": raw_uint8_bytes,
        "patch_raw_float32_bytes": raw_float32_bytes,
    }


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import yaml
    from src.datasets.partition import FederatedPartitioner
    from src.datasets.preprocessing import load_split

    with open("configs/texture.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["dataset"]["root"])
    num_clients = cfg["federated"]["num_clients"]
    seed = cfg["seed"]
    patch_size = cfg["dataset"]["patch_size"]
    overlap = cfg["dataset"]["overlap"]

    splits_samples: Dict[str, List[Tuple[Path, int]]] = {}
    all_paths: List[str] = []
    for split in ("train", "val", "test"):
        samples = load_split(root, split)
        splits_samples[split] = samples
        print(f"Split '{split}': {len(samples)} images", flush=True)
        for path, _ in samples:
            all_paths.append(str(path))

    worker_args = [(p, patch_size, overlap) for p in all_paths]
    image_cache: Dict[str, Dict[str, int]] = {}

    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 8)) as pool:
        for idx, (path_str, stats) in enumerate(pool.map(_measure_single_image, worker_args, chunksize=8), 1):
            image_cache[path_str] = stats
            if idx % 100 == 0 or idx == len(worker_args):
                print(f"Processed {idx}/{len(worker_args)} images...", flush=True)

    # Check raw dataset (data/dataset2_exp) and other dataset2 variants
    raw_datasets_info = {}
    for dname in ("dataset2_exp", "dataset2_exp_prepared", "dataset2_exp_prepared_flatfield", "dataset2_exp_prepared_flatfield_filesplit"):
        dpath = Path("data") / dname
        if dpath.exists():
            files = [f for f in dpath.rglob("*") if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            raw_datasets_info[dname] = {
                "num_files": len(files),
                "extensions": sorted(list({f.suffix.lower() for f in files})),
                "total_bytes": sum(f.stat().st_size for f in files),
                "total_mib": mb(sum(f.stat().st_size for f in files)),
                "total_mb_dec": mb_dec(sum(f.stat().st_size for f in files)),
            }

    def partition_split_fn(samples, s):
        return FederatedPartitioner(samples, num_clients, seed=s).stratified()

    train_partitions = partition_split_fn(splits_samples["train"], seed)
    val_partitions = partition_split_fn(splits_samples["val"], seed)
    test_partitions = partition_split_fn(splits_samples["test"], seed)

    def summarize_partitions(partitions: Dict[int, List[Tuple[Path, int]]]):
        per_client = []
        for cid in range(num_clients):
            samples = partitions[cid]
            keys = [str(p) for p, _ in samples]
            agg = {
                "client_id": cid,
                "num_images": len(keys),
                "num_patches": sum(image_cache[k]["num_patches"] for k in keys),
                "num_patches_epoch_04": sum(int(np.ceil(image_cache[k]["num_patches"] * 0.4)) for k in keys),
                "img_file_bytes": sum(image_cache[k]["img_file_bytes"] for k in keys),
                "img_jpeg_98_bytes": sum(image_cache[k]["img_jpeg_98_bytes"] for k in keys),
                "img_raw_uint8_bytes": sum(image_cache[k]["img_raw_uint8_bytes"] for k in keys),
                "img_raw_float32_bytes": sum(image_cache[k]["img_raw_float32_bytes"] for k in keys),
                "patch_png_c1_bytes": sum(image_cache[k]["patch_png_c1_bytes"] for k in keys),
                "patch_jpeg_98_bytes": sum(image_cache[k]["patch_jpeg_98_bytes"] for k in keys),
                "patch_raw_uint8_bytes": sum(image_cache[k]["patch_raw_uint8_bytes"] for k in keys),
                "patch_raw_float32_bytes": sum(image_cache[k]["patch_raw_float32_bytes"] for k in keys),
            }
            per_client.append(agg)
        return per_client

    train_client_stats = summarize_partitions(train_partitions)
    all_partitions = {
        cid: train_partitions[cid] + val_partitions[cid] + test_partitions[cid]
        for cid in range(num_clients)
    }
    all_client_stats = summarize_partitions(all_partitions)

    # Multi-seed stats across 5 seeds (42..46)
    seed_summaries = []
    for s in [42, 43, 44, 45, 46]:
        tp = partition_split_fn(splits_samples["train"], s)
        st = summarize_partitions(tp)
        seed_summaries.append({
            "seed": s,
            "client_img_mib_mean": float(np.mean([mb(c["img_file_bytes"]) for c in st])),
            "client_img_mib_std": float(np.std([mb(c["img_file_bytes"]) for c in st], ddof=1)),
            "client_patch_mib_mean": float(np.mean([mb(c["patch_png_c1_bytes"]) for c in st])),
            "client_patch_mib_std": float(np.std([mb(c["patch_png_c1_bytes"]) for c in st], ddof=1)),
        })

    # Now import torch and flwr only in main process after pool is closed
    import torch
    from flwr.common import ndarrays_to_parameters
    from main import build_model

    cfg_model = dict(cfg)
    cfg_model["model"] = dict(cfg["model"], pretrained=False)
    model = build_model(cfg_model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    state_dict = model.state_dict()
    ndarrays = [v.detach().cpu().numpy() for v in state_dict.values()]
    raw_model_bytes = sum(arr.nbytes for arr in ndarrays)
    flwr_params = ndarrays_to_parameters(ndarrays)
    flwr_serialized_bytes = sum(len(t) for t in flwr_params.tensors)

    buf_pt = io.BytesIO()
    torch.save(state_dict, buf_pt)
    torch_save_bytes = buf_pt.tell()

    rounds = cfg["federated"]["rounds"]

    results = {
        "dataset_summary": {
            "train_images": len(splits_samples["train"]),
            "val_images": len(splits_samples["val"]),
            "test_images": len(splits_samples["test"]),
            "total_images": sum(len(v) for v in splits_samples.values()),
            "sample_resolutions": list({(v["width"], v["height"]) for v in image_cache.values()}),
            "patches_per_image": list({v["num_patches"] for v in image_cache.values()}),
            "raw_datasets_info": raw_datasets_info,
        },
        "train_per_client": train_client_stats,
        "all_splits_per_client": all_client_stats,
        "multi_seed_train_summary": seed_summaries,
        "model_summary": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "num_state_dict_tensors": len(state_dict),
            "raw_ndarrays_bytes": raw_model_bytes,
            "raw_ndarrays_mib": mb(raw_model_bytes),
            "raw_ndarrays_mb_dec": mb_dec(raw_model_bytes),
            "flwr_serialized_bytes": flwr_serialized_bytes,
            "flwr_serialized_mib": mb(flwr_serialized_bytes),
            "flwr_serialized_mb_dec": mb_dec(flwr_serialized_bytes),
            "torch_pt_bytes": torch_save_bytes,
            "torch_pt_mib": mb(torch_save_bytes),
            "torch_pt_mb_dec": mb_dec(torch_save_bytes),
            "rounds": rounds,
            "num_clients": num_clients,
            "downlink_100r_10c_raw_mib": mb(raw_model_bytes * num_clients * rounds),
            "uplink_100r_10c_raw_mib": mb(raw_model_bytes * num_clients * rounds),
            "total_fl_100r_10c_raw_mib": mb(raw_model_bytes * num_clients * rounds * 2),
            "downlink_100r_10c_raw_mb_dec": mb_dec(raw_model_bytes * num_clients * rounds),
            "uplink_100r_10c_raw_mb_dec": mb_dec(raw_model_bytes * num_clients * rounds),
            "total_fl_100r_10c_raw_mb_dec": mb_dec(raw_model_bytes * num_clients * rounds * 2),
            "downlink_100r_10c_flwr_mib": mb(flwr_serialized_bytes * num_clients * rounds),
            "uplink_100r_10c_flwr_mib": mb(flwr_serialized_bytes * num_clients * rounds),
            "total_fl_100r_10c_flwr_mib": mb(flwr_serialized_bytes * num_clients * rounds * 2),
        },
    }

    out_path = Path("logs/data_transfer_measurements.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed measurements to {out_path}", flush=True)


if __name__ == "__main__":
    main()
