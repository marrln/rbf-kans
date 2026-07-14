#!/usr/bin/env python3
import sys
import os
import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

THIS_DIR = os.path.dirname(__file__)          # e.g., /path/to/cifar100
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

from rbfkan_utils.utils.dataset import group

DATASET_NAME = "cifar100"
DATASET_DIR = THIS_DIR

_DATA_DIR = os.path.join(THIS_DIR, 'dataset')
_DATA_PATH = Path(_DATA_DIR)
_TRAIN_PKL = _DATA_PATH / "cifar100_train.pkl"
_TEST_PKL  = _DATA_PATH / "cifar100_test.pkl"
_LABELS_JSON = _DATA_PATH / "labels.json"
_STATS_JSON  = _DATA_PATH / "statistics.json"
_MAPPING_JSON = _DATA_PATH / "fine_to_coarse.json"

# Torchvision root (parent of our dataset directory)
TORCH_ROOT = _DATA_PATH.parent

def _get_class_names_from_torchvision() -> List[str]:
    """Fetch class names from torchvision's CIFAR100."""
    try:
        from torchvision import datasets
    except ImportError:
        raise ImportError("torchvision is required for dataset preparation.")
    ds = datasets.CIFAR100(root=str(TORCH_ROOT), train=True, download=False)
    return ds.classes

def get_class_names() -> List[str]:
    """Return the 100 fine‑grained class names."""
    if _TRAIN_PKL.exists():
        with open(_TRAIN_PKL, "rb") as f:
            data = pickle.load(f)
        if "classes" in data:
            return data["classes"]
    return _get_class_names_from_torchvision()

def _get_fine_to_coarse_mapping() -> List[int]:
    """
    Return a list of length 100 mapping fine class index -> coarse class index.
    If a saved mapping exists, load it; otherwise build from training data and save.
    """
    if _MAPPING_JSON.exists():
        with open(_MAPPING_JSON, "r") as f:
            return json.load(f)

    # Build mapping from training pickle (must exist)
    if not _TRAIN_PKL.exists():
        raise FileNotFoundError(f"Training data not found. Run build_dataset() first.")
    with open(_TRAIN_PKL, "rb") as f:
        data = pickle.load(f)

    coarse_labels = data.get("coarse_labels")
    if coarse_labels is None:
        raise ValueError("Training pickle does not contain coarse_labels. Rebuild dataset with force=True.")

    fine_to_coarse = [0] * 100
    for fine_cls in range(100):
        idx = np.where(np.array(data["labels"]) == fine_cls)[0][0]
        fine_to_coarse[fine_cls] = int(coarse_labels[idx])

    _MAPPING_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_MAPPING_JSON, "w") as f:
        json.dump(fine_to_coarse, f, indent=2)

    return fine_to_coarse

def get_dataset_info() -> Dict:
    """Return dataset metadata, including the fine‑to‑coarse mapping."""
    info = {
        'num_classes': 100,
        'input_shape': (32, 32, 3),
        'task': 'multiclass',
    }
    try:
        info['fine_to_coarse'] = _get_fine_to_coarse_mapping()
    except Exception:
        pass
    return info

def _load_cifar100_pickle(filepath: Path) -> Dict:
    """
    Load a CIFAR‑100 pickle file (train or test) and return a dict with
    'data', 'fine_labels', 'coarse_labels', and 'batch_label'.
    """
    with open(filepath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    # Decode bytes keys to strings for easier handling
    decoded = {}
    for key, value in d.items():
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        if key == 'data':
            decoded['data'] = value
        elif key == 'fine_labels':
            decoded['labels'] = np.array(value)          # rename to 'labels'
            decoded['fine_labels'] = np.array(value)
        elif key == 'coarse_labels':
            decoded['coarse_labels'] = np.array(value)
        elif key == 'batch_label':
            decoded['batch_label'] = value.decode('utf-8') if isinstance(value, bytes) else value
        else:
            decoded[key] = value
    return decoded

def build_dataset(force: bool = False) -> None:
    """Download CIFAR‑100 and save as pickles with both fine and coarse labels."""
    if not force and _TRAIN_PKL.exists() and _TEST_PKL.exists():
        print("CIFAR‑100 dataset already exists. Use force=True to rebuild.")
        return

    _DATA_PATH.mkdir(parents=True, exist_ok=True)
    print("Downloading CIFAR‑100 dataset...")
    try:
        from torchvision import datasets
    except ImportError:
        raise ImportError("torchvision is required. Install with: pip install torchvision")

    # Download the dataset (this ensures the raw pickles are in TORCH_ROOT/cifar-100-python/)
    datasets.CIFAR100(root=str(TORCH_ROOT), train=True, download=True)
    datasets.CIFAR100(root=str(TORCH_ROOT), train=False, download=True)

    # Paths to the raw pickles (torchvision uses the same layout)
    cifar100_dir = TORCH_ROOT / 'cifar-100-python'
    train_pickle = cifar100_dir / 'train'
    test_pickle  = cifar100_dir / 'test'

    if not train_pickle.exists() or not test_pickle.exists():
        raise FileNotFoundError("Downloaded CIFAR‑100 pickles not found. Check torchvision download.")

    # Load the raw pickles manually
    train_dict = _load_cifar100_pickle(train_pickle)
    test_dict  = _load_cifar100_pickle(test_pickle)

    # Also get class names from the meta file (or from torchvision)
    meta_file = cifar100_dir / 'meta'
    if meta_file.exists():
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
        if b'fine_label_names' in meta:
            class_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]
        else:
            class_names = _get_class_names_from_torchvision()
    else:
        class_names = _get_class_names_from_torchvision()

    # Prepare final dictionaries with consistent keys
    train_out = {
        "data": train_dict["data"],
        "labels": train_dict["labels"],          # fine labels
        "coarse_labels": train_dict["coarse_labels"],
        "classes": class_names,
    }
    test_out = {
        "data": test_dict["data"],
        "labels": test_dict["labels"],
        "coarse_labels": test_dict["coarse_labels"],
        "classes": class_names,
    }

    with open(_TRAIN_PKL, "wb") as f:
        pickle.dump(train_out, f)
    with open(_TEST_PKL, "wb") as f:
        pickle.dump(test_out, f)

    print(f"Train: {train_out['data'].shape}, Test: {test_out['data'].shape}, Classes: {len(class_names)}")

    # Generate the fine‑to‑coarse mapping immediately
    _get_fine_to_coarse_mapping()

def create_labels(force: bool = False, save: bool = True) -> Dict[str, Dict[str, int]]:
    """Create or load the class label mapping."""
    if not save:
        class_names = get_class_names()
        return {"class": {name: idx for idx, name in enumerate(class_names)}}

    if force or not _LABELS_JSON.exists():
        class_names = get_class_names()
        label_dict = {"class": {name: idx for idx, name in enumerate(class_names)}}
        _LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(_LABELS_JSON, "w") as f:
            json.dump(label_dict, f, indent=4)
        print(f"Labels saved to {_LABELS_JSON}")

    with open(_LABELS_JSON, "r") as f:
        return json.load(f)

def _compute_stats_from_train() -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std from the training set."""
    if not _TRAIN_PKL.exists():
        raise FileNotFoundError(f"Training data not found at {_TRAIN_PKL}. Run build_dataset() first.")
    with open(_TRAIN_PKL, "rb") as f:
        data_dict = pickle.load(f)
    data = data_dict["data"].astype(np.float32) / 255.0
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    return mean, std

def calculate_statistics(force: bool = False, save: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute or load dataset mean and std."""
    if not save:
        return _compute_stats_from_train()

    if force or not _STATS_JSON.exists():
        mean, std = _compute_stats_from_train()
        stats = {"mean": mean.tolist(), "std": std.tolist()}
        with open(_STATS_JSON, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Statistics saved to {_STATS_JSON}")
    else:
        with open(_STATS_JSON, "r") as f:
            stats = json.load(f)
        mean, std = np.array(stats["mean"]), np.array(stats["std"])
    return mean, std

def get_dataset(subset: Literal["train_val", "test"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset and return (images, fine_labels, coarse_labels).
    """
    if subset == "train_val":
        pkl_path = _TRAIN_PKL
    elif subset == "test":
        pkl_path = _TEST_PKL
    else:
        raise ValueError(f"subset must be 'train_val' or 'test', got {subset}")

    with open(pkl_path, "rb") as f:
        data_dict = pickle.load(f)

    coarse = data_dict.get("coarse_labels")
    if coarse is None:
        raise KeyError("coarse_labels missing from pickle. Rebuild dataset with force=True.")

    return data_dict["data"], data_dict["labels"], coarse

def make_groups(df):
    return group(df, labels=['Label'])

def __save_groups(groups):
    groups_path = _DATA_PATH / 'groups.json'
    with open(groups_path, 'w') as fw:
        json.dump(groups, fw, indent=2)

def __nested_pop(groups, keys):
    if len(keys) > 1 and keys[0] in groups.keys():
        __nested_pop(groups[keys[0]], keys[1:])
        if len(groups[keys[0]]) == 0:
            groups.pop(keys[0])
    elif len(keys) == 1 and keys[0] in groups.keys():
        groups.pop(keys[0])

def exclude_groups(groups, exclude: list[list[str]] = None):
    if exclude is not None:
        for keys in exclude:
            __nested_pop(groups, keys)
    return groups

def get_groups(regenerate=False, exclude=None):
    groups_path = _DATA_PATH / 'groups.json'
    if groups_path.exists() and not regenerate:
        with open(groups_path, 'r') as fr:
            groups = json.load(fr)
    else:
        _, labels, _ = get_dataset('train_val')   # ignore images and coarse labels
        df = pd.DataFrame(data=labels, columns=['Label'])
        groups = make_groups(df)
        __save_groups(groups)
    return exclude_groups(groups, exclude)

def get_dataset_paths(subsets: Union[Literal["train", "val", "test"], List[Literal["train", "val", "test"]]]):
    if isinstance(subsets, str):
        subsets = [subsets]
    result = {}
    for s in subsets:
        if s == "test":
            result[s] = _TEST_PKL
        else:
            result[s] = _TRAIN_PKL
    if len(result) == 1:
        return list(result.values())[0]
    elif all(isinstance(v, Path) for v in result.values()):
        return list(result.values())
    else:
        return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    print(f"\n{'='*60}\nPreparing dataset: CIFAR-100\n{'='*60}\n")

    build_dataset(force=args.force)

    labels = create_labels()
    print(f"\nLabels created with {len(labels['class'])} classes")

    mean, std = calculate_statistics()
    print(f"Mean (RGB): {mean}, Std (RGB): {std}")

    train_data, train_labels, train_coarse = get_dataset("train_val")
    print(f"Train data: {train_data.shape}, labels: {train_labels.shape}, coarse: {train_coarse.shape}")

    test_data, test_labels, test_coarse = get_dataset("test")
    print(f"Test data:  {test_data.shape}, labels: {test_labels.shape}, coarse: {test_coarse.shape}")

    groups = get_groups()
    print(f"\nGroups created: {len(groups)} groups")

    print(f"\n{'='*60}\n✓ CIFAR‑100 dataset prepared successfully!\n{'='*60}\n")