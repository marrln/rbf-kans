#!/usr/bin/env python3   
import sys, os
import pickle
import numpy as np
import pandas as pd 
import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

THIS_DIR = os.path.dirname(__file__)          # e.g., /path/to/cifar10
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

from rbfkan_utils.utils.dataset import group

DATASET_NAME = 'cifar10'                     # single place to change dataset
DATASET_DIR = THIS_DIR

# Internal paths – not exported
_DATA_DIR = os.path.join(THIS_DIR, 'dataset')
_DATA_PATH = Path(_DATA_DIR)
_TRAIN_PKL = _DATA_PATH / f"{DATASET_NAME}_train.pkl"
_TEST_PKL  = _DATA_PATH / f"{DATASET_NAME}_test.pkl"
_LABELS_JSON = _DATA_PATH / "labels.json"
_STATS_JSON  = _DATA_PATH / "statistics.json"

def _get_class_names_from_torchvision(dataset_name: str) -> List[str]:
    try:
        from torchvision import datasets
    except ImportError:
        raise ImportError("torchvision is required for dataset preparation.")
    if dataset_name == 'cifar10':
        ds = datasets.CIFAR10(root=str(_DATA_PATH.parent), train=True, download=False)
        return ds.classes
    elif dataset_name == 'cifar100':
        ds = datasets.CIFAR100(root=str(_DATA_PATH.parent), train=True, download=False)
        return ds.classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_class_names() -> List[str]:
    if _TRAIN_PKL.exists():
        with open(_TRAIN_PKL, "rb") as f:
            data = pickle.load(f)
        if "classes" in data:
            return data["classes"]
    return _get_class_names_from_torchvision(DATASET_NAME)

def get_dataset_info() -> Dict:
    if DATASET_NAME == 'cifar10':
        return {
            'num_classes': 10,
            'input_shape': (32, 32, 3),
            'task': 'multiclass',
        }
    elif DATASET_NAME == 'cifar100':
        return {
            'num_classes': 100,
            'input_shape': (32, 32, 3),
            'task': 'multiclass',
        }
    else:
        raise ValueError(f"Unknown dataset: {DATASET_NAME}")

def build_dataset(force: bool = False) -> None:
    if not force and _TRAIN_PKL.exists() and _TEST_PKL.exists():
        print(f"Dataset {DATASET_NAME} already exists. Use force=True to rebuild.")
        return
    _DATA_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_NAME.upper()} dataset...")
    try:
        from torchvision import datasets
    except ImportError:
        raise ImportError("torchvision is required. Install with: pip install torchvision")
    if DATASET_NAME == 'cifar10':
        train_ds = datasets.CIFAR10(root=str(_DATA_PATH.parent), train=True, download=True)
        test_ds = datasets.CIFAR10(root=str(_DATA_PATH.parent), train=False, download=True)
    elif DATASET_NAME == 'cifar100':
        train_ds = datasets.CIFAR100(root=str(_DATA_PATH.parent), train=True, download=True)
        test_ds = datasets.CIFAR100(root=str(_DATA_PATH.parent), train=False, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {DATASET_NAME}")
    class_names = train_ds.classes
    train_data = train_ds.data
    train_labels = np.array(train_ds.targets)
    test_data = test_ds.data
    test_labels = np.array(test_ds.targets)
    with open(_TRAIN_PKL, "wb") as f:
        pickle.dump({"data": train_data, "labels": train_labels, "classes": class_names}, f)
    with open(_TEST_PKL, "wb") as f:
        pickle.dump({"data": test_data, "labels": test_labels, "classes": class_names}, f)
    print(f"Train: {train_data.shape}, Test: {test_data.shape}, Classes: {len(class_names)}")

def create_labels(force: bool = False, save: bool = True) -> Dict[str, Dict[str, int]]:
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

def calculate_statistics(force: bool = False, save: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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

def _compute_stats_from_train() -> Tuple[np.ndarray, np.ndarray]:
    if not _TRAIN_PKL.exists():
        raise FileNotFoundError(f"Training data not found at {_TRAIN_PKL}. Run build_dataset() first.")
    with open(_TRAIN_PKL, "rb") as f:
        data_dict = pickle.load(f)
    data = data_dict["data"].astype(np.float32) / 255.0
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    return mean, std

def get_dataset(subset: Literal["train_val", "test"]) -> Tuple[np.ndarray, np.ndarray]:
    if subset == "test":
        with open(_TEST_PKL, "rb") as f:
            data_dict = pickle.load(f)
    else:
        with open(_TRAIN_PKL, "rb") as f:
            data_dict = pickle.load(f)
    return data_dict["data"], data_dict["labels"]

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
        _, labels = get_dataset('train_val')
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
    print(f"\n{'='*60}\nPreparing dataset: {DATASET_NAME.upper()}\n{'='*60}\n")
    build_dataset(force=args.force)
    labels = create_labels()
    print(f"\nLabels created with {len(labels['class'])} classes")
    mean, std = calculate_statistics()
    print(f"Mean (RGB): {mean}, Std (RGB): {std}")
    train_data, train_labels = get_dataset("train_val")
    print(f"Train data: {train_data.shape}, labels: {train_labels.shape}")
    test_data, test_labels = get_dataset("test")
    print(f"Test data: {test_data.shape}, labels: {test_labels.shape}")
    groups = get_groups()
    print(f"\nGroups created: {len(groups)} groups")
    print(f"\n{'='*60}\n✓ Dataset {DATASET_NAME.upper()} prepared successfully!\n{'='*60}\n")