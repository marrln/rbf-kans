#!/usr/bin/env python3
import sys
import os
import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

THIS_DIR = os.path.dirname(__file__)          # e.g., /path/to/ship_performance
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

from rbfkan_utils.utils.dataset import group

# ============================================================
# CONFIGURATION – CHANGE THESE FOR YOUR TASK
# ============================================================
DATASET_NAME = 'ship_performance'            # single place to change dataset
DATASET_DIR = THIS_DIR

TARGET_COLUMN = "Ship_Type"                  # Column to predict (required)
TASK_TYPE = "classification"                 # "classification" or "regression" (clustering NOT supported)
TEST_SIZE = 0.2                              # Fraction for test set
RANDOM_SEED = 42

# Optional: limit which columns are used as features (None = use all except target)
FEATURE_COLUMNS = None

# Columns to drop entirely (e.g., date, id columns)
DROP_COLUMNS = ["Date"]

# ============================================================
# VALIDATION – Clustering is not allowed
# ============================================================
if TASK_TYPE == "clustering":
    raise ValueError(
        "Clustering mode is NOT supported. "
        "This script is designed for supervised learning (classification or regression). "
        "Please set TASK_TYPE to 'classification' or 'regression'."
    )

# ============================================================
# Internal paths – not exported
# ============================================================
_DATA_DIR = os.path.join(THIS_DIR, 'dataset')
_DATA_PATH = Path(_DATA_DIR)
_TRAIN_PKL = _DATA_PATH / f"{DATASET_NAME}_train.pkl"
_TEST_PKL  = _DATA_PATH / f"{DATASET_NAME}_test.pkl"
_LABELS_JSON = _DATA_PATH / "labels.json"
_STATS_JSON  = _DATA_PATH / "statistics.json"

# ------------------------------------------------------------
# 1. Dataset information
# ------------------------------------------------------------
def get_dataset_info() -> Dict:
    """Return metadata about the dataset based on current configuration."""
    info = {
        'task': TASK_TYPE,
        'target_column': TARGET_COLUMN,
        'test_size': TEST_SIZE,
        'random_seed': RANDOM_SEED
    }
    if TASK_TYPE == 'classification':
        info['num_classes'] = 'unknown'   # will be set after loading
    else:  # regression
        info['num_classes'] = None
    return info

# ------------------------------------------------------------
# 2. Build dataset: download, preprocess, split, save
# ------------------------------------------------------------
def build_dataset(force: bool = False) -> None:
    """Download Ship Performance dataset, preprocess, split into train/test."""
    if not force and _TRAIN_PKL.exists() and _TEST_PKL.exists():
        print(f"Dataset {DATASET_NAME} already exists. Use force=True to rebuild.")
        return

    _DATA_PATH.mkdir(parents=True, exist_ok=True)

    # ---- Load raw data (Kaggle or local) ----
    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "jeleeladekunlefijabi/ship-performance-clustering-dataset"
        )
        csv_path = Path(path) / "Ship_Performance_Dataset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Downloaded from Kaggle: {csv_path}")
    except Exception as e:
        # Fallback: look for local CSV in script directory
        local_csv = Path(THIS_DIR) / "Ship_Performance_Dataset.csv"
        if local_csv.exists():
            df = pd.read_csv(local_csv)
            print(f"Loaded local CSV from {local_csv}")
        else:
            raise RuntimeError(
                f"Could not download dataset and no local copy found.\n"
                f"Original error: {e}"
            )

    # ---- Basic cleaning ----
    df.columns = df.columns.str.strip()
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped column: {col}")

    # ---- Separate features and target ----
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    if FEATURE_COLUMNS is not None:
        missing = set(FEATURE_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        X = X[FEATURE_COLUMNS]

    # ---- Encode categorical features ----
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # ---- Handle target (encode classification, keep as float for regression) ----
    from sklearn.preprocessing import LabelEncoder as TargetEncoder
    if TASK_TYPE == "classification":
        le_target = TargetEncoder()
        y_encoded = le_target.fit_transform(y.astype(str))
        target_encoder = {
            "classes": le_target.classes_.tolist(),
            "mapping": dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
        }
    else:  # regression
        y_encoded = y.values.astype(np.float32)
        target_encoder = None

    # ---- Train/test split ----
    from sklearn.model_selection import train_test_split
    stratify = y_encoded if TASK_TYPE == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=stratify
    )

    # ---- Scale numerical features ----
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    if numerical_cols:
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Save scaler parameters (mean, std) for later use
    scaler_params = {
        "mean": scaler.mean_.tolist() if numerical_cols else [],
        "scale": scaler.scale_.tolist() if numerical_cols else [],
        "feature_names": numerical_cols
    }
    with open(_STATS_JSON, "w") as f:
        json.dump(scaler_params, f, indent=4)

    # ---- Save pickles ----
    train_data = {
        "data": X_train.to_numpy(dtype=np.float32),
        "labels": y_train,
        "feature_names": X.columns.tolist(),
        "target_name": TARGET_COLUMN,
        "task": TASK_TYPE,
        "categorical_info": label_encoders,
        "target_encoder": target_encoder
    }
    test_data = {
        "data": X_test.to_numpy(dtype=np.float32),
        "labels": y_test,
        "feature_names": X.columns.tolist(),
        "target_name": TARGET_COLUMN,
        "task": TASK_TYPE,
        "categorical_info": label_encoders,
        "target_encoder": target_encoder
    }

    with open(_TRAIN_PKL, "wb") as f:
        pickle.dump(train_data, f)
    with open(_TEST_PKL, "wb") as f:
        pickle.dump(test_data, f)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    if TASK_TYPE == "classification":
        print(f"Class distribution (train): {np.bincount(y_train)}")

# ------------------------------------------------------------
# 3. Labels (for classification tasks)
# ------------------------------------------------------------
def create_labels(force: bool = False, save: bool = True) -> Dict[str, Dict[str, int]]:
    """Create label mapping for classification target."""
    if TASK_TYPE != "classification":
        if save and (force or not _LABELS_JSON.exists()):
            dummy = {"info": "No classification labels"}
            with open(_LABELS_JSON, "w") as f:
                json.dump(dummy, f, indent=4)
        with open(_LABELS_JSON, "r") as f:
            return json.load(f)

    if not save:
        if _TRAIN_PKL.exists():
            with open(_TRAIN_PKL, "rb") as f:
                data = pickle.load(f)
            if data.get("target_encoder"):
                return data["target_encoder"]["mapping"]
        return {}

    if force or not _LABELS_JSON.exists():
        if not _TRAIN_PKL.exists():
            raise FileNotFoundError(f"Training data not found at {_TRAIN_PKL}. Run build_dataset() first.")
        with open(_TRAIN_PKL, "rb") as f:
            data = pickle.load(f)
        target_enc = data.get("target_encoder")
        label_dict = target_enc["mapping"] if target_enc else {}
        with open(_LABELS_JSON, "w") as f:
            json.dump(label_dict, f, indent=4)
        print(f"Labels saved to {_LABELS_JSON}")

    with open(_LABELS_JSON, "r") as f:
        return json.load(f)

# ------------------------------------------------------------
# 4. Statistics (mean/std of numerical features after scaling)
# ------------------------------------------------------------
def calculate_statistics(force: bool = False, save: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and std for numerical features (for compatibility)."""
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
        mean = np.array(stats.get("mean", []))
        std = np.array(stats.get("std", []))
    return mean, std

def _compute_stats_from_train() -> Tuple[np.ndarray, np.ndarray]:
    if not _TRAIN_PKL.exists():
        raise FileNotFoundError(f"Training data not found at {_TRAIN_PKL}. Run build_dataset() first.")
    with open(_TRAIN_PKL, "rb") as f:
        data = pickle.load(f)
    X = data["data"]
    return X.mean(axis=0), X.std(axis=0)

# ------------------------------------------------------------
# 5. Data accessors
# ------------------------------------------------------------
def get_dataset(subset: Literal["train_val", "test"]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (features, labels) for the requested subset."""
    pkl = _TRAIN_PKL if subset != "test" else _TEST_PKL
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    X = data["data"]
    y = data["labels"] if data["labels"] is not None else np.array([])
    return X, y

def get_full_dataframe(subset: Literal["train_val", "test"]) -> pd.DataFrame:
    """Return the preprocessed DataFrame (features + target as strings/numbers)."""
    pkl = _TRAIN_PKL if subset != "test" else _TEST_PKL
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    if data["labels"] is not None:
        if data["task"] == "classification" and data["target_encoder"]:
            inv_map = {v: k for k, v in data["target_encoder"]["mapping"].items()}
            y = pd.Series(data["labels"]).map(inv_map).values
        else:
            y = data["labels"]
        X[data["target_name"]] = y
    return X

# ------------------------------------------------------------
# 6. Grouping (uses rbfkan_utils.utils.dataset.group)
# ------------------------------------------------------------
def make_groups(df: pd.DataFrame):
    """Group by the target column."""
    # The 'group' function expects a DataFrame and a list of label column names
    return group(df, labels=[TARGET_COLUMN])

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
        df = get_full_dataframe("train_val")
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

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    print(f"\n{'='*60}\nPreparing dataset: {DATASET_NAME.upper()}")
    print(f"Target: {TARGET_COLUMN} | Task: {TASK_TYPE.upper()}\n{'='*60}\n")

    build_dataset(force=args.force)

    labels = create_labels()
    if labels:
        print(f"\nLabels created with {len(labels)} classes" if TASK_TYPE == "classification" else "\nNo labels (regression task)")

    mean, std = calculate_statistics()
    print(f"Statistics computed (mean shape: {mean.shape}, std shape: {std.shape})")

    train_data, train_labels = get_dataset("train_val")
    print(f"Train data: {train_data.shape}, labels: {train_labels.shape if len(train_labels) else 'None'}")
    test_data, test_labels = get_dataset("test")
    print(f"Test data: {test_data.shape}, labels: {test_labels.shape if len(test_labels) else 'None'}")

    groups = get_groups()
    print(f"\nGroups created: {len(groups)} groups")

    print(f"\n{'='*60}\n✓ Dataset {DATASET_NAME.upper()} prepared successfully!\n{'='*60}\n")