# RBF-KAN: Radial Basis Function Kolmogorov-Arnold Networks for Image Classification

This project implements **RBF-KAN** – a flexible neural network architecture that replaces traditional linear weights with learnable RBF kernels. It supports multiclass and binary classification tasks, with built‑in training, testing, hyperparameter sweeps, and comprehensive result comparison tools. There are four different datasets available for benchmarking: CIFAR‑10, CIFAR‑100, MNIST, and the ship performance dataset. You can apply the RBF‑KAN architecture to your own datasets by making a dedicated folder and implementing the dataset‑specific `prepare_dataset.py` file.

## Features

- **Modular RBF‑KAN layers** – configurable grid sizes, RBF modes, residual connections, normalisation, dropout.
- **Dataset support** – Easily extendable to all datasets.
- **Full pipeline** – single‑script training, testing, and metric extraction.
- **Hyperparameter sweep** – parallel execution of configurations with automatic hash‑based directory management.
- **Result comparison** – generates bar charts, top‑5 tables, confusion matrices, and PR curves.
- **Reproducibility** – every experiment is stored in a hash‑named folder containing model configs, checkpoints, logs, and a `hyperparameters.csv` file.

## Project Structure (simplified)

```
.
├── cifar10/                # dataset‑specific folder (similar for cifar100, mnist, etc.)
│   ├── dataset/            # pre‑processed pickles & statistics
│   ├── train/              # experiment outputs (hash‑named subfolders)
│   │   ├── <hash>/
│   │   │   └── test_0/     # test version subdirectory
│   │   │       ├── config/         # train.json, model.json, hyperparameters.csv
│   │   │       ├── models/         # best.pt, last.pt, summary.txt
│   │   │       ├── plot/           # training curves, confusion matrices
│   │   │       ├── rslt/           # predictions, ground truth
│   │   │       └── history.json    # loss & metrics per epoch
│   │   └── comparison/      # global comparison plots
│   └── prepare_dataset.py   # dataset download, splits, labels, statistics
├── rbfkan_utils/           # core library (models, training, metrics, utils)
├── compare_models.py       # aggregate & compare all trained models for a dataset
├── create_config.py        # generate train/model configs and hashes
├── custom_dataset.py       # Generic PyTorch Dataset with preprocessing
├── extract_rslt_stats.py   # extract and plot detailed results
├── test_model.py           # evaluate trained model on test set
├── train_model.py          # train model from config hash
├── run-kan.sh              # single‑experiment launcher
├── run-kan-sweep.sh        # hyperparameter sweep launcher
└── make-venv.sh            # virtual environment setup
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rbf-kan.git
   cd rbf-kan
   ```

2. **Create and activate a virtual environment** (Python 3.12+ required)
   ```bash
   ./make-venv.sh          # creates .venv and installs dependencies
   source .venv/bin/activate
   ```
   The script installs all the necessary packages.

3. **Prepare a dataset** (e.g., CIFAR‑10)
   ```bash
   cd cifar10
   python prepare_dataset.py
   cd ..
   ```
   This downloads the raw data, creates pickle files, computes normalisation statistics, and generates label mappings.

## Usage Overview

All scripts are designed to work with **config hashes** – a SHA‑1 hash derived from the full training and model configuration. This ensures that every hyperparameter combination produces a unique, reproducible directory under `train/<hash>/test_<version>/`.

### 1. Train a single model (`run-kan.sh`)

This script is the easiest way to launch one experiment. It:
- builds the configuration,
- generates the hash,
- trains the model,
- tests it on the best epoch,
- extracts statistics and plots.

**Example** (inside the project root):
```bash
./run-kan.sh --dataset cifar10 --layers "36" --num-grids "4" --epochs 100 --batch 100 --lr 5e-2
```

**Common flags**:
- `--dataset cifar10|cifar100|mnist`
- `--layers "32 64 32"` (space‑separated hidden layer sizes)
- `--num-grids "4 6"` (grid sizes per layer)
- `--residual`, `--dynamic`, `--use-v2`, `--no-normalize`, `--no-normalize-rbf`
- `--dropout 0.1`, `--dropout-linear 0.1`
- `--epochs`, `--patience`, `--batch`, `--lr`, `--optimizer`, etc.
- `--hash <existing_hash>` – reuse a previously generated config (skip config creation)

After execution, the output is stored in:  
`cifar10/train/<hash>/test_0/` (or `test_<version>/` if `--test-version` is set).

### 2. Hyperparameter sweep (`run-kan-sweep.sh`)

Edit the arrays at the top of the script to define the grid of parameters. Then run:

```bash
./run-kan-sweep.sh -j 2 --max-experiments 100
```

- `-j N` : number of parallel jobs (be careful with GPU memory).
- `--max-experiments N` : limit the total number of combinations.
- `-y` : skip confirmation prompt.
- `--no-pbar` : disable progress bars.

Each combination is run, and a `hyperparameters.csv` file is saved alongside the config. The sweep log is written to `train/sweep_results/sweep_log.txt`.

### 3. Compare all trained models (`compare_models.py`)

After several experiments are finished, run:

```bash
python compare_models.py --dataset cifar10
```

This script:
- Walks through `cifar10/train/` and loads every `history.json`.
- Extracts model hyperparameters from the config files (`num_layers`, `hidden_layers`, `mode`, `grids`, `dropout`, etc.).
- Creates a DataFrame with all metrics (`Accuracy`, `F1Score`, `AUROC`, `loss`).
- Prints the full configuration table, top‑5 models for each key metric, and min/max summaries.
- Generates two bar charts in `train/comparison/`:
  - `global.png` – all metrics for all runs.
  - `top30_f1score.png` – the 30 best runs by F1 score.

Additional options:
- `-l 10` : truncate hash prefixes to 10 characters in plots.
- `-d /path/to/train` : specify a different training root.

### 4. Extract detailed results for a single run (`extract_rslt_stats.py`)

For a specific hash, you can regenerate plots and confusion matrices:

```bash
python extract_rslt_stats.py --dataset cifar10 --hash <hash> --test-version 0 --epoch best
```

It produces:
- Training vs. validation loss plot (`plot/tr_vs_val.png`).
- Precision‑recall curve (if binary classification).
- Confusion matrices for each label group (e.g., `class`).
- Per‑target regression plots (if applicable).

## Custom Dataset Support

To add a new dataset (e.g., `my_dataset`):

1. Create a folder `my_dataset/`.
2. Copy `cifar10/prepare_dataset.py` into it and modify:
   - `DATASET_NAME` and `DATASET_DIR`.
   - `get_dataset_info()` – return `num_classes`, `input_shape`, `task`.
   - `get_dataset(subset)` – return numpy arrays `(data, labels)`.
   - `get_class_names()` – list of class names.
   - Optionally override `get_groups()` for hierarchical confusion matrices.
3. Place your raw data inside `my_dataset/dataset/` or implement download logic in `build_dataset()`.
4. Use `--dataset my_dataset` in all scripts.

The `custom_dataset.py` provides a `GenericDataset` class that handles preprocessing, flattening, and task‑specific target transformations (one‑hot ↔ indices).

## Configuration Generation (`create_config.py`)

This script is called internally by the launchers, but can be used standalone to inspect or export configurations:

```bash
python create_config.py --dataset cifar10 --layers "64 64" --export --hash
```

- `--export` : writes `train.json` and `model.json` to the appropriate directory.
- `--hash` : prints only the hash (useful for scripting).

All arguments mirror those described in `run-kan.sh`.

## About the Model

You can read more about the RBF-KAN architecture and its implementation details in `rbfkan_utils/models/README.md`. The core idea is to replace linear transformations with RBF expansions, allowing the model to learn more complex, non‑linear relationships. Each layer learns an activation function based on learnable grid points and inverse denominators and the chosen RBF mode. The architecture supports various RBF modes  and can be configured with residual connections, normalisation, and dropout.

## License

This project is released under the **MIT License** (see `LICENSE` file).