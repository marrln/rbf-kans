import numpy as np
import torch
from torch.utils.data import Dataset


class GenericDataset(Dataset):
    def __init__(
        self,
        data,
        labels,
        coarse_labels=None,
        task=None,
        return_key=False,
        return_weights=None,
        preprocess_data=None,
        preprocess_targ=None,
        flatten=False,
    ):
        self.data = data
        self.labels = labels
        self.coarse_labels = coarse_labels
        self.task = task
        self.return_key = return_key
        self.return_weights = return_weights
        self.preprocess_data = preprocess_data
        self.preprocess_targ = preprocess_targ
        self.flatten = flatten

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.coarse_labels is not None:
            coarse_y = self.coarse_labels[idx]
            y = (y, coarse_y)
        
        # Preprocess image (expects H,W,C)
        if self.preprocess_data is not None:
            try:
                x = self.preprocess_data(x=x)
            except TypeError:
                x = self.preprocess_data(x)
        else:
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            elif not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)

        if isinstance(x, torch.Tensor) and x.dtype != torch.float32:
            x = x.float()
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        if self.flatten and hasattr(x, 'reshape'):
            x = x.reshape(-1)

        # Process target
        if self.preprocess_targ is not None:
            try:
                y = self.preprocess_targ(target=y)
            except TypeError:
                y = self.preprocess_targ(y)
        else:
            # Convert to tensor
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y)
            elif not isinstance(y, torch.Tensor):
                y = torch.tensor(y)

            # Adjust based on task
            if self.task == 'multiclass':
                # For CrossEntropyLoss: need class indices (long, scalar or 0D)
                if y.numel() > 1:
                    # One-hot -> index
                    y = y.argmax().long()
                else:
                    # Already scalar (or 1-element vector)
                    y = y.long().squeeze()
            elif self.task in ('multilabel', 'binary'):
                # For BCEWithLogitsLoss: need one-hot float (or multi-label)
                # If y is already one-hot (2D), keep as float; if index, convert to one-hot
                if y.numel() == 1:
                    # Single index -> create one-hot
                    num_classes = 2 if self.task == 'binary' else len(self.labels[0]) if hasattr(self.labels[0], '__len__') else 2
                    one_hot = torch.zeros(num_classes, dtype=torch.float32)
                    one_hot[y.long()] = 1.0
                    y = one_hot
                elif y.dim() == 1 and y.sum() == 1.0:
                    # One-hot already
                    y = y.float()
                else:
                    # Multi-label vector
                    y = y.float()
            else:
                # Regression: keep as float
                y = y.float()

        # Return with optional weights and key
        if self.return_weights:
            weights = torch.ones_like(y, dtype=torch.float32)
            if self.return_key:
                return x, y, weights, idx
            return x, y, weights
        if self.return_key:
            return x, y, idx
        return x, y