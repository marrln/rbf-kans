import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np


class DataFrameToDataset (Dataset): 
    def __init__(
        self, 
        df : pd.DataFrame, 
        input_cols  : list[str], 
        output_cols : list[str], 
        return_weights : str = False,
        return_key = False,
    ):
        super().__init__()
        
        self.index  = pd.Series(df.copy().index)
        self.df = df.reset_index()
        self.input_cols, self.output_cols = input_cols, output_cols
        
        self.i_input_cols  = [self.df.columns.tolist().index(_) for _ in self.input_cols]
        self.i_output_cols = [self.df.columns.tolist().index(_) for _ in self.output_cols]
        
        self._return_key = return_key
        assert not bool(return_weights) or return_weights in df.columns
        self._return_weights = return_weights if not return_weights else self.df.columns.tolist().index(return_weights)
        
    def get_keys(self, index):
        return self.index.loc[list(index)].tolist()
        
    def __len__(self) :
        return len(self.df)
    
    def return_key(self, return_key = True):
        self._return_key = return_key
    
    def __getitem__(self, index):
        return self.__getitems__([index,])[0]
    
    def __getitems__(self, index):
        data = self.df.iloc[index, self.i_input_cols].values 
        targ = self.df.iloc[index, self.i_output_cols].values
        
        to_zip = (torch.tensor(data).float(), torch.tensor(targ).float(),)
        
        if self._return_weights:
            to_zip += torch.tensor(self.df.iloc[index, [self._return_weights,]].values).float(),
        
        if self._return_key:
            to_zip += self.get_keys(index),

        return [_ for _ in zip(*to_zip)]


def group(df : pd.DataFrame, indices = None, labels = None, label_dict = {}):
    '''A grouper for datasets with categorical data.
    
    Args
    ----
    df: DataFrame
        The target dataframe
    labels: list, Optional
        The target columns
    indices: list, Optional
        If not specified, assume `df.reset_index().index`.
    label_dict: dict[str, list], Optional
        The labels to apply the group. If not specified, assume 
        `label_dict = {
            label : df[label].unique()
                for label in labels
        }`
    Returns
    -------
    dict[str, dict] | dict[str, list[int]]
    '''
    if labels is not None and len(label_dict) == 0:
        label_dict = {
            label : df[label].unique()
                for label in labels
        }
    if indices is None:
        indices = df.index.to_list()
        
        # Reverse dictionary key order
        dict_label = {}
        while  len(label_dict) > 0:
            key, val = label_dict.popitem()
            dict_label[key] = val
        
        label_dict = dict_label
        
    if len(label_dict) < 1 or df.empty or len(indices) < 1:
        return indices
    
    key, labels = label_dict.popitem()
    # print(key, labels)
    df = df.loc[indices]
    subgroup = {
        f'{key} ({label})' : group(
            df,
            indices     = (
                df[df[key].isna()] if isinstance(label, float) and np.isnan(label) 
                    else df[df[key] == label]
            ).index.to_list(),
            label_dict  = label_dict.copy(),
        )
            for label in labels 
    }
    for key in list(subgroup.keys()):
        if len(subgroup[key]) == 0:
            subgroup.pop(key)
    
    return subgroup


def split_dataset(splits, full_dataset, seed = None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return random_split(full_dataset, splits, generator=generator)


def smart_split_dataset(splits: list[float], full_dataset, groups: dict[str, list] | dict[str, dict], seed=None):
    import torch
    from torch.utils.data import random_split

    sets = [[] for _ in splits]
    for key, val in groups.items():
        if isinstance(val, dict):
            # Recursively process nested groups
            subsets = smart_split_dataset(splits, full_dataset, val, seed)
        elif isinstance(val, list):
            # Leaf is a list of indices – split it directly
            total = len(val)
            split_sizes = [int(p * total) for p in splits[:-1]]
            split_sizes.append(total - sum(split_sizes))
            # Use a dummy dataset to leverage random_split
            dummy = torch.utils.data.TensorDataset(torch.arange(total))
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed)
            split_dummy = random_split(dummy, split_sizes, generator=generator)
            subsets = [[val[i] for i in sub.indices] for sub in split_dummy]
        else:
            # Assume val is a Dataset or Subset (original behavior)
            subsets = split_dataset(splits, val, seed)
            subsets = [[val[_] for _ in subset.indices] for subset in subsets]

        for _iter, subset in enumerate(subsets):
            sets[_iter].extend(subset)

    if full_dataset is not None:
        sets = [torch.utils.data.Subset(full_dataset, idx) for idx in sets]
    return sets


if __name__ == '__main__':
    df = pd.DataFrame({
        i : ((torch.tensor(range(100)) / 100.) ** i).tolist()
            for i in range(5)
    })
    input_cols  = df.columns[:len(df.columns)-1]
    output_cols = df.columns[len(df.columns)-2:]
    
    dataset = DataFrameToDataset(
        df          = df,
        input_cols  = input_cols,
        output_cols = output_cols,
    )
    print('Dataset lenght', len(dataset))
    
    splits = [0.8, 0.2]
    datasets = split_dataset(splits, dataset, 42)
    
    for i, (split, dataset_i) in enumerate(zip(splits, datasets), start=1):
        print('Split', i, '-- Percentage =', split, '-- Counts =', len(dataset_i))