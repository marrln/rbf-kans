from typing import Iterable, Union, Literal
import torch
import numpy as np
import pandas as pd
import os

# from rbfkan_utils.utils import save_dict, load_dict

class UpdatableFloat(float):
    def __new__(cls, x = 0):
        inst = super().__new__(cls, x)
        inst._value = [x]
        return inst

    def __iadd__(self, other):
        self._value[0] += other
        return self

    def __imul__(self, other):
        self._value[0] *= other
        return self

    def __float__(self):
        return float(self._value[0])
    
    def __repr__(self):
        return f"UpdatableFloat({self._value[0]})"
    
    def set(self, x):
        self._value[0] = x
        
    def value(self):
        return self._value[0]

class GatherStatistics :
    def __init__(
        self,
        input_cols,
        output_cols,
        task : Literal['binary','as_binary','multiclass','multilabel','regression'] = 'regression',
        export_path = None,
        overwrite = False,
    ):
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.task = task
        self.reset()
        self.export_path = export_path
        self.overwrite = overwrite
        
        if os.path.isfile(self.export_path):
            if self.overwrite:
                _iter = 0
                while os.path.isfile(self.export_path):
                    if isinstance(self.overwrite, int) and _iter >= self.overwrite:
                        self.locked = True
                        break
                    tmp = os.path.splitext(self.export_path)
                    self.export_path = tmp[0].rstrip(f'({_iter})') + f'({_iter+1})' + tmp[1]
                    _iter += 1
            else :
                self.locked = True
        
    def reset(self, *args, **kwargs):
        self.stats = pd.DataFrame(
            index   = self.input_cols + (
                ['Label'] if self.task in ('binary','as_binary','multiclass') else
                self.output_cols 
            ),
            columns = ['count','count_na','sum_x','sum_x2','min','max']
        )
        self.stats[self.stats.columns] = np.zeros_like(self.stats.values)
        self.stats['min'] = float('inf')
        self.stats['max'] = -float('inf')
        self.stats.index.name = 'Column'
        self.locked = False        
        
    def update(self, *args, data = None, target=None, **kwargs):
        if not self.locked:
            with torch.no_grad():
                if isinstance(data, np.ndarray):
                    data = torch.tensor(data)
                if isinstance(data, torch.Tensor):
                    if len(data.shape) == 1:
                        data = data.unsqueeze(-1)
                    na_vals = data.isnan()
                    self.stats.loc[self.input_cols, ['count']]    += len(data)
                    self.stats.loc[self.input_cols, ['count_na']] += na_vals.sum(0).unsqueeze(-1).cpu().numpy()
                    self.stats.loc[self.input_cols, ['min']]       = np.stack([
                        torch.masked.amin(data, dim=0, mask=~na_vals).cpu().unsqueeze(-1).numpy(), 
                        self.stats.loc[self.input_cols, ['min']].values        
                    ]).min(axis=0)
                    self.stats.loc[self.input_cols, ['max']]       = np.stack([
                        torch.masked.amax(data, dim=0, mask=~na_vals).cpu().unsqueeze(-1).numpy(), 
                        self.stats.loc[self.input_cols, ['max']].values        
                    ]).max(axis=0)
                    self.stats.loc[self.input_cols, ['sum_x']]    += torch.masked.sum(data   , dim=0, mask=~na_vals).cpu().unsqueeze(-1).numpy()
                    self.stats.loc[self.input_cols, ['sum_x2']]   += torch.masked.sum(data**2, dim=0, mask=~na_vals).cpu().unsqueeze(-1).numpy()
                
                if isinstance(target, np.ndarray):
                    target = torch.tensor(target)
                if isinstance(target, torch.Tensor):
                    data = target
                    if len(data.shape) == 1:
                        data = data.unsqueeze(-1)
                    na_vals = data.isnan()
                    
                    if self.task in ('binary','as_binary','multiclass') :
                        self.stats.loc[['Label'], ['count']]    += len(data)
                        self.stats.loc[['Label'], ['count_na']] += na_vals.sum(0).unsqueeze(-1).cpu().numpy()
                    
                    else :
                        self.stats.loc[self.output_cols, ['count']]    += len(data)
                        self.stats.loc[self.output_cols, ['count_na']] += na_vals.sum(0).unsqueeze(-1).cpu().numpy()
                        
                        self.stats.loc[self.output_cols, ['min']]       = np.stack([
                            torch.masked.amin(data, dim=0, mask=~na_vals).cpu().unsqueeze(-1).numpy(), 
                            self.stats.loc[self.output_cols, ['min']].values        
                        ]).min(axis=0)
                        self.stats.loc[self.output_cols, ['max']]       = np.stack([
                            torch.masked.amax(data, dim=0, mask=~na_vals).cpu().unsqueeze(-1).numpy(), 
                            self.stats.loc[self.output_cols, ['max']].values        
                        ]).max(axis=0)
                        self.stats.loc[self.output_cols, ['sum_x']]    += torch.masked.sum(data   , dim=0, mask=~na_vals).cpu().unsqueeze(-1).numpy()
                        self.stats.loc[self.output_cols, ['sum_x2']]   += torch.masked.sum(data**2, dim=0, mask=~na_vals).cpu().unsqueeze(-1).numpy()
            
    def finalize(self, *args, **kwargs):
        if not self.locked:
            self.stats['mean'] = self.stats['sum_x']  / (self.stats['count'] - self.stats['count_na']) 
            self.stats['std']  = self.stats['sum_x2'] / (self.stats['count'] - self.stats['count_na']) - self.stats['mean']**2
            self.stats.drop(columns=['sum_x','sum_x2'], inplace=True)
            
            if isinstance(self.export_path, str):
                import os
                os.makedirs(os.path.dirname(self.export_path), exist_ok=True)
                self.stats.to_csv(self.export_path)
            self.locked = True
        
    def __call__(self, *args, **kwds):
        return self.update(*args, **kwds)