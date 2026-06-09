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

# class FlattenBatch:
#     def __init__(self, data_dim):
#         super(FlattenBatch,self).__init__()
#         self.data_dim  = int(data_dim)
        
#     @classmethod
#     def __find_batch_size(cls, shape):
#         a = 1
#         for dim in shape:
#             a *= dim
#         return a
        
#     def __flatten(self, x : Union[torch.Tensor, tuple[torch.Tensor]]):
#         if isinstance(x, (tuple, list)) :
#             for _iter, _ in enumerate(x):
#                 batch_shape = self.__find_batch_size(x[_iter].shape[:self.data_dim])
#                 data_shape = x[_iter].shape[self.data_dim:]
#                 x[_iter] = x[_iter].resize_(batch_shape, *data_shape)
#             return x
#         else :
#             batch_shape = self.__find_batch_size(x.shape[:self.data_dim])
#             data_shape = x.shape[self.data_dim:]
#             return x.resize_(batch_shape, *data_shape)

#     def __call__(self, data : torch.Tensor, target : torch.Tensor, *args, **kwargs):
#         data = self.__flatten(data)
#         target = self.__flatten(target)
#         return data, target

# class ProbabilityAdjuster :
#     def __init__(
#         self,
#         input : list = None,
#         input_categories = None,
#         output : list = None,
#         output_categories = None,
#         confusion_matrix : pd.DataFrame = None,
#         metric_name = 'Accuracy',
#         smoothing_coef = 0.75,
#         log_dir : str = None,
#         saving_interval : int = 1,
#         saving_is_blocking : bool = True,
#     ):
#         # Translate input column names to positions
#         self.input, self.input_categories, self.input_regression_type = \
#             self._initialize(input, input_categories)
            
#         # Translate output column names to positions
#         self.output, self.output_categories, self.output_regression_type = \
#             self._initialize(output, output_categories)
            
#         # Initiate target metric & smoothing coefficient
#         self.metric_name    = metric_name
#         self.smoothing_coef = smoothing_coef
        
#         # Transform confusion matrix
#         self.input_cols  = [*[self.input[_]  for _ in self.input_regression_type],  *self.input_categories.keys()]
#         self.output_cols = [*[self.output[_] for _ in self.output_regression_type], *self.output_categories.keys()]

#         if confusion_matrix is None :
#             self.cm = torch.ones(len(self.input_cols), len(self.output_cols)) / len(self.input_cols)
#             common = np.asarray(self.input_cols)[np.isin(self.input_cols, self.output_cols)]
#             couples = ((self.input_cols.index(_),self.output_cols.index(_)) for _ in common)
            
#             for idx_i, cols_i in couples:
#                 self.cm[idx_i,cols_i] = 1.
#         else :
#             self.cm = torch.tensor(
#                 pd.DataFrame(confusion_matrix).loc[pd.Index(self.input_cols), self.output_cols].values, 
#                 requires_grad=False
#             )
            
#         self.cm = self.cm.float().abs() ** 2
        
#         self.logs = None if log_dir is None else os.path.join(log_dir, 'ProbabilityAdjuster.json')
#         if self.logs is not None:            
#             if os.path.exists(self.logs):
#                 self.prob_dict = load_dict(self.logs)
#             else :
#                 self.prob_dict = {
#                     'input_probabilities' : {
#                         label : {} for label in self.input_cols
#                     }, 
#                     'output_probabilities' : {
#                         label : {} for label in self.output_cols
#                     },
#                 }
#             self.saving_interval = saving_interval
#             self.saving_counter = saving_interval
            
#             self.saving_is_blocking = saving_is_blocking
#             if not self.saving_is_blocking:
#                 self.mp = None
        
#         # Initiate probabilities
#         self._input_prob  = torch.ones(len(self.input_cols),  requires_grad=False).float()
#         self._output_prob = torch.ones(len(self.output_cols), requires_grad=False).float()
        
#     @classmethod  
#     def _initialize(cls, columns, categories) :
#         if categories is None:
#             categories = []
#         columns = {
#             key : idx for idx, key in enumerate(columns) 
#         }
#         categories = {
#             category[0][:category[0].find('_Is_')] : [
#                 columns[key] for key in category
#             ] for category in categories
#                 if len(category) > 0
#         }
#         regression_type = list(columns.values())
        
#         for category in categories.values():
#             regression_type = [
#                 idx for idx in regression_type
#                     if idx not in category
#             ]
#         columns = {idx : key for key, idx in columns.items()}
#         return columns, categories, regression_type
    
#     @classmethod  
#     def _to_prob(cls, logits):
#         # logits[logits < 0.] = 0.
#         # logits[logits > 1.] = 1.
#         # return logits ** 2
#         return logits.abs() * torch.sigmoid(logits)
    
#     @classmethod  
#     def _expand_prob(cls, prob, columns, categories, regression_type):
#         target_prob = torch.empty(len(columns), dtype=prob.dtype, device=prob.device)

#         # Copy regression-type probabilities
#         target_prob[regression_type] = prob[:len(regression_type)]
        
#         # Copy probabilities for each category
#         for offset, category in enumerate(categories.values()):
#             target_prob[category] = target_prob[len(regression_type)+offset].clone()
        
#         return target_prob
    
#     def _backprop_prob(self, out_prob):
#         return (out_prob @ self.cm.T) / self.cm.sum(1)
    
#     def _smooth(self, x, x_smoothed):
#         return self.smoothing_coef * x + (1-self.smoothing_coef) * x_smoothed
    
#     def _update_logs(self, epoch, in_prob, out_prob):
#         if self.logs is not None:
#             # Add input probabilities
#             for idx, val in enumerate(in_prob):
#                 self.prob_dict['input_probabilities'][self.input_cols[idx]][epoch] = val
                
#             # Add output probabilities
#             for idx, val in enumerate(out_prob):
#                 self.prob_dict['output_probabilities'][self.output_cols[idx]][epoch] = val
                
#             self.saving_counter -= 1
#             if self.saving_counter == 0:
#                 self.save_logs(timeout=10)
#                 self.saving_counter = self.saving_interval
                
                
#     def __save_logs(self, prob_dict):
#         os.makedirs(os.path.dirname(self.logs), exist_ok=True)
#         self.logs = save_dict(prob_dict, self.logs)
                
#     def save_logs(self, timeout=None):
#         if self.logs is not None:
#             if not self.saving_is_blocking:
#                 if self.mp is not None:
#                     self.mp.join(timeout)
#                     if timeout is not None and self.mp.is_alive():
#                         print("Warning: Previous saving process is still running and timeout reached!")
#                         return
#                 assert self.mp is None or not self.mp.is_alive(), "Previous saving process is still running!"    
#                 self.mp = torch.multiprocessing.get_context('spawn').Process(
#                     target=self.__save_logs,
#                     args=(self.prob_dict.copy(),)
#                 )
#                 self.mp.start()
#             else :
#                 self.__save_logs(self.prob_dict.copy())
                
#     def update(
#         self,
#         history : dict[str, list | float] = None,
#         epoch  = 0,
#         **kwargs
#     ):
#         with torch.no_grad():
#             acc = history['val'][epoch][self.metric_name]
#             if isinstance(acc, dict):
#                 acc = list(acc.values())
#             out_prob = self._to_prob(torch.tensor(acc).float()).view(-1)
            
#             self._output_prob = self._smooth(out_prob, self._output_prob)
#             in_prob = self._backprop_prob(out_prob)
            
#             self._input_prob = self._smooth(in_prob, self._input_prob)
#             self._update_logs(epoch, in_prob.tolist(), out_prob.tolist())
        
#     def get_input_prob(self, *args, expanded = False, **kwargs):
#         if expanded:
#             return self._expand_prob(
#                 self._input_prob.detach().clone(),
#                 self.input,
#                 self.input_categories,
#                 self.input_regression_type,
#             )
#         else :
#             return self._input_prob.detach().clone()
    
#     def get_output_prob(self, *args, expanded = False, **kwargs):
#         if expanded:
#             return self._expand_prob(
#                 self._output_prob.detach().clone(),
#                 self.output,
#                 self.output_categories,
#                 self.output_regression_type,
#             )
#         else :
#             return self._output_prob.detach().clone()
    
#     def __call__(self, *args, **kwargs):
#         self.update(*args, **kwargs)

# class MaskInput :
#     def __init__(
#         self, 
#         input : list = None,
#         input_categories = None,
#         max_probability = 0.25,
#         x_shift = 0.,
#         masked_value = -1,
#     ):
#         self.max_probability = max_probability
#         self.x_shift = x_shift
#         self.masked_value = torch.tensor(masked_value)
        
#         self.input = input
#         self.input_categories = input_categories
        
#         if self.input_categories is None:
#             self.input_categories = []
            
#         if self.input is None:
#             self.input_categories = []
            
#         else :
#             self._initialize_input(input)
    
#     def _initialize_input(self, input) :
#             self.input = {
#                 key : idx for idx, key in enumerate(self.input) 
#             }
#             self.input_categories = [[
#                     self.input[key] for key in category
#                 ] for category in self.input_categories
#             ]
#             self.input_regression_type = list(self.input.values())
            
#             for category in self.input_categories:
#                 self.input_regression_type = [
#                     idx for idx in self.input_regression_type
#                         if idx not in category
#                 ] 
#             self._initialize_input = lambda input : None
            
#     def __call__(
#         self,
#         data        : torch.Tensor, 
#         epoch       : int,
#         epochs      : int,
#         dataloader  : Iterable,
#         iteration   : int = None,
#         device      = torch.device('cpu'),
#         probability_adjuster : ProbabilityAdjuster = None,
#         **kwargs
#     ) :
#         self._initialize_input(data.shape[-1])
#         if iteration is None:
#             iteration = len(dataloader)
            
#         probability = self.max_probability / (1 + np.exp( ((epoch + (iteration / len(dataloader))) / epochs) - self.x_shift))
        
#         if probability_adjuster is not None:
#             probability *= probability_adjuster.get_input_prob(expanded=True)
#         probability = torch.as_tensor(probability, device = device)
        
#         mask = torch.rand(len(self.input_regression_type) + len(self.input_categories), device = device)
        
#         mask_extended = torch.empty(len(self.input), device = device, dtype = mask.dtype)
#         mask_extended[self.input_regression_type] = mask[:len(self.input_regression_type)]
        
#         for offset, category in enumerate(self.input_categories):
#             mask_extended[category] = mask[len(self.input_regression_type) + offset]
            
#         mask_extended = mask_extended < probability
        
#         data.masked_fill_(mask_extended, self.masked_value.to(data.dtype).to(device))

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