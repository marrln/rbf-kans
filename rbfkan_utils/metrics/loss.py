from typing import Literal
import torch
# from ..models import MultiHead

class WeightedLoss(torch.nn.Module):
    def __init__(
        self, 
        loss
    ):
        super(WeightedLoss,self).__init__()
        self.isintegrated = hasattr(loss, 'weight')
        self.loss = loss
        
        if not self.isintegrated:
            self.reduction = loss.reduction
            self.loss.reduction = 'none'
            
            if self.reduction not in ('sum','mean','none'):
                raise ValueError(f'Unsupported reduction type; got {self.reduction}')
        
    def forward(self, input : torch.Tensor, target : torch.Tensor, weight : torch.Tensor = None):
        if self.isintegrated:
            self.loss.weight = weight
            loss = self.loss(input, target)
            self.loss.weight = None
            return loss
        
        loss = self.loss(input, target)
        
        if weight is not None:
            # print(input.shape, target.shape, weight.shape, loss.shape)
            loss *= weight / weight.mean()
        
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else :
            return loss

# class MixedLoss(torch.nn.Module):
#     '''A wrapper for wrapping criteria when the predicted and target values
#     contain one-hot encoded data and regression-type data.
#     '''
#     def __init__(
#         self, 
#         output_cols : list, 
#         categories : list[list[str]], 
#         categoriesLoss = torch.nn.BCEWithLogitsLoss(), 
#         regressionLoss = torch.nn.MSELoss(),
#         reduction : Literal['sum','mean','random','none'] = 'sum'
#     ):
#         '''
#         Parameters
#         ----------
#         output_cols : list
#             The columns to expect in the predicted and target values in order of appearance. 
#         categories : list[list[str]]
#             The columns of `output_cols` grouped in categories. Columns not present in `categories`
#             are treated as regression-type data.
#         categoriesLoss = torch.nn.BCEWithLogitsLoss(), 
#             The loss to use in categorized columns.
#         regressionLoss = torch.nn.MSELoss()
#             The loss to use in regression-type columns.
#         '''
#         super(MixedLoss,self).__init__()
        
#         self.categoriesLoss = categoriesLoss
#         self.regressionLoss = regressionLoss
#         self.output_cols = output_cols
        
#         self.categoriesCols = [
#             [output_cols.index(label) for label in group_i ]
#                 for group_i in categories
#         ]
#         self.regressionCols = []
        
#         for group_i in categories:
#             self.regressionCols.extend(group_i)
        
#         self.regressionCols = [
#             output_cols.index(label) 
#                 for label in output_cols 
#                 if label not in self.regressionCols
#         ]
#         self.reduction = reduction
#         if self.reduction not in ('sum','mean','random','none'):
#             raise ValueError(f'Unrecognised reduction type; got {self.reduction}')
        
#         self.probabilities = torch.ones(len(self.regressionCols)+len(self.categoriesCols))
        
#         # print(self.regressionCols, self.categoriesCols)
#         # exit(-1)
        
#     def update_probabilities(self, prob):
#         self.probabilities = prob 
#         if torch.sum(self.probabilities) == 0:
#             self.probabilities = torch.ones_like(self.probabilities)
        
#     def forward(self, pred : torch.Tensor, targ : torch.Tensor):
#         loss = [*[
#                 self.regressionLoss(pred[:,idx], targ[:,idx])
#                     for idx in self.regressionCols
#             ],
#             *[
#                 self.categoriesLoss(pred[:,group_i], targ[:,group_i])
#                     for group_i in self.categoriesCols
#         ]]
#         if self.reduction == 'sum':
#             return torch.stack(loss).sum()
#         elif self.reduction == 'mean':
#             return torch.stack(loss).mean()
#         elif self.reduction == 'random':
#             x = [0]
#             while sum(x) == 0:
#                 x = torch.rand(len(loss)) * self.probabilities
#                 x = x < (2./len(x))
#             return torch.stack([loss[idx] for idx, x_i in enumerate(x) if x_i]).mean()
#         elif self.reduction == 'none':
#             # flat = []
#             # for category in loss:
#             #     flat.extend(torch.flatten(category))
#             # return torch.stack(flat)
#             loss_dict = {
#                 self.output_cols[self.regressionCols[idx]] : loss[idx].double().cpu().item()
#                     for idx in range(len(self.regressionCols))
#             }
#             for offset, category in enumerate(self.categoriesCols):
#                 label = self.output_cols[category[0]]
#                 label = label[:label.find('_Is_')]
#                 loss_dict[label] = loss[len(self.regressionCols)+offset].double().cpu().item()
                
#             return loss_dict
#         else :
#             raise ValueError(f'Unrecognised reduction type; got {self.reduction}')
    
# class Accuracy2Loss(torch.nn.Module):
#     '''A wrapper for transforming accuracy metrics to loss criteria.
    
#     Args
#     ----------
#     target: Module
#         The target accuracy metric class.
#     *args : Any, Optional
#         If `instantiated == False`, the positional arguments for `target`, if any.
#     instantiated: bool, Optional
#         If `True`, target is treated as a callable object and any extra arguments are ignored. Default is `False`.
#     **kwargs : Any, Optional
#         If `instantiated == False`, the keyword arguments for `target`, if any.
#     '''
#     def __init__(
#         self, 
#         target : torch.nn.Module,
#         *args, 
#         instantiated = False,
#         **kwargs
#     ):
#         super(Accuracy2Loss,self).__init__()
#         if instantiated:
#             self.target = target
#         else :
#             self.target = target(*args, **kwargs)
        
#     def forward(self, pred : torch.Tensor, targ : torch.Tensor):
#         acc = self.target(pred, targ)
#         return torch.ones_like(acc) - acc
    
# class MultiHeadLoss(MultiHead):
#     def __init__(
#         self, 
#         *args, 
#         reduction : Literal['sum','mean','random','none'] = 'sum',
#         expect_type : Literal['list','dict'] = 'list',
#         **kwargs
#     ):
#         if len(kwargs) > 0:
#             args += kwargs,
#         super(MultiHeadLoss,self).__init__(
#             *args,
#             return_type = expect_type,
#         )
        
#         if reduction not in ['sum','mean','random','none']:
#             raise NotImplementedError(f'Reduction method "{reduction}" is not implemented.')
#         self.reduction = reduction
        
#         self.probabilities = torch.ones(len(self.heads))
        
#     def update_probabilities(self, prob):
#         self.probabilities = prob.to(self.probabilities.device)
#         if torch.sum(self.probabilities) == 0:
#             self.probabilities = torch.ones_like(self.probabilities)
        
#     def to(self, device):
#         self.probabilities = self.probabilities.to(device)
#         return super().to(device)
        
#     def forward(self, preds : torch.Tensor, targs : torch.Tensor):
#         if self.return_type == 'list':
#             assert len(preds) == len(targs) == len(self.heads), f"Unexpected number of values for pred ({len(preds)}), targ ({len(targs)}); expected ({len(self.heads)})"
#             keys = [f'head.{_iter}' for _iter in range(len(self.heads))]
#             heads = self.heads
#         elif self.return_type == 'dict':
#             keys = self.heads.keys() if isinstance(self.heads, torch.nn.ModuleDict) else [
#                 f'head.{_iter}' for _iter in range(len(self.heads))
#             ]
#             keys = set.intersection(set(preds.keys()), set(targs.keys()), set(keys),)
#             targs = [targs[key]      for key in keys]
#             preds = [preds[key]      for key in keys]
#             heads = [self.heads[key] for key in keys]
#         else :
#             raise NotImplementedError(f'Return type "{self.return_type}" is not supported.')

#         loss = tuple(
#             head(pred, targ)
#                 for head, pred, targ in zip(
#                     heads,
#                     # self.heads if isinstance(self.heads, torch.nn.ModuleList) else self.heads.values(),
#                     preds,
#                     targs
#                 )
#         )
#         # loss = ()
#         # for key, head, pred, targ in zip(
#         #     keys,
#         #     heads,
#         #     # self.heads if isinstance(self.heads, torch.nn.ModuleList) else self.heads.values(),
#         #     preds,
#         #     targs
#         # ):
#         #     print('MultiHeadLoss', key, pred.shape, targ.shape)
#         #     print(pred[0])
#         #     print(targ[0])
#         #     loss += head(pred, targ),
            
#         if self.reduction == 'sum':
#             return torch.stack(loss).sum()
#         elif self.reduction == 'mean':
#             return torch.stack(loss).mean()
#         elif self.reduction == 'random':
#             x = [0]
#             while sum(x) == 0:
#                 x = torch.rand(len(loss)) * self.probabilities
#                 x = x < (2./len(x))
#             return torch.stack([loss[idx] for idx, x_i in enumerate(x) if x_i]).mean()
#         elif self.reduction == 'none':
#             if self.return_type == 'list':
#                 try :
#                     loss = tuple(
#                         loss_i.double().cpu().item()
#                             for loss_i in loss 
#                     )
#                 except :
#                     pass
#                 return loss
#             elif self.return_type == 'dict':
#                 try :
#                     loss = {
#                         key : loss_i.double().cpu().item()
#                             for key, loss_i in zip(keys, loss)
#                     }
#                 except :
#                     loss = {
#                         key : loss_i
#                             for key, loss_i in zip(keys, loss)
#                     }
#             return loss
#         else :
#             raise ValueError(f'Unrecognised reduction type; got {self.reduction}')
    
class CombinedLoss(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.loss = torch.nn.ModuleList(args)
        self.w = torch.nn.Parameter(torch.zeros([len(self.loss)], requires_grad=True))

    def forward(self, *args, **kwargs):
        return torch.exp(-self.w) @ torch.stack([
            # print(loss, args[1].sum(-1)) or loss(*args, **kwargs)
            loss(*args, **kwargs)
                for loss in self.loss
        ]) + self.w.sum()