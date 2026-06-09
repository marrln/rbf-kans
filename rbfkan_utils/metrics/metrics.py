import torchmetrics
import torch

class OneHotMulticlassAccuracy(torchmetrics.classification.MulticlassAccuracy):
    def __init__(self, num_classes=None, top_k=1, average='macro', multidim_average='global', ignore_index=None, validate_args=True, **kwargs):
        super(OneHotMulticlassAccuracy, self).__init__(num_classes, top_k, average, multidim_average, ignore_index, validate_args, **kwargs)
        
    def forward(self, input, target):
        return super().forward(input, torch.argmax(target,dim=-1))
    
class ProcessAndApplyMetric(torch.nn.Module):
    def __init__(self, metric, pred_apply = None, targ_apply = None, metr_apply = None):
        super(ProcessAndApplyMetric, self).__init__()
        
        self.metric = metric
        self.pred_apply = torch.nn.Identity() if pred_apply is None else pred_apply 
        self.targ_apply = torch.nn.Identity() if targ_apply is None else targ_apply 
        self.metr_apply = torch.nn.Identity() if metr_apply is None else metr_apply 
    
    def __getattr__(self, name):
        if name in ('pred_apply', 'targ_apply', 'metr_apply', 'metric'):
            return super().__getattr__(name)
        else :
            return getattr(self.metric, name)
        
    def forward(self, input, target, *args, **kwargs):
        return self.metr_apply(
            self.metric(
                self.pred_apply(input),
                self.targ_apply(target),
                *args, **kwargs
            )
        )