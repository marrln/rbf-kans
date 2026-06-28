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
        
# New metric for CIFAR‑100 superclass top‑1 accuracy
class SuperclassAccuracy(torchmetrics.Metric):
    """
    Computes top‑1 accuracy after mapping fine‑grained predictions and
    labels to their corresponding superclasses.

    Args:
        fine_to_coarse_map: list/array of length num_fine_classes mapping
            each fine class index -> coarse class index.
        num_superclasses: number of coarse classes (default 20 for CIFAR‑100).
    """
    is_differentiable = False
    higher_is_better = True

    def __init__(self, fine_to_coarse_map, num_superclasses=20, **kwargs):
        super().__init__(**kwargs)
        # Register the mapping as a buffer (not a parameter, but a fixed tensor)
        self.register_buffer('map', torch.tensor(fine_to_coarse_map, dtype=torch.int64))
        self.num_superclasses = num_superclasses
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds: (N, C_fine) probabilities/logits -> take argmax
        fine_preds = preds.argmax(dim=-1)
        # Map both to coarse indices
        coarse_preds = self.map[fine_preds]
        coarse_target = self.map[target.to(torch.int64)]
        self.correct += (coarse_preds == coarse_target).sum()
        self.total += coarse_target.numel()

    def compute(self):
        return self.correct.float() / self.total