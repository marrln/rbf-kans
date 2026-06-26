import torch
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
    
class CombinedLoss(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.loss = torch.nn.ModuleList(args)
        self.w = torch.nn.Parameter(torch.zeros([len(self.loss)], requires_grad=True))

    def forward(self, *args, **kwargs):
        return torch.exp(-self.w) @ torch.stack([
            loss(*args, **kwargs)
                for loss in self.loss
        ]) + self.w.sum()