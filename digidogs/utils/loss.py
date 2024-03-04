import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.core import Tensor

class LocationmapLoss(nn.Module):
    def __init__(self):
        super(LocationmapLoss, self).__init__()
    
    def forward(self, pred_Loc, gt_Loc, gt_H):
        diff = pred_Loc - gt_Loc
        prod = torch.mul(gt_H, diff)
        loss = torch.norm(prod, p=2)
        return loss

class MaskedMSELoss(nn.Module): 
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    
    def forward(self, pred_h, gt_h, mask): 
        x = pred_h.reshape(pred_h.size(0), pred_h.size(1), -1) 
        y = gt_h.reshape(gt_h.size(0), gt_h.size(1), -1) 
        m = mask.reshape(mask.size(0), mask.size(1), 1)

        diff2 = (x - y) ** 2
        diff2 = diff2 * m 
        diff2 = diff2.mean(dim=2)
        diff2 = torch.divide(diff2.sum(dim=1), torch.sum(m!=0, dim=1).reshape(-1)) 
        diff2[diff2!=diff2] = 0 
        diff2 = diff2.mean()
        return diff2

def _kl_div_2d(p: Tensor, q: Tensor) -> Tensor:
    # D_KL(P || Q)
    batch, chans, height, width = p.shape
    unsummed_kl = F.kl_div(
        q.reshape(batch * chans, height * width).log(), p.reshape(batch * chans, height * width), reduction='none'
    )
    kl_values = unsummed_kl.sum(-1).view(batch, chans)
    return kl_values


def _js_div_2d(p: Tensor, q: Tensor) -> Tensor:
    # JSD(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * _kl_div_2d(p, m) + 0.5 * _kl_div_2d(q, m)


def _reduce_loss(losses: Tensor, reduction: str) -> Tensor:
    if reduction == 'none':
        return losses
    return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)


def js_div_loss_2d_masked(input: Tensor, target: Tensor,mask: Tensor = None, reduction: str = 'mean') -> Tensor:
    r"""Calculate the Jensen-Shannon divergence loss between heatmaps.

    Args:
        input: the input tensor with shape :math:`(B, N, H, W)`.
        target: the target tensor with shape :math:`(B, N, H, W)`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Examples:
        >>> input = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = js_div_loss_2d(input, input)
        >>> loss.item()
        0.0
    """
    if mask is not None:
        if len(mask.shape) == 3:
            mask = mask.reshape(mask.shape[0], -1)
            print(mask.shape)
        loss = _js_div_2d(target, input)
        loss_mask = loss*mask
        loss_mask = torch.sum(loss_mask,dim=1)/ torch.sum(mask!=0, dim=1) 
        loss_mask[loss_mask!=loss_mask] = 0
    return _reduce_loss(loss_mask, reduction)

if __name__ == "__main__":

    x = torch.randint(10,(3,2,4,4)).float()
    y = torch.randint(10,(3,2,4,4)).float()
    v = torch.randint(2,(3,2,1))
    l = js_div_loss_2d(x,y,mask=v, reduction='sum')
    print(l.item())


#class MultiLoss(nn.Module):
#    """
#    preds is a list of predictions 
#    targets is a list of targets
#    """
#    def __init__(self, num_task = 2):
#        self.num_task = num_task
#        self.log_vars = nn.Parameter(torch.zeros((num_task)))
#
#    def forward(self, preds, targets):
#        loss0 = 
