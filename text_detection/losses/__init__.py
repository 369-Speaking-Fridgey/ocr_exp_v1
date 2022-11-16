from .balanced_cross_entropy_loss import BalanceCrossEntropyLoss
from .dice_loss import *
from .iou_loss import *
from .l1_loss import MaskL1Loss, BalanceL1Loss
from .east_loss import dice_loss, geo_loss, EASTLoss
from .ctpn_loss import CTPNLoss

loss_registry = {
    "EASTLOSS": EASTLoss,
    "CTPNLOSS": CTPNLoss,
    "BALANCEDCE": BalanceCrossEntropyLoss,
    "BALNCEDL1": BalanceL1Loss,
    "MASKL1": MaskL1Loss

}

def load_loss(train_cfg):
    loss_info = train_cfg['criterion']
    losses, lamdas = [], []
    loss_fn, lamda = loss_info['loss'], loss_info['lamda']
    losses.append(loss_registry[loss_fn.upper()]())
    lamdas.append(lamda)
    
    return losses, lamdas

def calculate_loss(losses, lamdas, *args):
    loss = 0.0
    for fn, lamda in zip(losses, lamdas):
        loss += fn(args) * lamda
    return loss