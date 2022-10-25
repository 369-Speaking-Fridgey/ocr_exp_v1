from balanced_cross_entropy_loss import BalancedCrossEntropyLoss
from dice_loss import *
from iou_loss import *
from l1_loss import MaskL1Loss, BalanceL1Loss
from east_loss import dice_loss, geo_loss, EastLoss

loss_registry = {
    "EASTLOSS": EastLoss,
    "CTPNLOSS": CtpnLoss,
    
}

def load_loss(train_cfg):
    loss_info = train_cfg['criterion']
    losses, lamdas = [], []
    for info in loss_info:
        loss_fn, lamda = info
        losses.append(loss_registry[loss_fn.upper()]())
        lamdas.append(lamda)
    
    return losses, lamdas

def calculate_loss(losses, lamdas, *args):
    loss = 0.0
    for fn, lamda in zip(losses, lamdas):
        loss += fn(args) * lamda
    return loss