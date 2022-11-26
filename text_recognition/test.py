import os, sys
BASE = os.path.dirname(os.path.abspath(__file__))
from loguru import logger
import torch
sys.path.append(BASE)
sys.path.append(
    os.path.join(BASE, 'rec_model', 'hangulnet')
)

from hangulnet import HangulNet

if __name__ == "__main__":
    net = HangulNet()
    x = torch.rand((2, 3, 32, 128))
    out = net(x) 
    """ Output Shape
    (Batch Size, Max Sequence Length, Grapheme Class Num)
    """
    logger.info(out.shape)