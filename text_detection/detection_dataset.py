## score map should be made 
from utils.east_utils import *
from utils.ctpn_utils import ctpn_data_utils as ctpn_utils
from utils.east_utils import east_utils as east_utils
from torch.utils.data import Dataset, DataLoader

## DETECTION DATASET ##
class Dataset(Dataset):
    def __init__(self, data_cfg, mode = 'train'):
        super(Dataset, self).__init__()
        self.mode = mode
        self.data_cfg = data_cfg
        ## 데이터의 형태가 다 동일할수는 없지만 기본적으로 원하는 형태로 어떤 데이터던 바꿔줄 수 있어야 한다.
    def __len__(self):
        return 
    
    def __getitem__(self, idx):
        return