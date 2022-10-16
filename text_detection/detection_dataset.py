## score map should be made 
from utils.east_utils import *
from utils.ctpn_data_utils import *
from utils.geo_map_utils import *
from torch.utils.data import Dataset, DataLoader

## DETECTION DATASET ##
class Dataset(Dataset):
    def __init__(self, data_cfg, mode = 'train'):
        super(Dataset, self).__init__()
        self.mode = mode
        self.data_cfg = data_cfg
    
    def __len__(self):
        return 
    
    def __getitem__(self, idx):
        return