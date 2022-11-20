from .east.east import EAST
from .ctpn.ctpn import CTPN
# from .textfuse.textfuse import TEXTFUSE
from loguru import logger
import torch

model_registry = {
    'EAST': EAST,
    'CTPN': CTPN,
    # 'TEXTFUSE': TEXTFUSE,
}

def load_model(model_name, model_cfg):
    model = model_registry[model_name.upper()](**model_cfg['params'])
    model.cuda()
    if model_cfg['pretrained_model'] != '':
        temp_weight = model.state_dict()
        load_weight = torch.load(model_cfg['pretrained_model'])
        if len(list(load_weight.keys())) == 2:
            load_weight = load_weight['model_state_dict']
        
        load_weight = {key: value for (key, value) in load_weight.items() if key in temp_weight and 
                            value.shape == temp_weight[key].shape}
        temp_weight.update(load_weight)
        for key, value in load_weight.items():
            logger.info(key)
        model.load_state_dict(temp_weight)
    
    return model

