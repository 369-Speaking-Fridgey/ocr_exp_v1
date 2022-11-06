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
        load_weight = [{key: value} for (key, value) in load_weight.values() if key in temp_weight and 
                            value.size == temp_weight[key].size]
        model.load_state_dict(load_weight)
    
    return model


if __name__ == "__main__":
    model = load_model('east', model_cfg = {
        'params': {'branch_name': 'vgg19_bn','geo_type': 'rbox', 'output_scope' : 512, 
                    'pretrained_bbone': True, 'freeze_bbone': True},
        'pretrained_model' : ''
    })
    logger.info(model)