import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dtr.sequence_modeling import BidirectionalLSTM
from dtr.prediction import Attention
from dtr.feature_extraction import ResNet_FeatureExtractor
from dtr.transformation import TPS_SpatialTransformerNetwork

class Model(nn.Module):
    def __init__(self, class_n, imgH=32, imgW=128, 
                    input_channel=3, output_channel=512,
                    hidden_size=256, num_fiducial=20):
        super(Model, self).__init__()
        self.stages = {
            'Trans': 'TPS', 'Feat' : 'ResNet',
            'Seq': 'BiLSTM', 'Pred': 'Attn'
        }
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=num_fiducial, I_size=(imgH, imgW), 
            I_r_size=(imgH, imgW), I_channel_num=input_channel)
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  
        self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(output_channel, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size
        if self.stages['Pred'] == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output,hidden_size,class_n)
        else:
            self.Prediction = nn.Linear(self.SequenceModeling_output, class_n)
 

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=25)

        return prediction