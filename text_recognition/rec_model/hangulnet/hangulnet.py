import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from encoder import ResTransformer
from decoder import AttentionalDecoder


""" Hangul Net
1. ResModel (ResNet45)
2. Transformer Encoder
3. Position Attention based Decoder
4. Linear Classifier (Generates the class of the hangul graphemes)
"""
class HangulNet(nn.Module):
  def __init__(self, 
               max_seq_length=75,
               embedding_dim=512,
               class_n=52,
               ):
    super(HangulNet, self).__init__()
    #self.resnet = resnet45()
    self.transformer_encoder = ResTransformer()
    self.attention_decoder = AttentionalDecoder()
    self.cls = nn.Linear(embedding_dim, class_n)

  
  def forward(self, x):
    #feature = self.resnet(x)
    #logger.info(feature.shape)
    encoder_out = self.transformer_encoder(x)
    att_vec, att_score = self.attention_decoder(encoder_out)
   
    pred = self.cls(att_vec)

    return pred
