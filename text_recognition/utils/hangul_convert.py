GRAPEME_DICT={
    "first": ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅍ','ㅎ'],
    "middle": ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ'],
    "last": [' ','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ', 'ㄼ','ㄽ','ㄾ', 'ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
}
KEYS=GRAPHEME_DICT.keys()
VALUES=GRAPHEME_DICT.values()
""" CAUTION
- Hangul Net을 사용하지 않는다면 어쩔 수 없이 [UNK], 즉 OOV(Out of Vocab) 문제가 발생하게 된다.
- Grapheme dict에서 first와 last에서는 겹치는 부분이 분명히 존재한다. 따라서 이 부분에 대해서는 예측을 position encoding 정보를 통해서 알아서 학습이 잘 되어야 한다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from jamo import h2j, j2hcj

from .jamo_utils import join_jamos

class HangulLabelConverter(object):
  """ Hangul Label Converter
  1. 실제 정답 한글 문자에서 Hangul Net 학습을 위한 라벨로 변경
  2. Hangul Net의 출력값을 예측된 merged grapheme으로 바꾸어 준다.
  3. 
  """
  def __init__(self,
               character=' ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ', 
               max_length=75):
    # character (str): string of the possible characters (includes hangul and numbers)
    # [s]: end of the sequence token
    # [GO] : head of the token of the decoder
    tokens = ['[GO]', '[s]']
    character_list = list(character)
    self.character = tokens + character_list
    self.char_encoder_dict = {}
    self.char_decoder_dict = {}
    self.max_length = max_length
    for i, char in enumerate(self.character):
      self.char_encoder_dict[char] = i
      self.char_decoder_dict[i] = char

  
  def encode(self, text, one_hot=True):
    """ 한글+숫자 문자열 -> 정수 인덱스 라벨
    - text: 입력 문자열
    - one_hot: one hot encoding을 할지 말지
    """
    def onehot(label, depth, device=None):
      """ 
      Args:
        label: shape (n1, n2, ..., )
        depth: a scalar
      Returns:
        onehot: (n1, n2, ..., depth)
      """
      if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, device=device)
      onehot = torch.zeros(label.size() + torch.Size([depth]), device=device)
      onehot = onehot.scatter_(-1, label.unsqueeze(-1), 1)

      return onehot
    
    new_text = ''
    label = ''
    ## (1) 입력된 한글 + 숫자 문자열의 자음과 모음을 분리한다.
    jamo_str = j2hcj(h2j(text)) # 자모 분할을 한 결과
    for idx, jamo in enumerate(jamo_str.strip(' ')):
      if jamo != ' ':
        new_text += jamo
        label += self.char_encoder_dict[jamo] # 자/모음 -> 정수 인덱스
    ## (2) char_dict를 사용해서 라벨을 만들어 준다.
    length = torch.tensor(len(new_text) + 1).to(dtype=torch.long) ## end token을 위해서 전체 길이+1을 length로 사용한다.
    label = torch.tensor(label).to(dtype = torch.long)
    ## (3) Cross Entropy Loss 학습을 위해서 one hot vector으로 바꾸어 준다.
    if one_hot:
      label = onehot(label, len(self.character))

    return label


  def decode(self, predicted):
    ## (1) Softmax 처리를 해서 0-1사이의, 합이 1인 logit으로 구성하게 한다.
    scores = F.softmax(predicted, dim = 2)
    pred_text, pred_scores, pred_lengths = [], [], []
    for score in scores:
      ## (2) Argmax
      score_ = score.argmax(dim=1)
      text = ''
      for idx, s in enumerate(score_):
        if s < 52: ## 한글인 경우에
          text += 
        else: ## 영어나 숫자인 경우에
          text += self.char_decoder_dict[s]
        key = KEYS[idx % 3]

      ## (3) 자음 모음이 분리되어 연결된 string을 글자로 merge 
      text = join_jamos(text)
      pred_text.append(text)
      pred_scores.append(score.max(dim=1)[0])
      pred_lengths.append(min(len(text) + 1, self.max_length)) ## ['e'] token을 위해서 1을 더해줌

    return pred_text, pred_scores, pred_lengths

def merge_grapheme_single(predicted, character):
  """ Args
  predicted: (B, Sequence Length, Class Number) (Torch Tensor) -> single기준이기 때문에 하나의 이미지만 입력으로 받아야 한다.
  output: (B, None) -> 각각의 batch의 sample 마다 어떤 grapheme에 속하는지 예측할 수 있어야 한다.
  """
  converter = HangulLabelConverter(character)
  ## (1) Get the prediction score of the output from the hangul net
  predicted_p = F.softmax(predicted, dim = -1)
  B, S, C = predicted_p.shape ## (Batch Size, Max Sequence Length, Number of Class)
  predicted = predicted.detach().cpu()
  pred_class = torch.argmax(predicted_p, dim=-1, keepdims=False).squeeze(0)
  ## (2) Change to numpy array
  numpy_predicted = pred_class.numpy()
  answer = converter.decode(numpy_predicted)

  return answer