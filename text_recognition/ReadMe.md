1. https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.pdf

2. https://arxiv.org/pdf/1904.01906.pdf

3. `Transformation` -> `Feature Extraction` -> `Sequence Modeling` -> `Prediction` 

4. `AI HUB`에서 다운받은 한국어 글자체 이미지를 사용해서 학습을 시키고자 한다.
    - 물론 `All-In-One`의 방법으로 이미지를 `text detection` 모델에 넣어주고 이 출력 값을 바탕으로 잘라서 `text recognition`을 학습 시킬 수 도 있으나 그렇게 하면 text detector의 정확도가 보장되지 않은 상황에서는 학습이 불가능하다.
    - 최대한으로 한국어와 숫자만을 사용해서 학습을 시키고자 한다.
    - 숫자가 없다면 이부분은 분명히 더 고려해 보아야 하지만, 예를 들어서 앞서 `text detection`의 학습에 사용이 되었던 데이터셋을 사용해도 좋을 것이다.

#### `HangulNet Architecture`
1. `Transformer Encoder`
    - CNN layer + transformer-unit is used to learn the context between graphemes that define a character.
2. `Attentional Decoder`
    - The decoder learns to attend feature corresponding to each grapheme in the given rich feature map, which is the output of the transfomer (=Encoder)
    - Each grapheme exists in certain geometric prior.
    - The positional relationship among each set of [first-middle-last] graphemes on the feature map by using positional encoding.
    - The length of the total sequence is L, and K is the reconstructed feature map with U-net structure.
    
3. `Grapheme-based Predictor`
    - Eacg grapheme is classified with a linear classifier, where the number of class is the number of graphemes in the Hangul. 
        - Cross Entropy Loss is used for the grapheme classification
    - Finally, the classified grapheme is merged in a form of character.