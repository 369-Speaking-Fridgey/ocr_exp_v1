import numpy as np
import cv2

#IOU_NEGATIVE=0.3
IOU_NEGATIVE=0.3
IOU_POSITIVE=0.5
IOU_SELECT=0.7


RPN_POSITIVE_NUM=128
RPN_NEGATIVE_NUM=RPN_POSITIVE_NUM * 3
RPN_TOTAL_NUM=RPN_POSITIVE_NUM + RPN_NEGATIVE_NUM
# IMAGE_MEAN=[123.68, 116.779, 103.939]
IMAGE_SIZE = [1024, 2048]
IMAGE_STD = [0.20037157, 0.18366718, 0.19631825]
IMAGE_MEAN = [0.90890862, 0.91631571, 0.90724233]
OHEM=False
# OHEM=False
'''
anchor generation
문제: 먼저 base_anchor이 초기 위치 지점에 대해 생성된 anchor은 단계별로 feature map의 각 지점에 anchor을 생성한 후 anchor의 shape는 (10, H*W, 4)가 된다.
여기서 처음에 anchor rehspa를 (10*H*W, 4)로 직접 구성했는데, 이렇게 했더니 훈련 중에 수렴을 하지 않았다.
원인: 직접 (10, H*W, 4) -> (10*H*W, 4), anchor의 배열 순서는 feature map의 점순이 아닌 다른 총 10개의 anchor의 모양으로 배열이 된다.
'''
def gen_anchor( featuresize, scale, 
                # heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283], 
                heights = [11, 15, 22, 32,45, 65, 93, 133, 190, 273],
                widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]):
    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)

    #base center(x,,y) -> (x1, y1, x2, y2)
    base_anchor = np.array([0, 0, 15, 15])
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))
    
    anchor = list()
    for i in range(base_anchor.shape[0]):
        anchor_x1 = shift[:,0] + base_anchor[i][0]
        anchor_y1 = shift[:,1] + base_anchor[i][1]
        anchor_x2 = shift[:,2] + base_anchor[i][2]
        anchor_y2 = shift[:,3] + base_anchor[i][3]
        anchor.append(np.dstack((anchor_x1, anchor_y1, anchor_x2, anchor_y2)))

    return np.squeeze(np.array(anchor)).transpose((1,0,2)).reshape((-1, 4))

'''
Intersection of Union을 정답 bounding box와 예측한 bounding box를 사용해서 계산한다.
iou = inter_area/(bb_area + anchor_area - inter_area)
'''
def compute_iou(anchors, bbox):
    ious = np.zeros((len(anchors), len(bbox)), dtype=np.float32)
    anchor_area = (anchors[:,2] - anchors[:,0])*(anchors[:,3] - anchors[:,1])
    for num, _bbox in enumerate(bbox):
        bb = np.tile(_bbox,(len(anchors), 1))
        bb_area = (bb[:,2] - bb[:,0])*(bb[:,3] - bb[:,1])
        inter_h = np.maximum(np.minimum(bb[:,3], anchors[:,3]) - np.maximum(bb[:,1], anchors[:,1]), 0)
        inter_w = np.maximum(np.minimum(bb[:,2], anchors[:,2]) - np.maximum(bb[:,0], anchors[:,0]), 0)
        inter_area = inter_h*inter_w
        ious[:,num] = inter_area/(bb_area + anchor_area - inter_area)

    return ious

''' 논문에서 6페이지의 내용을 사용
anchor와 gtboxes의 수직 방향 차이 매개 변수 regression_factor(Vc, Vh) 계산
1、(x1, y1, x2, y2) -> (ctr_x, ctr_y, w, h)
2、 Vc = (gt_y - anchor_y) / anchor_h
Vh = np.log(gt_h / anchor_h)
'''
def bbox_transfrom(anchors, gtboxes):
    gt_y = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    gt_h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0

    anchor_y = (anchors[:, 1] + anchors[:, 3]) * 0.5
    anchor_h = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (gt_y - anchor_y) / anchor_h
    Vh = np.log(gt_h / anchor_h)

    return np.vstack((Vc, Vh)).transpose()

'''
anchor와 차분 파라미터 regression_factor(Vc, Vh), 계산 대상 박스 bbox
'''
def transform_bbox(anchor, regression_factor):
    anchor_y = (anchor[:, 1] + anchor[:, 3]) * 0.5
    anchor_x = (anchor[:, 0] + anchor[:, 2]) * 0.5
    anchor_h = anchor[:, 3] - anchor[:, 1] + 1

    Vc = regression_factor[0, :, 0]
    Vh = regression_factor[0, :, 1]

    bbox_y = Vc * anchor_h + anchor_y
    bbox_h = np.exp(Vh) * anchor_h

    x1 = anchor_x - 16 * 0.5
    y1 = bbox_y - bbox_h * 0.5
    x2 = anchor_x + 16 * 0.5
    y2 = bbox_y + bbox_h * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox

'''
bbox를 원본 이미지의 크기에 맞게 자르기
    x1 >= 0
    y1 >= 0
    x2 < im_shape[1] (=W)
    y2 < im_shape[0] (=H)
'''
def clip_bbox(bbox, im_shape):
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox

'''
최소 크기보다 작은 bounding box는 버려야 한다.
'''
def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep

'''
RPN module
1, anchor 생성
2, anchor와 참값 상자 gtboxes의 iou를 계산합니다.
3. iou에 따라 각 anchor에 라벨을 할당하며, 0은 음의 샘플, 1은 양의 샘플, -1은 포기 항목입니다.
    (1) 각 참값 상자 bbox에 대해 iou와 가장 큰 anchor를 찾아 플러스 샘플로 설정
    (2) 각 anchor에 대해 각 bbox에서 구한 iou 중 가장 큰 값 max_overlap을 기록합니다.
    (3) max_overlap이 설정역치보다 큰 anchor에 대해서는 플러스 샘플로 설정하고, 설정역치보다 작으면 마이너스 샘플로 설정
4, 경계를 벗어난 앵커박스를 필터링하여 라벨을 -1로 설정
5. 설정 수량을 초과하지 않는 양의 샘플과 음의 샘플을 선택한다.
6. anchor가 max_overlap을 얻을 때의 gtbbox 간의 진값 차이량(Vc, Vh)을 구한다.
'''
def cal_rpn(imgsize, featuresize, scale, gtboxes):
    base_anchor = gen_anchor(featuresize, scale)
    overlaps = compute_iou(base_anchor, gtboxes)

    gt_argmax_overlaps = overlaps.argmax(axis=0) ## condition(2) 가장 IoU overlap의 값이 큰 Ground Truth Box를 사용
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)
    labels[gt_argmax_overlaps] = 1
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1
    labels[anchor_max_overlaps < IOU_NEGATIVE] = 0

    ## (3) anchor들을 만들고 그 anchor이 원본 image size를 넘어가게 되면 -1로 표시가 되게 한다.
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgsize[1]) |
        (base_anchor[:, 3] >= imgsize[0])
    )[0] 
    labels[outside_anchor] = -1

    fg_index = np.where(labels == 1)[0]
    ## (4) 설정 수량을 초과 하는 경우, 즉 예측하는 것이 가능한 RPN의 최대 개수를 초과하면 포기
    if (len(fg_index) > RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1
    if not OHEM:
        bg_index = np.where(labels == 0)[0]
        num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
        if (len(bg_index) > num_bg):
            labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])

    return [labels, bbox_targets]


'''
Non Maximum Suppression
'''
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


'''
그림 기반 텍스트 줄 구성 알고리즘
하위 그래프 연결 규칙, 그래프에서 쌍을 이루는 텍스트 상자에 따라 텍스트 행을 생성합니다.
1. graph의 행과 열을 돌아다니며, 열이 모두 false이고, 행이 완전히 false인 행과 열을 찾고, 인덱스 번호는 index입니다.
2. graph의 index행에서 true인 항목의 인덱스 번호를 찾아 서브그래프에 추가하고 인덱스 번호를 index에 반복한다.
3. graph의 index행 전체가 false가 될 때까지 2단계를 반복한다.
4, 순서 1, 2, 3을 반복하여 graph를 완주한다
텍스트 행 list [텍스트 상자 색인]을 되돌립니다.
'''
class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)

        return sub_graphs

'''
配置参数
MAX_HORIZONTAL_GAP: 文本行内，文本框最大水平距离
MIN_V_OVERLAPS: 文本框最小垂直iou
MIN_SIZE_SIM: 文本框尺寸最小相似度
'''
class TextLineCfg:
    SCALE = 600
    MAX_SCALE = 1200
    TEXT_PROPOSALS_WIDTH = 16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO = 0.5
    LINE_MIN_SCORE = 0.9 # 0.7
    TEXT_PROPOSALS_MIN_SCORE = 0.9 # 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    MAX_HORIZONTAL_GAP = 60 # 60
    MIN_V_OVERLAPS = 0.7 # 0.6
    MIN_SIZE_SIM = 0.7 # 0.6


class TextProposalGraphBuilder:
    '''
    쌍을 이루는 텍스트 상자 만들기
    '''
    def get_successions(self, index):
        '''
        트래버스[x0, x0+MAX_HORIZONTAL_GAP]
        지정한 인덱스 번호의 후속 텍스트 상자 가져오기
        '''
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results

        return results

    def get_precursors(self, index):
        '''
        Get the previous (right-side) text proposals belonging to the same group of the current text proposals
        Args
        index: id of the current vertex
        Returns
        List of integer contains the index of suitable text proposals
        '''
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results

        return results

    def is_succession_node(self, index, succession_index):
        '''
        쌍을 이루는 텍스트 상자인지 여부
        '''
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True

        return False

    def meet_v_iou(self, index1, index2):
        '''
        두 텍스트 상자가 수직 방향의 iou 조건을 충족하는지 여부를 판단합니다.
        overlaps_v: 텍스트 상자의 수직 방향의 iou 계산. iou_v = inv_y/min(h1, h2)
        size_similarity: 텍스트 상자의 수직 방향 높이 치수의 유사도. sim = min(h1, h2)/max(h1, h2)
        '''
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
                size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        '''
        텍스트 상자에 따라 텍스트 상자 쌍 만들기
        self.heights: 모든 텍스트 상자의 높이
        self.boxes_table: 왼쪽 위의 x1 좌표에 따라 텍스트 상자를 그룹화합니다.
        graph: bool 형식의 [n, n] 배열로, 두 텍스트 상자가 쌍으로 되어 있는지 여부를 나타내며, n은 텍스트 상자의 개수입니다.
        (1) 현재 텍스트 상자 Bi의 후속 텍스트 상자 가져오기
        (2) 후속 텍스트 상자에서 가장 높은 점수를 받은 것을 Bj로 기록한다.
        (3) Bj의 이전 텍스트 상자 가져오기
        (4) 만약 Bj의 선행 텍스트 상자에서 가장 높은 점수를 받은 것이 Bi라면, <Bi, Bj>는 텍스트 상자 쌍을 구성합니다.
        '''
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                graph[index, succession_index] = True

        return Graph(graph)


class TextProposalConnectorOriented:
    """
    text box여러개를 연결하고 bbox를 구성한다.
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        '''
        텍스트 상자를 연결하고 텍스트 줄에 따라 그룹화합니다
        '''
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        '''
        일원 선형 함수는 X, Y에 적합하고, y1, y2의 좌표값을 반환한다.
        '''
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        '''
        텍스트 상자에 따라 텍스트 줄 만들기
        1. 텍스트 상자를 텍스트 행 그룹으로 나누고, 각 텍스트 행 그룹에는 규칙에 맞는 텍스트 상자를 포함합니다.
        2. 각 텍스트 행 그룹을 처리하여 큰 텍스트 행으로 묶습니다.
            (1) 텍스트 줄 그룹의 모든 텍스트 상자 가져오기 text_line_boxes
            (2) 각 그룹 내의 각 텍스트 상자의 중심 좌표 (X, Y), 최소, 최대 너비 좌표 (x0, x1)를 찾습니다.
            (3) 모든 중심점 직선 z1에 맞추기
            (4) offset을 텍스트 상자의 너비의 절반으로 설정합니다
            (5) 그룹 내 모든 텍스트 상자의 왼쪽 상단 모서리에 직선을 맞추고 (x0+offset, x1-offset) 의 극을 오른쪽 y 좌표 (lt_y, rt_y) 로 되돌립니다.
            (6) 그룹 내 모든 텍스트 상자의 왼쪽 하단 모서리에 직선을 맞추고 (x0+offset, x1-offset) 의 극을 오른쪽 y 좌표 (lb_y, rb_y) 로 되돌립니다.
            (7) 텍스트 행 그룹 내의 모든 상자의 점수의 평균을 해당 텍스트 행의 점수로 취합니다
            (8) 텍스트 줄 기본 데이터 생성
        3. 큰 텍스트 상자 생성
        '''
        # (1) Group text proposals
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size) 
        
        # (2) Initialize the list of text bounding scores and boxes
        text_lines = np.zeros((len(tp_groups), 8), np.float32)
        new_text_lines = np.zeros((len(tp_groups), 4), np.float32)
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]

            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2
            x0 = np.min(text_line_boxes[:, 0]) ## xmin
            x1 = np.max(text_line_boxes[:, 2]) ## xmax

            z1 = np.polyfit(X, Y, 1) 
            
            ## (3) Find the vertical coordinates of the text lines
            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5 
            
            ## find the vertical coordinates of the text lines
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)
            
            ##  (4) The average of the scores is the score of the text line
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))
            
            new_text_lines[index,0] = x0
            new_text_lines[index, 1] = min(lt_y, rt_y)
            new_text_lines[index, 2] = x1
            new_text_lines[index, 3] = max(lb_y, rb_y)

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)  
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y) 
            text_lines[index, 4] = score 
            text_lines[index, 5] = z1[0]  
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  
            text_lines[index, 7] = height + 2.5
        
        new_text_lines = clip_bbox(new_text_lines, im_size)
        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  
            x2 = line[2]
            y2 = line[5] * line[2] + b1  
            x3 = line[0]
            y3 = line[5] * line[0] + b2  
            x4 = line[2]
            y4 = line[5] * line[2] + b2  
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  

            fTmp0 = y3 - y1 
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs, new_text_lines

if __name__=='__main__':
    anchor = gen_anchor((10, 15), 16)
