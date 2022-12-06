import numpy as np

from typing import Tuple, List
import numpy as np

def clip_bboxes(bboxes: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Clip the bounding boxes within the image boundary.
    Args:
        bboxes (numpy.ndarray): The set of bounding boxes.
        image_size (int, tuple): The image's size.
    Returns:
        THe bounding boxes that are within the image boundaries.
    """

    height, width = image_size

    zero = 0.0
    w_diff = width - 1.0
    h_diff = height - 1.0

    # x1 >= 0 and x2 < width
    bboxes[:, 0::2] = np.maximum(np.minimum(bboxes[:, 0::2], w_diff), zero)
    # y1 >= 0 and y2 < height
    bboxes[:, 1::2] = np.maximum(np.minimum(bboxes[:, 1::2], h_diff), zero)

    return bboxes
class Graph:
    def __init__(self, graph: object):
        """
        Object represents the graph containing the connected text proposals.
        
        Args:
            graph (object): The graph object.
            
        """
        self.graph: object = graph

    def sub_graphs_connected(self) -> List[List[int]]:
        """
        Refine the original graph having num_proposals x num_proposals vertices
        into a list of group of connected text proposals
        
        Returns:
            A sub-graph, i.e., a list of indexes of connected text proposals.
            
        """
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class TextProposalGraphBuilder(object):

    def __init__(self):
        """
        Build text proposals into a graph.
        
        Args:
            configs (dict): The config path_to_file.
            
        """
        self.configs: dict = None
        self.MIN_V_OVERLAPS=0.9
        self.MIN_SIZE_SIM=0.9
        self.MAX_HORI_GAP=20

    def get_successions(self, index: int) -> List[int]:
        """
        Find text proposals belonging to same group of the current text proposal.
        
        Args:
            index (int): The id of current vertex.

        Returns:
            List of integer contains the index of suitable text proposals.
            
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1,
                          min(int(box[0]) + self.MAX_HORI_GAP+ 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index: int) -> List[int]:
        """
        Get the previous (right-side) text proposals belonging to the same group of the current text proposals.
        
        Args:
            index (int): The id of current vertex.

        Returns:
            List of integer contains the index of suitable text proposals.
            
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - self.MAX_HORI_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index: int, succession_index: int) -> bool:
        """
        Check if a provided text proposal is connected to the current text proposal.
        
        Args:
            index (int): The ID of current vertex.
            succession_index (int): The ID of the next vertex.

        Returns:
            A boolean indication whether a given text proposal is connected to current text proposal.
            
        """

        # Get all right-side text proposals belonging to same group
        precursors = self.get_precursors(succession_index)

        # If text proposal having higher or equal score than right-side text proposals, return True.
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True

        # Otherwise False.
        return False

    def meet_v_iou(self, first_index: int, second_index: int) -> bool:
        """
        Check if two text proposals belong into same group.
        Fist, we check the vertical overlap and then check the size similarity.
        
        Args:
            first_index (int): the index of the first text proposal
            second_index (int): the index of the second text proposal

        Returns:

        """

        def vertical_overlap(first_index: int, second_index: int) -> float:
            h1 = self.heights[first_index]
            h2 = self.heights[second_index]
            y0 = max(self.text_proposals[second_index][1], self.text_proposals[first_index][1])
            y1 = min(self.text_proposals[second_index][3], self.text_proposals[first_index][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(first_index: int, second_index: int) -> float:
            h1 = self.heights[first_index]
            h2 = self.heights[second_index]
            return min(h1, h2) / max(h1, h2)

        return vertical_overlap(first_index, second_index) >= self.MIN_V_OVERLAPS and \
               size_similarity(first_index, second_index) >= self.MIN_SIZE_SIM

    def build_graph(self, text_proposals: np.ndarray, scores: np.ndarray, im_size: Tuple[int, int]) -> Graph:
        """
        Build graph of text proposals. This graph has num_proposals x num_proposals vertices, and vertices is connected
        if corresponding text proposals is also connected (belong in to a same text boxes).
        
        Args:
            text_proposals (np.ndarray): A Numpy array containing the coordinates of each text proposal. Shape: [N, 4]
            scores (np.ndarray): A Numpy array that contains the predicted confidence of each text proposal. Shape: [N,]
            im_size (Tuple, int): The image's size.

        Returns:
            A graph
        """
        self.text_proposals: np.ndarray = text_proposals
        self.scores: np.ndarray = scores
        self.im_size: Tuple[int, int] = im_size
        self.heights: np.ndarray = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table: List[List[int]] = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors)
                # if multiple successions(precursors) have equal scores.
                graph[index, succession_index] = True

        G = Graph(graph)

        return G
def fit_y(X: np.ndarray, Y: np.ndarray, x1: np.ndarray, x2: np.ndarray):
    """
    Interpolate the vertical coordinates based on data and 2 given horizontal coordinates.

    Args:
        X (numpy array): A numpy array contains the horizontal coordinates.
        Y (numpy array): A numpy array contains the vertical coordinates.
        x1 (numpy array): The horizontal coordinate of point 1.
        x2 (numpy array): The horizontal coordinate of point 2.

    Returns:
        An interpolation of the vertical coordinates.

    """
    # if X only include one point, the function will get line2Match y=Y[0]
    if np.sum(X == X[0]) == len(X):
        return Y[0], Y[0]
    p = np.poly1d(np.polyfit(X, Y, 1))
    return p(x1), p(x2)


class TextProposalConnector(object):
    def __init__(self,):
        """
        Connect text proposals into text bouding boxes.
        
        Args:
            configs (dict): The configuration file.
            
        """

        self.graph_builder: object = TextProposalGraphBuilder()

    def group_text_proposals(self,
                             text_proposals: np.ndarray,
                             scores: np.ndarray,
                             im_size: Tuple[int, int]) -> List[List[int]]:
        """
        Group text proposals into groups. Each group contains the text proposals belong into the same line of text.
        
        Args:
            text_proposals (numpy array): A Numpy array that contains the coordinates of each text proposal.
            scores (numpy array): A Numpy array that contains the predicted confidence of each text proposal.
            im_size (int, tuple): The image's size.

        Returns:
            A group of the text proposals.
            
        """

        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)

        return graph.sub_graphs_connected()

    def get_text_lines(self,
                       text_proposals: np.ndarray,
                       scores: np.ndarray,
                       im_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine all text proposals into bounding boxes.
        
        Args:
            text_proposals (numpy array): A Numpy array that contains the coodinates of each text proposal.
            scores (numpy array): A Numpy array that contains the predicted confidence of each text proposal.
            im_size (int, tuple): The image's size.

        Returns:
            The bounding boxes and scores for each line.
            
        """
        # Group text proposals
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)

        # Initialize the list of text bounding boxes and scores.
        text_lines = np.zeros((len(tp_groups), 4), dtype=np.float32)
        average_scores = []

        # Now, connect the text proposals in each group
        for index, tp_indices in enumerate(tp_groups):
            # Get the coordinates, offset, and scores of each proposal in group.
            text_line_boxes = text_proposals[list(tp_indices)]
            # Get the predicted top left and bottom right x-coordinates of the text lines.
            xmin = np.min(text_line_boxes[:, 0])
            xmax = np.max(text_line_boxes[:, 2])

            # Find vertical coordinates of text lines
            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) / 2.
            lt_y, rt_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], xmin + offset, xmax - offset)
            lb_y, rb_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], xmin + offset, xmax - offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line.
            average_scores.append(scores[list(tp_indices)].sum() / float(len(tp_indices)))

            # Appending the bounding boxes coordinates and scores.
            text_lines[index, 0] = xmin
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = xmax
            text_lines[index, 3] = max(lb_y, rb_y)

        # Keep bounding boxes inside the image size.
        text_lines = clip_bboxes(text_lines, im_size)

        average_scores = np.array(average_scores)

        return text_lines, average_scores