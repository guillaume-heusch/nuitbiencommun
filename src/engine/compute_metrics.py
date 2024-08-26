import numpy as np
import networkx as nx

class MetricsComputer():
    """
    Class responsible to compute various metrics
    related to object detection.

    Count True Positives, False Positives, False Negatives
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp

    Attributes
    ---------

    """
    def __init__(self):
        """
        """
        self.iou_threshold = 0.5

    def run_on_batch(self, predictions: list, ground_truth: list):
        """
        Runs the computation of metrics on a batch of images

        Parameters
        ----------
        predictions: list
            The list of predictions for this batch
        ground_truth: dict
            The list of ground truth 

        """
        for pred, gt in zip(predictions, ground_truth):
            self.run_on_image(pred, gt)


    def run_on_image(self, predictions: dict, ground_truth: dict):
        """
        Runs the computation of metrics on one image

        Parameters
        ----------
        predictions: dict
            The predictions (boxes, labels, scores)
        ground_truth: dict
            The ground truth (boxes, lables, scores)

         
        """
        matches = self._match_boxes(predictions["boxes"], ground_truth["boxes"])
        tp = len(matches)
        fp = len(predictions["boxes"]) - tp
        fn = len(ground_truth["boxes"]) - tp

        print(tp, fp, fn)

    def _match_boxes(self, pred_boxes: list, gt_boxes: list) -> list:
        """
        Match a prediction box to a ground truth box using
        the Maximum Weighted Bipartite Matching algorithm.
    
        Parameters
        ----------
        pred_boxes: list
            list of predictions bounding boxes
        gt_boxes: list
            list of ground truth bounding boxes
        
        Returns
        -------
        list:
            The matches

        """
        pred_boxes = pred_boxes.detach().numpy()
        gt_boxes = np.array(gt_boxes)

        # Create a bipartite graph
        G = nx.Graph()
        G.add_nodes_from(range(len(pred_boxes)), bipartite=0)
        G.add_nodes_from(range(len(gt_boxes)), bipartite=1)

        # Add edges with weights (IoU values)
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou = self._compute_iou(pred_box, gt_box)
                if iou >= self.iou_threshold:
                    G.add_edge(i, len(pred_boxes) + j, weight=iou)

        # Find the maximum weighted bipartite matching
        matching = nx.max_weight_matching(G, maxcardinality=True)
        print(matching)

        # Convert the matching to a list of pairs
        matches = []
        for u, v in matching:
            if u < len(pred_boxes) and v >= len(pred_boxes):
                matches.append((pred_boxes[u], gt_boxes[v - len(pred_boxes)]))

        return matches

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Computes the intersection over union of 2 bounding boxes.

        Parameters
        ----------
        bbox1: numpy.ndarray
          First bounding box
        bbox2: numpy.ndarray
          Second bounding box

        Returns
        -------
        float:
          The intersection over union

        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth rectangles
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = float(box1Area + box2Area - interArea)
        iou = interArea / union if union != 0 else 0
        return iou
