#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import torch

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


logger = logging.getLogger("PREDICTOR")

def keep_best_predictions(predictions: dict, score_threshold=0.5) -> dict:
    """
    Keep the best predictions: the ones for which the 
    score is above the given threshold.

    TODO: there must be a simpler way ...

    Parameters
    ----------
    predictions: dict
        The predictions, as returned by the model
    score_threshold: float
        the score threshold (between 0 and 1)

    Returns
    -------
    dict:
        The predictions to keep

    """
    #print(predictions)
    boxes = predictions["boxes"]
    labels = predictions["labels"]
    scores = predictions["scores"]
    all_predictions = zip(boxes, labels, scores)

    predictions_to_keep = {}
    predictions_to_keep["boxes"] = []
    predictions_to_keep["labels"] = []
    predictions_to_keep["scores"] = []
    for index, (box, label, score) in enumerate(all_predictions):
        if score > score_threshold:
            predictions_to_keep["boxes"].append(box)
            predictions_to_keep["labels"].append(label)
            predictions_to_keep["scores"].append(score)

    predictions_to_keep["boxes"] = torch.vstack(predictions_to_keep["boxes"])
    predictions_to_keep["labels"] = torch.Tensor(predictions_to_keep["labels"])
    predictions_to_keep["scores"] = torch.Tensor(predictions_to_keep["scores"]).double()
    #print(predictions_to_keep)
    return predictions_to_keep

def get_corresponding_bounding_box(box, targets):
    """
    Return the ground truth bounding box.
    The ground truth bounding box is the one with the highest IoU

    Parameters
    ----------
    box: numpy.ndarray
      The detected bounding box
    targets: dict
      dictionary for the annotations

    Returns
    -------
    numpy.ndarray:
      The ground truth bounding box corresponding to the detection
    float:
      The Intersection over Union (IoU) value

    """
    ranked_boxes = {}
    for gt_box in targets["boxes"]:
        iou = compute_iou(np.array(gt_box), box.detach().numpy())
        ranked_boxes[iou] = gt_box 
    index = np.max(list(ranked_boxes.keys()))
    return ranked_boxes[index], index


def get_detection_box_with_highest_iou(gt_box, detections):
    """
    Get the detection having the highest IoU with the
    provided ground truth bounding box

    Parameters
    ----------
    gt_box: list
        the ground truth bounding box
    detections: list of torch.Tensors
        the list contains the tuple (box, score)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, int]:
      The detected bounding box with the highest IoU, its score and its index
    float:
      The Intersection over Union (IoU) value

    """
    ranked_boxes = {}
    for detection_index, (box, score) in enumerate(detections): 
        iou = compute_iou(np.array(gt_box), box.detach().numpy())
        ranked_boxes[iou] = (box, score, detection_index)
    max_iou = np.max(list(ranked_boxes.keys()))
    return ranked_boxes[max_iou], max_iou


def compute_iou(box1, box2):
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

    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


def compute_F1_score(tp, fp, fn):
    """
    Compute the F1 score

    Parameters
    ----------
    tp: int
      Number of True Positives
    fp: int
      Number of False Positives
    fn: int
      Number of False Negatives

    Returns
    -------
    float:
      the F1 score

    """
    denominator = tp + (0.5 * (fp + fn))
    if denominator != 0:
        return float(tp) / denominator
    else:
        return 0


def get_tp_fp_fn(
    img,
    predictions,
    target=None,
    score_threshold=0.5,
    iou_threshold=0.5,
    plot=0,
    show_tn=False,
    savedir=None,
    image_name="image",
    save_faulty=False,
):
    """
    Get the number of true positives, false positives and false negatives.

    Parameters
    ----------
    img: numpy.ndarray
      The image
    predictions: tuple
      The predictions retrieved by the model
    target: dict
      The ground truth
    score_threshold: float
      min score to consider a reliable detection
    iou_threshold: float
      min IoU to be considered as a reliable detection
    plot: int
      Plotting level
    show_tn: bool
      If you want to show true negatives as well
    savedir: str
      Directory where you want to save images
    image_name: str
      Filename of the image currently processed
    save_faulty: bool
     If you want to save only images containing errors

    Returns
    -------
    tuple:
      Number of true positives, false positives and false negatives

    """
    tp = 0
    fp = 0
    fn = 0

    boxes = predictions[0]["boxes"]
    scores = predictions[0]["scores"]

    # sort the scores from lowest to highest (better for plotting)
    zipped = list(zip(boxes, scores))
    sorted_detections = sorted(zipped, key=lambda x: x[1])

    # get the ground truth
    if target is not None:
        gt_boxes = target["boxes"]
        logger.debug("There are {} detections (gt is {})".format(len(boxes), len(gt_boxes)))
    else:
        logger.debug("There are {} detections".format(len(boxes)))

    # go through all ground truth boxes
    index_processed = []
    for gt_box in gt_boxes:

        # get the detection having the biggest overlap with the ground truth
        ((detection_box, detection_score), 
         iou, 
         detection_index
         ) = get_detection_box_with_highest_iou(gt_box, sorted_detections)
        print(detection_box)
        print(detection_score)
        print(iou)
        import sys
        sys.exit()

        # if there's not enough overlap, that's a miss (false negative)
        if iou < iou_threshold:
            fn += 1

        #if iou > iou_threshold and score > score_threshold

    # go through ground_truth boxes
    #for b, s in sorted_detections:

    #    # get the ground truth bbox corresponding to the found box, and the IoU
    #    if target is not None:

    #        gt_box, iou = get_corresponding_bounding_box(b, target)
    #        b = b.detach().numpy()

    #        # true positive [TP]
    #        if ((s >= score_threshold) and (iou >= iou_threshold)):
    #            tp += 1
    #            logger.debug("--------------------")
    #            logger.debug(
    #                "Detection score = {:.2f}, with bounding box {}".format(s, b)
    #            )
    #            logger.debug(
    #                "Got ground truth: bounding box {}".format(gt_box)
    #            )
    #            logger.debug("[TP] with IoU {:.2f}".format(iou))

    #            # plot true positive example in green
    #            if (savedir is not None) or plot:
    #                rect = Rectangle(
    #                    (b[0], b[1]),
    #                    (b[2] - b[0]),
    #                    (b[3] - b[1]),
    #                    ec="g",
    #                    lw=2,
    #                    facecolor="none",
    #                )
    #                ax.add_patch(rect)
    #        
    #        # false positive [FP] (something is detected but there's nothing)
    #        if (s >= score_threshold) and (iou < iou_threshold):
    #            fp += 1
    #            logger.debug("--------------------")
    #            logger.debug(
    #                "Detection score = {:.2f}, with bounding box {}".format(s, b)
    #            )
    #            logger.debug(
    #                "Got ground truth: bounding box {}".format(gt_box)
    #            )
    #            logger.debug(
    #                "[FP] IoU is {:.2f} (too low)".format(iou)
    #            )
    #            
    #            # plot false positive example in red
    #            if (savedir is not None) or plot:
    #                rect = Rectangle(
    #                    (b[0], b[1]),
    #                    (b[2] - b[0]),
    #                    (b[3] - b[1]),
    #                    ec="r",
    #                    lw=2,
    #                    facecolor="none",
    #                )
    #                ax.add_patch(rect)
    #        
    #        # false negative (something is not detected, but should have been)
    #        if (s < score_threshold) and (iou >= iou_threshold):
    #            fn += 1
    #            logger.debug("--------------------")
    #            logger.debug(
    #                "Detection score = {:.2f}, with bounding box {}".format(s, b)
    #            )
    #            logger.debug(
    #                "Got ground truth: bounding box {}".format(gt_box)
    #            )
    #            logger.debug(
    #                "[FN] IoU is {:.2f}, but score is too low".format(iou)
    #            )

    #            # plot false negative example in yellow
    #            if (savedir is not None) or plot:
    #                rect = Rectangle(
    #                    (b[0], b[1]),
    #                    (b[2] - b[0]),
    #                    (b[3] - b[1]),
    #                    ec="y",
    #                    lw=2,
    #                    facecolor="none",
    #                )
    #                ax.add_patch(rect)
    #        
    #        # true negative (nothing detected where nothing should be detected)
    #        # this happens with not confident enough detections (low score -> discarded)
    #        if (s < score_threshold) and (iou < iou_threshold):
    #            logger.debug("--------------------")
    #            logger.debug(
    #                "Detection score = {:.2f}, with bounding box {}".format(s, b)
    #            )
    #            logger.debug(
    #                "Got ground truth: bounding box {}".format(gt_box)
    #            )
    #            logger.debug(
    #                "[TN] There is nothing to detect here (and that's right, score is low) !"
    #            )
    #            
    #            # plot true negative example in limegreen
    #            if ((savedir is not None) or plot) and show_tn:
    #                rect = Rectangle(
    #                    (b[0], b[1]),
    #                    (b[2] - b[0]),
    #                    (b[3] - b[1]),
    #                    ec="limegreen",
    #                    lw=2,
    #                    facecolor="none",
    #                )
    #                ax.add_patch(rect)

    #    # there is no ground truth, so just plot detections
    #    else:
    #        b = b.detach().numpy()
    #        if (savedir is not None) or plot:
    #            rect = Rectangle(
    #                (b[0], b[1]),
    #                (b[2] - b[0]),
    #                (b[3] - b[1]),
    #                ec="cornflowerblue",
    #                lw=2,
    #                facecolor="none",
    #            )
    #            ax.add_patch(rect)

    #if (savedir is not None) or plot:
    #    ax.imshow(display)
    #    ax.set_xticks([])
    #    ax.set_yticks([])

    #if savedir is not None:
    #    plt.savefig(os.path.join(savedir, image_name))
    #if plot:
    #    plt.show()
    #plt.close()

    return tp, fp, fn
