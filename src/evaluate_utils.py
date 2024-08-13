#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

logger = logging.getLogger("chardetrec")


def get_corresponding_bounding_box(label, box, targets):
    """
    Return the ground truth bounding box given the label

    Parameters
    ----------
    label: int
      The detected label
    box: numpy.ndarray
      The detected bounding box
    targets: dict
      dictionary for the annotations

    Returns
    -------
    numpy.ndarray:
      The bounding box corresponding to the detection
    int:
      The ground truth label
    float:
      The Intersection over Union value

    """
    candidates = {
        "boxes": [],
        "labels": [],
    }  # candidates for gt boxes and labels (there may be more than one per label)
    for gt_label, gt_box in zip(
        targets["labels"].tolist(), targets["boxes"].tolist()
    ):
        if gt_label == label:
            candidates["boxes"].append(gt_box)
            candidates["labels"].append(gt_label)

    # if there is only one ground truth candidate, return it directly
    if len(candidates["boxes"]) == 1:
        iou = compute_iou(
            np.array(candidates["boxes"][0]), box.detach().numpy()
        )
        return candidates["boxes"][0], candidates["labels"][0], iou
    # if there is no ground truth candidate, this is a false positive
    elif len(candidates["boxes"]) == 0:
        return None, None, 0
    # otherwise, get the gt_box with highest IoU
    else:
        ranked_boxes = {}
        for b, l in zip(candidates["boxes"], candidates["labels"]):
            iou = compute_iou(np.array(b), box.detach().numpy())
            ranked_boxes[iou] = (b, l)
        index = np.max(list(ranked_boxes.keys()))
        return ranked_boxes[index][0], ranked_boxes[index][1], index


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
    ret = {}
    for c in allowed_chars:
        ret[c] = [0, 0, 0]  # TP, FP, FN

    boxes = predictions[0]["boxes"]
    labels = predictions[0]["labels"]
    scores = predictions[0]["scores"]

    # sort the scores from lowest to highest (better for plotting)
    zipped = list(zip(boxes, labels, scores))
    sorted_detections = sorted(zipped, key=lambda x: x[2])

    # get the ground truth
    if target is not None:
        gt_boxes = target["boxes"].tolist()
        gt_labels = target["labels"].tolist()
        logger.debug(
            "There are {} detections (gt is {})".format(
                len(boxes), len(gt_boxes)
            )
        )
    else:
        logger.debug("There are {} detections".format(len(boxes)))

    if savedir is not None:
        if not os.path.isdir(savedir):
            os.makedirs(savedir)

    if (savedir is not None) or plot:
        display = np.moveaxis(img.numpy(), 0, -1)
        fig, ax = plt.subplots(1, figsize=(16, 9))
        fig.suptitle(image_name)

        # display the ground truth
        if target is not None:
            for gt_box, label in zip(gt_boxes, gt_labels):
                rect = Rectangle(
                    (gt_box[0], gt_box[1]),
                    (gt_box[2] - gt_box[0]),
                    (gt_box[3] - gt_box[1]),
                    ec="k",
                    lw=2,
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    gt_box[0] + (0.5 * (gt_box[2] - gt_box[0])),
                    gt_box[1] + (0.2 * (gt_box[3] - gt_box[1])),
                    label_to_char[label],
                    c="k",
                )

    box_color_tp = "cornflowerblue"
    for b, l, s in sorted_detections:

        # get the ground truth bbox corresponding to the found box, and the IoU
        if target is not None:

            gt_box, gt_label, iou = get_corresponding_bounding_box(
                l.item(), b, target
            )
            b = b.detach().numpy()

            # true positive
            if (
                (s >= score_threshold)
                and (l.item() == l)
                and (iou >= iou_threshold)
            ):
                ret[label_to_char[l.item()]][0] += 1
                logger.debug("--------------------")
                logger.debug(
                    "Detection is {} ({:.2f}), with bounding box {}".format(
                        label_to_char[l.item()], s, b
                    )
                )
                logger.debug(
                    "Got ground truth: bounding box {} with label {}".format(
                        gt_box, label_to_char[gt_label]
                    )
                )
                logger.debug("[TP] with IoU {:.2f}".format(iou))
                if (savedir is not None) or plot:
                    rect = Rectangle(
                        (b[0], b[1]),
                        (b[2] - b[0]),
                        (b[3] - b[1]),
                        ec="g",
                        lw=2,
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        b[0],
                        b[1],
                        label_to_char[l.item()]
                        + " (IoU: {:.2f}, s: {:.2f})".format(iou, s),
                        c="g",
                    )
            # false positive
            if (s >= score_threshold) and (
                (l.item() != l) or (iou < iou_threshold)
            ):
                ret[label_to_char[l.item()]][1] += 1
                logger.debug("--------------------")
                logger.debug(
                    "Detection is {} ({:.2f}), with bounding box {}".format(
                        label_to_char[l.item()], s, b
                    )
                )
                if gt_box is not None and gt_label is not None:
                    logger.debug(
                        "Got ground truth: bounding box {} with label {}".format(
                            gt_box, label_to_char[gt_label]
                        )
                    )
                    logger.debug(
                        "[FP] IoU is {:.2f}, label is {}".format(
                            iou, label_to_char[l.item()]
                        )
                    )
                else:
                    logger.debug("[FP] There is nothing to detect here !")
                if (savedir is not None) or plot:
                    rect = Rectangle(
                        (b[0], b[1]),
                        (b[2] - b[0]),
                        (b[3] - b[1]),
                        ec="r",
                        lw=2,
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        b[0],
                        b[1],
                        label_to_char[l.item()]
                        + " (IoU: {:.2f}, s: {:.2f})".format(iou, s),
                        c="r",
                    )
            # false negative
            if (
                (s < score_threshold)
                and (l.item() == l)
                and (iou >= iou_threshold)
            ):
                ret[label_to_char[l.item()]][2] += 1
                logger.debug("--------------------")
                logger.debug(
                    "Detection is {} ({:.2f}), with bounding box {}".format(
                        label_to_char[l.item()], s, b
                    )
                )
                logger.debug(
                    "Got ground truth: bounding box {} with label {}".format(
                        gt_box, label_to_char[gt_label]
                    )
                )
                logger.debug(
                    "[FN] IoU is {:.2f}, label is {}, but score is too low".format(
                        iou, label_to_char[l.item()]
                    )
                )
                if (savedir is not None) or plot:
                    rect = Rectangle(
                        (b[0], b[1]),
                        (b[2] - b[0]),
                        (b[3] - b[1]),
                        ec="r",
                        lw=2,
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        b[0],
                        b[1],
                        label_to_char[l.item()]
                        + " (IoU: {:.2f}, s: {:.2f})".format(iou, s),
                        c="r",
                    )
            # true negative
            if (s < score_threshold) and (
                (l.item != l) or (iou < iou_threshold)
            ):
                logger.debug("--------------------")
                logger.debug(
                    "Detection is {} ({:.2f}), with bounding box {}".format(
                        label_to_char[l.item()], s, b
                    )
                )
                if gt_box is not None and gt_label is not None:
                    logger.debug(
                        "Got ground truth: bounding box {} with label {}".format(
                            gt_box, label_to_char[gt_label]
                        )
                    )
                    logger.debug(
                        "[TN] There is no {} to detect here (and that's right, score is low) !".format(
                            label_to_char[l.item()]
                        )
                    )
                else:
                    logger.debug(
                        "[TN] There is nothing to detect here (and that's right, score is low) !"
                    )
                if ((savedir is not None) or plot) and show_tn:
                    rect = Rectangle(
                        (b[0], b[1]),
                        (b[2] - b[0]),
                        (b[3] - b[1]),
                        ec="b",
                        lw=2,
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        b[0],
                        b[1],
                        label_to_char[l.item()] + " ({:.2f})".format(s),
                        c="b",
                    )

        # there is no ground truth, so just plot detections and scores
        else:
            if (savedir is not None) or plot:
                rect = Rectangle(
                    (b[0], b[1]),
                    (b[2] - b[0]),
                    (b[3] - b[1]),
                    ec=box_color_tp,
                    lw=2,
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    b[0],
                    b[1],
                    label_to_char[l.item()] + " ({:.2f})".format(s),
                    c=box_color_tp,
                )

    if (savedir is not None) or plot:
        ax.imshow(display)
        ax.set_xticks([])
        ax.set_yticks([])

    # check if this image contains a faulty example:
    faulty = False
    for k, v in ret.items():
        for i in range(1, 3):
            if v[i] > 0:
                faulty = True

    if savedir is not None:
        if faulty and save_faulty:
            plt.savefig(os.path.join(savedir, image_name))
        else:
            plt.savefig(os.path.join(savedir, image_name))
    if plot:
        plt.show()
    plt.close()

    return ret
