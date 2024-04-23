#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
import sys

import click
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

logging.basicConfig(level=logging.INFO)


def keep_detection():
    """ """
    while True:
        response = (
            input("Do you want to keep the current detection ? [y/n]: ")
            .strip()
            .lower()
        )
        if response == "y":
            return True
        elif response == "n":
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'")


def load_image(image_filename: str) -> np.ndarray:
    """
    Loads an image

    Parameters
    ----------
    image_filename: str

    Returns
    -------
    np.ndarray
        the image

    """
    try:
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error:
        logging.error("Unable to load image {}".format(image_filename))
        sys.exit()
    return image


def get_detections_filename(image_filename: str, detections_dir: str) -> str:
    """
    Build and retrieve the filename of the
    file containing the detections, based on the image filename.

    Parameters
    ----------
    image_filename: str
        The image filename
    detections_dir: str
        Name of the directory containing the detections, as txt files

    Returns
    -------
    str
        the annotation filename

    """
    _, ext = os.path.splitext(image_filename)
    detections_filename = "res_" + os.path.basename(image_filename)
    detections_filename = detections_filename.replace(ext, ".txt")
    detections_filename = os.path.join(detections_dir, detections_filename)
    return detections_filename


def load_detections(detections_file: str) -> list:
    """
    Read the detections file to get the coordinates of
    the detections in a list.

    Parameters
    ----------
    detections_file: str
        The filename of the txt file containing the detections

    Returns
    -------
    list
        A list of detections, given by the coordinates of the 4 corners

    """
    try:
        detections = []
        with open(detections_file) as csvfile:
            reader = csv.reader(csvfile)
            for row in enumerate(reader):
                detections.append([int(x) for x in row[1]])
    except FileNotFoundError:
        logging.warning("No detections !")
        return None
    return detections


def show_and_ask_for_detections(image: np.ndarray, detections: list) -> list:
    """
    Show detections 1 by 1 and ask the user if the just seen
    detection should be kept

    Parameters
    ----------
    image: np.ndarray
        The image
    detections: list
        The list of detections

    Returns
    -------
    list:
        The list of detections to keep

    """
    detections_to_keep = []
    for line in detections:

        fig, ax = plt.subplots(1, figsize=(16, 9))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image)

        x = [line[i] for i in range(len(line)) if i % 2 == 0]
        y = [line[i] for i in range(len(line)) if i % 2 == 1]
        xy = np.column_stack((x, y))
        poly = Polygon(
            xy,
            ec="springgreen",
            lw=2,
            facecolor="none",
        )
        ax.add_patch(poly)
        plt.show()

        if keep_detection():
            detections_to_keep.append(line)
        else:
            continue

    return detections_to_keep


def write_valid_detections(detections_filename, detections):
    """ """
    with open(detections_filename, "w") as csvfile:
        writer = csv.writer(csvfile)
        for d in detections:
            writer.writerow(d)


@click.command()
@click.argument("images-dir")
@click.argument("detections-dir")
@click.argument("corrected-annotations-dir")
def process(**kwargs):
    """
    Show detections, and prompt the user to keep it (or not).

    """
    images_dir = kwargs["images_dir"]
    detections_dir = kwargs["detections_dir"]
    corrected_annotations_dir = kwargs["corrected_annotations_dir"]
    if os.path.isdir(corrected_annotations_dir):
        logging.warning(" Corrected annotation dir already exists !")
        sys.exit()
    else:
        os.makedirs(corrected_annotations_dir)

    images = os.listdir(images_dir)
    images.sort()

    for i, img_file in enumerate(images):
        logging.info(
            "Processing {} ({}/{})".format(img_file, i + 1, len(images))
        )

        detections_file = get_detections_filename(img_file, detections_dir)
        detections = load_detections(detections_file)
        if detections is None:
            continue

        img_file = os.path.join(images_dir, img_file)
        image = load_image(img_file)
        detections_to_keep = show_and_ask_for_detections(image, detections)
        corrected_annotations_file = detections_file.replace(
            detections_dir, corrected_annotations_dir
        )
        write_valid_detections(corrected_annotations_file, detections_to_keep)


if __name__ == "__main__":
    process()
