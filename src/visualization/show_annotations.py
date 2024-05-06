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

from pathlib import Path

logging.basicConfig(level=logging.INFO)


def keep_detection():
    """
    Ask the user if the last shown detection should be kept

    """
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


def get_detections_filename(image_filename: Path, detections_dir: Path) -> Path:
    """
    Build and retrieve the filename of the
    txt file containing the CRAFT detections,
    based on the image filename.

    The CRAFT txt file containing the annotations is built by
    prefixing "res_" to the original image name, and changing
    its extension to txt. It is put on a folder called result
    where the CRAFT script has been launched.

    Parameters
    ----------
    image_filename: str
        The image filename
    detections_dir: str
        Name of the directory containing the detections, as txt files

    Returns
    -------
    Path
        the file containing the CRAFT detections

    """
    ext = image_filename.suffix
    detections_file = Path("res_" + image_filename.name)
    detections_file = detections_file.with_suffix('.txt')
    detections_file = detections_dir / detections_file
    return detections_file


def load_detections(detections_file: Path) -> list:
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
        with open(str(detections_file)) as csvfile:
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


def write_valid_detections(detections_filename: str, detections: list):
    """ 
    Write the detections to keep in a csv file.

    The format of the detection is a polygon with 4 corners and
    the polygon is defined by x1, y1, x2, y2, x3, y3, x4, y4

    Parameters
    ----------
    detections_filename: str
        The file where to write the detections
    detections: list
        The list of detections

    """
    with open(detections_filename, "w") as csvfile:
        writer = csv.writer(csvfile)
        for d in detections:
            writer.writerow(d)


def write_image_with_detections(image: np.ndarray, detections: list, img_file: Path):
    """
    Write the image with the provided detections as polygons

    Parameters
    ----------
    image: np.ndarray
        The image
    detections: list
        The detections
    img_file: str
        The filename to write the image to

    """
    image_with_detections = image.copy()
    for d in detections:
        xs = [d[i] for i in range(len(d)) if i % 2 == 0]
        ys = [d[i] for i in range(len(d)) if i % 2 == 1]
        assert len(xs) == len(ys)
        for i in range(-1, (len(xs) - 1), 1):
            cv2.line(image_with_detections, (xs[i], ys[i]), (xs[i+1], ys[i+1]), (0, 255, 0), 5)

    plt.imshow(image_with_detections)
    plt.show()
    image_with_detections = cv2.cvtColor(image_with_detections, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_file), image_with_detections)


def get_corrected_annotation_file(corrected_annotations_dir: Path, image_file: Path) -> Path:
    """
    Get the path for the new, corrected annotation file

    Parameters
    ----------
    corrected_annotations_dir: Path
        The dir to store the new annotation files
    image_file: Path
        The corresponding image path

    Returns
    -------
    Path:
        The path of the corrected annotation file

    """
    tmp = image_file.stem
    annotation_file = Path(tmp).with_suffix(".txt")
    return corrected_annotations_dir / annotation_file


@click.command()
@click.argument("images-dir")
@click.argument("detections-dir")
@click.argument("corrected-annotations-dir")
@click.argument("annotated-images-dir")
@click.option("-s", "--step", default=10, type=int, help="the step between frames")
def process(**kwargs):
    """
    Show detections, and prompt the user to keep it (or not).

    """
    images_dir = Path(kwargs["images_dir"])
    detections_dir = Path(kwargs["detections_dir"])
    corrected_annotations_dir = Path(kwargs["corrected_annotations_dir"])
    annotated_images_dir = Path(kwargs["annotated_images_dir"])
    step = kwargs["step"]

    corrected_annotations_dir.mkdir(parents=True, exist_ok=True)
    annotated_images_dir.mkdir(parents=True, exist_ok=True)

    images = [i for i in images_dir.iterdir()]
    images.sort()
    images = images[0::step]

    for i, img_file in enumerate(images):
        logging.info(
            "Processing {} ({}/{})".format(img_file.name, i + 1, len(images))
        )

        detections_file = get_detections_filename(img_file, detections_dir)
        corrected_annotations_file = get_corrected_annotation_file(corrected_annotations_dir, img_file)
        image_with_detections_file = annotated_images_dir / Path(img_file.name) 

        if corrected_annotations_file.is_file() and image_with_detections_file.is_file():
            logging.warning(f"annotation and image files already exists for {img_file.name}, skipping !")
            continue

        detections = load_detections(detections_file)
        if detections is None:
            logging.warning(f"No detections for {img_file.name}, skipping !")
            continue

        image = load_image(str(img_file))
        detections_to_keep = show_and_ask_for_detections(image, detections)
        write_valid_detections(corrected_annotations_file, detections_to_keep)
        write_image_with_detections(image, detections_to_keep, image_with_detections_file)


if __name__ == "__main__":
    process()
