#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import sys
from pathlib import Path

import click
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

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


def get_image_filename(
    detections_file: Path, images_dir: Path
) -> Path:
    """
    Build and retrieve the filename of the
    image file based on the detections filename.

    Parameters
    ----------
    detections_file: str
        The detections filename
    images_dir: str
        Name of the directory containing the images

    Returns
    -------
    Path
        the image file

    """
    image_file = Path(detections_file.name)
    image_file = image_file.with_suffix(".jpg")
    image_file = images_dir / image_file
    return image_file


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
        A list of detections, given by the coordinates of
        the bounding boxes [xmin, ymin, xmax, ymax]

    """
    try:
        detections = []
        with open(str(detections_file)) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                detections.append([int(x) for x in row[1:]])
    except FileNotFoundError:
        logging.warning("No detections !")
        return None
    return detections


def show_and_ask_for_detections(image: np.ndarray, detections: list, color: str = "red") -> list:
    """
    Show detections 1 by 1 and ask the user if the just seen
    detection should be kept

    Parameters
    ----------
    image: np.ndarray
        The image
    detections: list
        The list of detections
    color: str
        The color to display the bounding boxes
        (should be a valid matplotlib color)

    Returns
    -------
    list:
        The list of detections to keep

    """
    detections_to_keep = []
    for line in detections:

        fig, ax = plt.subplots(1, figsize=(16, 9))
        fig.suptitle(
            f"Check the detection ({color} rectangle), hit 'q' and go back to the terminal"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image)

        rect = Rectangle(
                (line[0], line[1]),
                line[2] - line[0],
                line[3] - line[1],
                edgecolor=color,
                facecolor="none",
                linewidth=2,
              )

        ax.add_patch(rect)
        plt.show()

        if keep_detection():
            detections_to_keep.append(line)
        else:
            continue

    return detections_to_keep


def write_valid_detections(detections_filename: str, detections: list):
    """
    Write the detections to keep in a csv file.

    The format of the detection is a bounding box
    defined by xmin, ymin, xmax, ymax

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


def write_image_with_detections(
    image: np.ndarray, detections: list, img_file: Path
):
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
        cv2.rectangle(image_with_detections, (d[0], d[1]), (d[2], d[3]), color=(255, 0, 0), thickness=3)
    plt.figure(figsize=(16, 9))
    plt.imshow(image_with_detections)
    plt.title("Final detections (hit 'q' to close)")
    plt.show()
    image_with_detections = cv2.cvtColor(
        image_with_detections, cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(str(img_file), image_with_detections)


def get_corrected_annotation_file(
    corrected_annotations_dir: Path, image_file: Path
) -> Path:
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
def process(**kwargs):
    """
    Show detections, and prompt the user to keep it (or not).

    """
    images_dir = Path(kwargs["images_dir"])
    detections_dir = Path(kwargs["detections_dir"])
    corrected_annotations_dir = Path(kwargs["corrected_annotations_dir"])
    annotated_images_dir = Path(kwargs["annotated_images_dir"])

    corrected_annotations_dir.mkdir(parents=True, exist_ok=True)
    annotated_images_dir.mkdir(parents=True, exist_ok=True)
    
    detections_files = [d for d in detections_dir.iterdir()]
    detections_files.sort()

    for i, detections_file in enumerate(detections_files):
        logging.info(
            "Processing {} ({}/{})".format(detections_file.name, i + 1, len(detections_files))
        )

        img_file = get_image_filename(detections_file, images_dir)
        corrected_annotations_file = get_corrected_annotation_file(
            corrected_annotations_dir, img_file
        )
        image_with_detections_file = annotated_images_dir / Path(img_file.name)


        if (
            corrected_annotations_file.is_file()
            and image_with_detections_file.is_file()
        ):
            logging.warning(
                f"annotation and image files already exists for {img_file.name}, skipping !"
            )
            continue

        detections = load_detections(detections_file)
        
        if detections is None:
            logging.warning(f"No detections for {img_file.name}, skipping !")
            continue

        image = load_image(str(img_file))
        detections_to_keep = show_and_ask_for_detections(image, detections)
        write_valid_detections(corrected_annotations_file, detections_to_keep)
        write_image_with_detections(
            image, detections_to_keep, image_with_detections_file
        )


if __name__ == "__main__":
    process()
