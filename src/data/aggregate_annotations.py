#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

import click
import cv2
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

def read_annotation_file(filename: Path) -> list:
    """
    Reads an annotation file containing polygons
    and returns the list of annotations for polygons
   
    Parameters
    ----------
    filename: Path

    Returns
    -------
    list:
        The list of polygons

    """
    polygons = []
    with open(filename, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            polygons.append(row)
    return polygons

def show_annotations_on_frame(frame_filename, polygons):
    """
    Shows the final annotations on the frame

    Parameters
    ----------
    frame_filename: Path
        The path to the frame image
    ploygons: list
        The list of polygons to display

    """
    try:
        image = cv2.imread(str(frame_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error:
        logging.error("Unable to load image {}".format(image_filename))
   
    
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)
    for p in polygons:
        x = [p[i] for i in range(len(p)) if i % 2 == 0]
        y = [p[i] for i in range(len(p)) if i % 2 == 1]
        xy = np.column_stack((x, y))
        poly = Polygon(
            xy,
            ec="springgreen",
            lw=2,
            facecolor="none",
        )
        ax.add_patch(poly)
    plt.show()




@click.command()
@click.argument("craft_annotation_dir", type=click.Path())
@click.argument("cvat_annotation_dir", type=click.Path(exists=True))
@click.argument("final_annotation_dir", type=click.Path())
@click.argument("frames_dir", type=click.Path())
def process(**kwargs):
    """
    Go through (completed) CVAT annotation files, find
    the corresponding (corrected) CRAFT annotation file
    and aggregate the two: polygons in both file will
    be written in a final annotation file.

    """
    craft_annotation_dir = Path(kwargs["craft_annotation_dir"])
    cvat_annotation_dir = Path(kwargs["cvat_annotation_dir"])
    final_annotation_dir = Path(kwargs["final_annotation_dir"])
    frames_dir = Path(kwargs["frames_dir"])

    final_annotation_dir.mkdir(exist_ok=True, parents=True)
    n_frames_annotated= 0
    n_polygons_total = 0

    for cvat_file in sorted(cvat_annotation_dir.iterdir()):
        
        if cvat_file.is_file() and cvat_file.suffix == ".txt":
            
            craft_poly = []
            craft_file = craft_annotation_dir / cvat_file.name
            if craft_file.is_file():
                craft_poly = read_annotation_file(craft_file)

            cvat_poly = read_annotation_file(cvat_file)
            all_polygons = cvat_poly + craft_poly

        frame_filename = frames_dir / cvat_file.name
        frame_filename = frame_filename.with_suffix(".png")
        if len(all_polygons) > 0:
            n_frames_annotated += 1
            n_polygons_total += len(all_polygons)
            print(f"{len(all_polygons)} annotated panels for {frame_filename.name}")
            show_annotations_on_frame(frame_filename, all_polygons)
        else:
            print(f"No annotations for {frame_filename.name}")

        final_annotation_file = final_annotation_dir / cvat_file.name 
        with open(final_annotation_file, "w") as csvfile:
            writer = csv.writer(csvfile)
            for p in all_polygons:
                writer.writerow(p)
        
    print(f"{n_frames_annotated} frames with annotations")
    print(f"{n_polygons_total} panels annotated")

if __name__ == "__main__":
    process()
