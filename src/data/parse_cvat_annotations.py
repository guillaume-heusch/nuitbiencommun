#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import click


@click.command()
@click.argument("cvat-xml-file")
@click.argument("annotations-dir")
def process(**kwargs):
    """
    Reads the annotation XML file exported from CVAT, and create
    an annotation file for each image. The annotation file contains
    the polygons around panels, defined by the four corner points:
    x1,y1,x2,y2,x3,y3,x4,y4

    .. warning::

        The XML file should have been generated uisng
        CVAT and exported using the CVAT for images 1.1 format.

    """
    xml_file = kwargs["cvat_xml_file"]
    annotations_dir = kwargs["annotations_dir"]
    Path(annotations_dir).mkdir(exist_ok=True, parents=True)

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # get all "image" entries
    for image in root.iter("image"):

        # get the image name
        for k, v in image.attrib.items():
            if k == "name":
                image_filename = v
                print(f"Processing {image_filename}")
                annotation_filename = Path(annotations_dir) / Path(
                    image_filename
                )
                annotation_filename = annotation_filename.with_suffix(".txt")

        # for each image, get all the polygons
        panels_coordinates = []
        n_polygons = 1
        for poly in image.iter("polygon"):
            for k1, v1 in poly.attrib.items():
                if k1 == "points":
                    panels_coordinates.append(v1)
                    points = v1
                    n_polygons += 1

        # turn the polygons from x1,y1;x2,y2;x3,y3;x4,y4
        # to x1,y1,x2,y2,x3,y3,x4,y4 (and convert to int too)
        polygons_to_write = []
        for p in panels_coordinates:
            p = p.replace(";", ",")
            poly = p.split(",")
            polygons_to_write.append([int(float(i)) for i in poly])

        with open(annotation_filename, "w") as csvfile:
            writer = csv.writer(csvfile)
            for p in polygons_to_write:
                writer.writerow(p)


if __name__ == "__main__":
    process()
