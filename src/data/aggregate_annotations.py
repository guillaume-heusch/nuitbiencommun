#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

import click


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


@click.command()
@click.argument("craft_annotation_dir", type=click.Path(exists=True))
@click.argument("cvat_annotation_dir", type=click.Path(exists=True))
@click.argument("final_annotation_dir", type=click.Path())
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
    
    final_annotation_dir.mkdir(exist_ok=True, parents=True)

    for cvat_file in sorted(cvat_annotation_dir.iterdir()):
        
        if cvat_file.is_file() and cvat_file.suffix == ".txt":
            craft_file = craft_annotation_dir / cvat_file.name
            cvat_poly = read_annotation_file(cvat_file)
            craft_poly = read_annotation_file(craft_file)
            all_polygons = cvat_poly + craft_poly

        final_annotation_file = final_annotation_dir / cvat_file.name 
        with open(final_annotation_file, "w") as csvfile:
            writer = csv.writer(csvfile)
            for p in all_polygons:
                writer.writerow(p)
        

if __name__ == "__main__":
    process()
