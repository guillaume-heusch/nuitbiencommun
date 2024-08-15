import csv
from pathlib import Path


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
    with open(filename, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            polygons.append(row)
    return polygons
