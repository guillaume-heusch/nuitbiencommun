#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import xml.etree.ElementTree as ET


@click.command()
@click.argument("xml-file")
def process(**kwargs):
    """
    """
    xml_file = kwargs["xml_file"]

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for image in root.iter("image"):
        
        for k, v in image.attrib.items():
            print(k, v)

        for p in image.iter("polygon"):
            print(p)
            for k1, v1 in p.attrib.items():
                print(k1, v1)



if __name__ == "__main__":
    process()
