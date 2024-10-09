.. nuitbiencommun documentation master file, created by
   sphinx-quickstart on Wed Sep 25 16:48:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================
Documentation for nuitbiencommun 
================================

This package contains the code developed to detect numbers
on signboards during the event of "La Nuit du Bien Commun".

In particular, it contains the material needed to:

  1. Extraction of images from video sequences
  2. Annotation of the panels in images
  3. Training a Faster R-CNN model to detect the panels
  4. Applying a pre-trained OCR engine to recognize the numbers in the detected panels

The raw data used throughout this project consists in several video sequences recorded
during different events. They can be found
`here <https://drive.google.com/drive/folders/1bd9vROXaN5U77lois_i7p-lyyqjx4-TT>`_

Users Guide
===========

.. toctree::
   :maxdepth: 2

   extract_frames
   rough_detection_craft
   rough_detection_faster_rcnn
   cvat



