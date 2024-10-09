============================
Adding annotations with CVAT
============================

Since the model used are not perfect, they may also miss some panels. 
So, to get annotations for those missed panels, you should use a free 
web-based tool, `CVAT <cvat.ai>`_.

After having created an account, you should log-in and create a new project


Annotations using CRAFT detections
----------------------------------

To start from CVAT annotations - and if you want to be more generic - you should
annotate images using **polygons**. 

To do so, create a project named 
*panel_polygon* for instance, add a label (i.e. *panel*) and select its type
to be "Polygon" in the drop-down menu. Now create a new task, choose a name
(*panel_polygon_annotation* for instance) and select the right project.
Finally upload the images located in the  ``{images_with_detection}`` 
folder you created before.


input panel as label, and the type should be "Polygon". Doing this allows you
to delineate panels using a polygon and eentually train and evaluate a model
detecting these kind of shapes

- For annotations, you must use the "polygon" with 4 points: it's better than the bounding box, 
  since it allows non-rectangular regions. Also, **make sure to name your annotation "panel"**: 
  the subsequent script assumes this name when looking for polygons in the XML file !

Annotations using Faster-RCNN detections
----------------------------------------

To start from Faster-RCNN annotations, you should
input panel as label, and the type should be "Rectangle".

Now, I let you go through all the images of the sequence, and draw nice polygons around not-detected panels ;) 
Note that you don't need to be super precise (as you may have seen, the models are not that precise either).

Once you annotated all the missed panels in all the frames, you should got to "Menu / Export job dataset". 
Choose the "CVAT for images 1.1" export format and click OK. This will save an XML file with quite a lot of information.


**A few important things though:**

- The images you should upload are the ones in the ``{images_with_detection}`` folder created before. 
  Here you already have the detections from CRAFT outlined (in green), 
  so you don't need to annotate the already detected panels.

