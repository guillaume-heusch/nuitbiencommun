Extract frames from video sequences
===================================

To extract frames from video sequences, I use ffmpeg.

You can find ffmpeg `here <https://www.ffmpeg.org/download.html>`_

Here is the command you should use to extract frames from video sequences:

  .. code-block:: shell

     ffmpeg -i {video_sequence_file} {folder_with_frames}/{sequence_name}-frame%04d.jpg

  .. warning::

     The ``{folder_with_frames}`` should exist before running the command. 

This command will process the given ``video_sequence_file`` and 
output frames in the folder ``folder_with_frames``. 
Each frame will have the name ``{sequence_name}-frame``, followed by a 4-digit number 
representing the frame number. 

Here's a concrete example with the video sequence ``IMG_1622.MOV``:

  .. code-block:: shell
  
     ffmpeg -i IMG_1622.MOV 1622/1622-frame%04d.png

This will create a folder named `1622`, which will contain 
image files named `1622-frame0001.png`, `1622-frame0002.png` 
and so on until `1622-frame0713.png`.

