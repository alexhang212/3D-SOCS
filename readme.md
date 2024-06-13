# Peering into the world of wild passerines with 3D-SOCS: synchronized video capture for posture estimation


**Michael Chimento**, **Alex Hoi Hang Chan**, Lucy M. Aplin, and Fumihiro Kano.
**Bold denotes co-first authorship**.
![Banner](./media/3DSOCS_Banner.jpeg)

## Abstract

> Collection of large behavioral data-sets on wild animals in natural habitats is vital to answering a range of questions in ecology and evolution. Modern sensor technologies, including GPS and passive transponder tags, have enhanced the scale and quality of behavioral data. Moreover, recent developments in machine learning and computer vision, combined with inexpensive and customizable microcomputers, have unlocked a new frontier of fine-scale measurements rivaling what is possible in controlled laboratory conditions. Here, we leverage these advancements to develop a 3D Synchronized Outdoor Camera System (3D-SOCS): an inexpensive, mobile and automated method for collecting fine-scale data on wild animals using sub-millisecond synchronized video frames from multiple Raspberry Pi controlled cameras. To test this system, we placed 3D-SOCS at a wild bird feeder with a stimulus presentation device. From this, machine learning and computer vision methods were applied to estimate 2D and 3D keypoints, object detection and 3D trajectory tracking for multiple individuals of different species. Accuracy tests demonstrate that we can estimate 3D postures of birds within a 3mm tolerance, enabling fine-scale behavioural quantification. We illustrate the research potential of 3D-SOCS by characterizing the visual field configuration of wild great tits (*Parus major*), a model species in behavioral ecology. We find that birds have optic axes of approximately $\pm60\degree$ azimuth and $-5\degree$ elevation and exhibit individual differences in lateralization. We also show that by using the convex hull of birds to estimate body weight, 3D-SOCS can be used for non-invasive population monitoring. In summary, 3D-SOCS is a first-of-its-kind camera system for wild research, presenting exciting new potential to measure fine-scaled morphology and behaviour in wild birds.

## About
This repository contains code/instructions for running our markerless tracking system 3DSOCS yourself, from the manuscript "Peering into the world of wild passerines with 3D-SOCS: synchronized video capture for posture estimation". To use our system, you must first capture frame-synchronized videos of birds (great tits and blue tits, as tested in our manuscript). This can be done using a network of inexpensive RaspberryPi Compute Module 4s (minimum of 2 cameras) and our custom software. Once you have video data, you then send it through our pipeline to extract the 3D postures. The repository is thus split into two folders: one for video data collection (developed by Michael Chimento), and one for the 3D tracking pipeline (developed by Alex Chan). Each folder contains its own readme with further details. We also supply a parts list with prices/working links (as of June 2024).

## Contents
Directory  | Description
------------- | -------------
3D_SOCSraspi | Python code and instructions for running 3DSOCS and collecting video data using RaspberryPi CM4, shell scripts for administration of system.
3DTracking | Python code and instructions for 3D posture tracking pipeline from video data.
equipment_list.ods | List of hardware used in the manuscript.

## Sample Video


<figure class="video_container">
  <iframe src="media/bird_3B00186CDA_trial_32_time_2023-11-05 10_30_28.117871_VisualField_sample.mp4" frameborder="0" allowfullscreen="true"> 
</iframe>
</figure>


## Citation
```

```
## Contact
If you have any questions or problems with the code, feel free to raise a github issue or contact us!

- **Michael Chimento**: mchimento[at]ab.mpg.de
- **Alex Chan**: hoi-hang.chan[at]uni-konstanz.de

## Acknowledgement
This work was supported by the Max Planck Society and the Centre for the Advanced Study of Collective Behaviour, funded by the Deutsche Forschungsgemeinschaft (DFG) under Germany’s Excellence Strategy (EXC 2117-422037984). LMA and MC were also partly funded by the Swiss State Secretariat for Education, Research and Innovation (SERI) under contract number MB22.00056.
