# IM-BaIP-IBVR

Final iteration of Ian Michel, Bachelor IP files:
1) Dataset Creation scripts: Coversion scripts from recorded ROSbags to Raw Images, and Kitti-style positional data. Grounding-DINO Juypter Notebook script that can imported and run in Google Colab, for 2D bounding box style label creation. Unifiying python script that parses images and assocaite YOLO style labels from post-processing in Roboflow (expect .rf.xxx. hash) to final .txt Kitti labels.
2) NexusROS: src folder that the ROS workspace can be built from, including packages for IBVR control, and data-collection.
3) JetsonROS: src folder that the ROS workspace can be built from, including packages for IBVR pecetion.
4) Matlab scripts: Relevant for plotting system behavioure, 1-Agent Formation Control, 2-Agent Formation Control, and path recording.
5) Harbrok Scripts: SLURM job scripts for training the used YOLO models (v8 and v11), and 3D bounding box regression network.
