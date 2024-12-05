"""Visualize 3D tracking"""
"""
VisualizeData.py

This script is designed to visualize 3D tracking data for bird movements using multiple camera angles. It utilizes a dictionary to map body parts to specific colors for visualization purposes.

Modules:
    ReprojectVisualFields: Contains the Visualize3D_VS function for visualizing 3D data.

Functions:
    Visualize3D_VS: Visualizes the 3D tracking data.

Variables:
    ColourDictionary (dict): A dictionary mapping body parts to their respective colors for visualization.
    InputDir (str): The input directory containing the dataset.
    DataDir (str): The directory containing calibration data.
    SubDataDir (str): The subdirectory containing the processed data.
    VideoDir (str): The directory containing the video files.
    CamNames (list): A list of camera names.
    VideoFiles (list): A sorted list of video files.
    Out3DDict (dict): The processed 3D data dictionary.

Usage:
    Run this script to visualize the 3D tracking data of birds. The script loads the necessary data, processes it, and then visualizes it using the specified parameters.
"""

import os
import sys
import pickle
import numpy as np
from natsort import natsorted
sys.path.append("Utils")
from ReprojectVisualFields import Visualize3D_VS

# Dictionary to map body parts to their respective colors for visualization
ColourDictionary = {
    "hd_bill_tip": (180, 95, 26),
    "hd_bill_base":(92, 228, 248),
    "hd_eye_left" :(137, 227, 87),
    "hd_eye_right":(59, 51, 237),
    "hd_cheek_left":(241, 193, 153),
    "hd_cheek_right":(0, 120, 255),
    "hd_neck_left": (156, 61, 129),
    "hd_neck_right":(17, 194, 245),
    "hd_head_back":(90, 131, 181),
    "bd_bar_left_front":(216, 113, 28),
    "bd_bar_left_back":(126, 194, 46),
    "bd_bar_right_front":(0, 120, 255),
    "bd_bar_right_back":(81, 97, 246),
    "bd_tail_base":  (68, 106, 152)
}

if __name__ == "__main__":
    # Input directory containing the dataset
    InputDir = "../SampleDataset/3DTracking/bird_3B00186CDA_trial_32_time_2023-11-05 10_30_28.117871/"
    # Directory containing calibration data
    DataDir = "../SampleDataset/Calibration/data"
    # Subdirectory containing the processed data
    SubDataDir = os.path.join(InputDir,"data")
    # Directory containing the video files
    VideoDir = os.path.join(InputDir,"videos")

    # List of camera names
    CamNames = ["mill1","mill2","mill3","mill4","mill5","mill6"]

    # Get sorted list of video files
    VideoFiles = natsorted([os.path.join(VideoDir,subfile) for subfile in os.listdir(VideoDir) if subfile.endswith(".mp4")])
    # Load the processed 3D data dictionary
    Out3DDict = pickle.load(open(os.path.join(SubDataDir,"Out3DDictProcessed.p"),"rb"))

    # Visualize the 3D tracking data
    Visualize3D_VS(Out3DDict, DataDir, VideoFiles, CamNames, FrameDiffs=[0]*len(CamNames),
                   startFrame=0, TotalFrames=-1, VisualizeIndex=2, ColourDictionary=ColourDictionary,
                   Extrinsic="Initial", show=True, save=True, VSAngle=60, ElvAngle=0, Magnitude=3)