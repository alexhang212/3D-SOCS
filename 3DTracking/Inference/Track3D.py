"""
Sample 3D tracking script for sample dataset
This script performs 3D tracking on a sample dataset using YOLO and DeepLabCut. 
It includes the following steps:
1. Load necessary libraries and set up paths.
2. Load YOLO and DeepLabCut models.
3. Run YOLO to detect objects in video frames.
4. Run DeepLabCut to track body parts in the detected objects.
5. Apply post-processing to smooth the 2D tracking results.
6. Perform triangulation to obtain 3D coordinates from 2D tracking data.
Constants:
    INFERENCE_BODYPARTS (list): List of body parts to be tracked.
Main Function:
    __main__: Executes the 3D tracking pipeline.
        - Sets up input directories and camera names.
        - Loads YOLO and DeepLabCut models.
        - Runs YOLO to detect objects in video frames.
        - Runs DeepLabCut to track body parts in the detected objects.
        - Applies post-processing to smooth the 2D tracking results.
        - Performs triangulation to obtain 3D coordinates from 2D tracking data.
"""
import os
import sys
import pickle
import pandas as pd
sys.path.append("../Repositories/DeepLabCut-live")
sys.path.append("./")

from ultralytics import YOLO
from dlclive import DLCLive, Processor
import deeplabcut as dlc
from Utils import Track3DFunctions as T3D

from natsort import natsorted

import datetime as dt
import numpy as np

import cv2
from tqdm import tqdm

# List of body parts to be tracked
INFERENCE_BODYPARTS = [
    "hd_bill_tip",
    "hd_bill_base",
    "hd_cheek_left",
    "hd_cheek_right",
    "hd_head_back",
    "bd_tail_base",
    "bd_bar_left_front",
    "bd_bar_left_back",
    "bd_bar_right_back",
    "bd_bar_right_front",
    "hd_eye_right",
    "hd_eye_left",
    "bd_shoulder_right",
    "bd_tail_tip",
    "hd_neck_right",
    "bd_shoulder_left",
    "hd_neck_left",
    "bd_keel_top"
]

if __name__ == "__main__":
    # Set up input directories and camera names
    InputDir = "../SampleDataset/3DTracking/bird_3B00186CDA_trial_32_time_2023-11-05 10_30_28.117871/"
    DataDir =  "../SampleDataset/Calibration/data"
    SubDataDir = os.path.join(InputDir, "data")
    VideoDir = os.path.join(InputDir, "videos")

    CamNames = ["mill1", "mill2", "mill3", "mill4", "mill5", "mill6"]

    # Get list of video files
    VideoFiles = natsorted([os.path.join(VideoDir, subfile) for subfile in os.listdir(VideoDir) if subfile.endswith(".mp4")])

    # Create sub-data directory if it doesn't exist
    if not os.path.exists(SubDataDir):
        os.makedirs(SubDataDir)

    # Load YOLO and DeepLabCut models
    YOLOPath = "../SampleDataset/Weights/GretiBlutiYOLO.pt"
    ExportModelPath = "../SampleDataset/Weights/Greti_DLC"

    dlc_proc = Processor()
    dlc_liveObj = DLCLive(ExportModelPath, processor=dlc_proc)
    CropSize = (320, 320)

    # Run YOLO to detect objects in video frames
    OutbboxDict, BBoxConfDict = T3D.RunYOLOTrack(YOLOPath, VideoFiles, DataDir, CamNames, startFrame=0, TotalFrames=-1, ScaleBBox=1, 
                                                 FrameDiffs=[0]*len(CamNames), Objects=["greti", "bluti"], Extrinsic="Initial", undistort=True, YOLOThreshold=0.7)
    
    # Save YOLO output
    pickle.dump(OutbboxDict, open(os.path.join(SubDataDir, "OutbboxDict.p"), "wb"))
    pickle.dump(BBoxConfDict, open(os.path.join(SubDataDir, "BBoxConfDict.p"), "wb"))

    # Load YOLO output
    OutbboxDict = pickle.load(open(os.path.join(SubDataDir, "OutbboxDict.p"), "rb"))
    BBoxConfDict = pickle.load(open(os.path.join(SubDataDir, "BBoxConfDict.p"), "rb"))

    # Run DeepLabCut to track body parts in the detected objects
    Out2DDict = T3D.RunDLC(dlc_liveObj, OutbboxDict, BBoxConfDict, DataDir, 
                           VideoFiles, CamNames, CropSize, INFERENCE_BODYPARTS,
                           startFrame=0, TotalFrames=-1, FrameDiffs=[0]*len(CamNames), Extrinsic="Initial",
                           DLCConfidenceThresh=0.3, YOLOThreshold=0.5,
                           undistort=True)

    # Save DeepLabCut output
    pickle.dump(Out2DDict, open(os.path.join(SubDataDir, "Out2DDict.p"), "wb"))

    # Load DeepLabCut output
    Out2DDict = pickle.load(open(os.path.join(SubDataDir, "Out2DDict.p"), "rb"))
    BBoxConfDict = pickle.load(open(os.path.join(SubDataDir, "BBoxConfDict.p"), "rb"))

    # Apply post-processing to smooth the 2D tracking results
    Filtered2DDict = T3D.Filter2D(Out2DDict, CamNames, INFERENCE_BODYPARTS, WindowSize=3)

    # Save filtered 2D tracking results
    pickle.dump(Filtered2DDict, open(os.path.join(SubDataDir, "Filtered2DDict.p"), "wb"))

    # Perform triangulation to obtain 3D coordinates from 2D tracking data
    Out3DDict, Out2DDict = T3D.TriangulateFromPointsMulti(Filtered2DDict, BBoxConfDict, DataDir, 
                                                          CamNames, INFERENCE_BODYPARTS, Extrinsic="Initial",
                                                          TotalFrames=-1, YOLOThreshold=0.5, CombinationThreshold=0.7,
                                                          DminThresh=50)
    
    # Save 3D tracking results
    pickle.dump(Out3DDict, open(os.path.join(SubDataDir, "Out3DDict.p"), "wb"))
