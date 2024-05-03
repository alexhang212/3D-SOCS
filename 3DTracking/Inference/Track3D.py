"""Sample 3D tracking script for sample dataset"""

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
    InputDir = "../SampleDataset/3DTracking/bird_3B00186CDA_trial_32_time_2023-11-05 10_30_28.117871/"
    DataDir =  "../SampleDataset/Calibration/data"
    SubDataDir = os.path.join(InputDir,"data")
    VideoDir = os.path.join(InputDir,"videos")

    CamNames = ["mill1","mill2","mill3","mill4","mill5","mill6"]

    VideoFiles = natsorted([os.path.join(VideoDir,subfile) for subfile in os.listdir(VideoDir) if subfile.endswith(".mp4")])

    if not os.path.exists(SubDataDir):
        os.makedirs(SubDataDir)

    ## Load Weights
    YOLOPath = "../SampleDataset/Weights/GretiBlutiYOLO.pt"
    ExportModelPath = "../SampleDataset/Weights/Greti_DLC"

    
    dlc_proc = Processor()
    dlc_liveObj = DLCLive(ExportModelPath, processor=dlc_proc)
    CropSize = (320,320)
    ###

    ###Run YOLO:
    OutbboxDict, BBoxConfDict = T3D.RunYOLOTrack(YOLOPath,VideoFiles,DataDir,CamNames,startFrame=0,TotalFrames=-1,ScaleBBox=1, 
                FrameDiffs=[0]*len(CamNames),Objects=["greti","bluti"],Extrinsic = "Initial",undistort = True,YOLOThreshold = 0.7)
    
    pickle.dump(OutbboxDict,open(os.path.join(SubDataDir,"OutbboxDict.p"),"wb"))
    pickle.dump(BBoxConfDict,open(os.path.join(SubDataDir,"BBoxConfDict.p"),"wb"))


    ## Run DeepLabCut
    OutbboxDict = pickle.load(open(os.path.join(SubDataDir,"OutbboxDict.p"),"rb"))
    BBoxConfDict = pickle.load(open(os.path.join(SubDataDir,"BBoxConfDict.p"),"rb"))

    
    Out2DDict = T3D.RunDLC(dlc_liveObj,OutbboxDict,BBoxConfDict,DataDir, 
        VideoFiles,CamNames,CropSize,INFERENCE_BODYPARTS,
        startFrame=0,TotalFrames=-1,FrameDiffs=[0]*len(CamNames),Extrinsic = "Initial",
        DLCConfidenceThresh = 0.3, YOLOThreshold = 0.5,
        undistort = True)

    pickle.dump(Out2DDict,open(os.path.join(SubDataDir,"Out2DDict.p"),"wb"))



    ###Post Processing
    ####2D rolling average
    Out2DDict = pickle.load(open(os.path.join(SubDataDir,"Out2DDict.p"),"rb"))
    BBoxConfDict = pickle.load(open(os.path.join(SubDataDir,"BBoxConfDict.p"),"rb"))
    # import ipdb;ipdb.set_trace()

    Filtered2DDict = T3D.Filter2D(Out2DDict,CamNames,INFERENCE_BODYPARTS, WindowSize=3)

    pickle.dump(Filtered2DDict,open(os.path.join(SubDataDir,"Filtered2DDict.p"),"wb"))



    ###Matching and Triangulation:
    Out3DDict,Out2DDict = T3D.TriangulateFromPointsMulti(Filtered2DDict,BBoxConfDict,DataDir, 
            CamNames,INFERENCE_BODYPARTS, Extrinsic = "Initial",
            TotalFrames=-1,YOLOThreshold=0.5,CombinationThreshold = 0.7,
            DminThresh=50)
    
    pickle.dump(Out3DDict,open(os.path.join(SubDataDir,"Out3DDict.p"),"wb"))

