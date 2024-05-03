"""Visualize 3D tracking"""

import os
import sys
import pickle
import numpy as np
from natsort import natsorted
sys.path.append("Utils")
from ReprojectVisualFields import Visualize3D_VS

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
    InputDir = "../SampleDataset/3DTracking/bird_3B00186CDA_trial_32_time_2023-11-05 10_30_28.117871/"
    DataDir = "../SampleDataset/Calibration/data"
    SubDataDir = os.path.join(InputDir,"data")
    VideoDir = os.path.join(InputDir,"videos")

    CamNames = ["mill1","mill2","mill3","mill4","mill5","mill6"]

    VideoFiles = natsorted([os.path.join(VideoDir,subfile) for subfile in os.listdir(VideoDir) if subfile.endswith(".mp4")])
    Out3DDict = pickle.load(open(os.path.join(SubDataDir,"Out3DDictProcessed.p"),"rb"))

    Visualize3D_VS(Out3DDict,DataDir, VideoFiles,CamNames,FrameDiffs=[0]*len(CamNames),
                   startFrame=0,TotalFrames=-1,VisualizeIndex=2,ColourDictionary=ColourDictionary,
                   Extrinsic = "Initial",show=True,save=False,VSAngle=60,ElvAngle=0, Magnitude = 3)