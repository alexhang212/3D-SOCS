""" 
Given a folder of videos, do intrinsic + extrinsic calibration

"""

import sys
import cv2
from cv2 import aruco

sys.path.append("./")

from Calibration import ExtrinsicCalibration
from Calibration import IntrinsicCalibration
from glob import glob
import os
import pickle


def main():
    InputDir = "../SampleDataset/Calibration/videos" ##Input directory to videos
    DataDir = "../SampleDataset/Calibration/data"
    CamNames = ["mill1","mill2","mill3","mill4","mill5","mill6"] #Camera Names

    if not os.path.exists(DataDir):
        os.mkdir(DataDir)

    imsize = None
    
    ##Definition of calibration board
    CalibBoardDict = {
    "widthNum": 5,
    "lengthNum" : 8,
    "squareLen" : 24, ## in mm
    "arucoLen" : 19, ## in mm
    "ArucoDict" : aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    }

    VideoPaths = sorted(glob(os.path.join(InputDir, "*.mp4")))

    ##Intrinsic:
    for x in range(len(CamNames)):
        InputVid = VideoPaths[x]
        retCal, cameraMatrix, distCoeffs, rvecs, tvecs,pVErr,NewCamMat,roi = IntrinsicCalibration.calibrate_auto(InputVid,CalibBoardDict,Freq=10,debug = False)
        IntSavePath = os.path.join(DataDir,"%s_Intrinsic.p"%CamNames[x])
        pickle.dump((retCal, cameraMatrix, distCoeffs, rvecs, tvecs,pVErr,NewCamMat,roi), open(IntSavePath,"wb" ))
        
    
    # #Extrinsics:
    ExtrinsicCalibration.AutoExtrinsicCalibrator(VideoPaths,DataDir,None,CalibBoardDict,
                                                 CamNames,imsize,PrimaryCamera=0,Undistort=True)
    
    ExtrinsicCalibration.GetReprojectErrors(CamNames,DataDir,Optimized=False)




if __name__ == "__main__":
    main()