""" 
Given a folder of videos, do intrinsic + extrinsic calibration
"""

import sys
import cv2
from cv2 import aruco

# Add the current directory to the system path
sys.path.append("./")

from Calibration import ExtrinsicCalibration
from Calibration import IntrinsicCalibration
from glob import glob
import os
import pickle

if __name__ == "__main__":
    # Input directory containing videos for calibration
    InputDir = "../SampleDataset/Calibration/videos"
    # Directory to save calibration data
    DataDir = "../SampleDataset/Calibration/data"
    # List of camera names
    CamNames = ["mill1","mill2","mill3","mill4","mill5","mill6"]

    # Create the data directory if it does not exist
    if not os.path.exists(DataDir):
        os.mkdir(DataDir)

    imsize = None
    
    # Definition of calibration board parameters
    CalibBoardDict = {
        "widthNum": 5,  # Number of squares in width
        "lengthNum" : 8,  # Number of squares in length
        "squareLen" : 24,  # Length of a square in mm
        "arucoLen" : 19,  # Length of an ArUco marker in mm
        "ArucoDict" : aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # Predefined ArUco dictionary
    }

    # Get sorted list of video paths
    VideoPaths = sorted(glob(os.path.join(InputDir, "*.mp4")))

    # Intrinsic calibration for each camera
    for x in range(len(CamNames)):
        InputVid = VideoPaths[x]
        # Perform intrinsic calibration
        retCal, cameraMatrix, distCoeffs, rvecs, tvecs, pVErr, NewCamMat, roi = IntrinsicCalibration.calibrate_auto(
            InputVid, CalibBoardDict, Freq=10, debug=False
        )
        # Save intrinsic calibration results
        IntSavePath = os.path.join(DataDir, "%s_Intrinsic.p" % CamNames[x])
        pickle.dump((retCal, cameraMatrix, distCoeffs, rvecs, tvecs, pVErr, NewCamMat, roi), open(IntSavePath, "wb"))
        
    # Extrinsic calibration
    ExtrinsicCalibration.AutoExtrinsicCalibrator(
        VideoPaths, DataDir, None, CalibBoardDict, CamNames, imsize, PrimaryCamera=0, Undistort=True
    )
    
    # Get reprojection errors for extrinsic calibration
    ExtrinsicCalibration.GetReprojectErrors(CamNames, DataDir, VideoName=None)
