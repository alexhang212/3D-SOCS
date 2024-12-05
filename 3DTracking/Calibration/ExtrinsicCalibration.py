# !/usr/bin/env python3
"""

Given synchronized videos from multiple views with moving checkerboard, do extrinsic calibration 

"""

import os
import cv2
from glob import glob
from cv2 import aruco
from tqdm import tqdm
import pickle
import numpy as np
import json
import sys
import itertools
from joblib import Parallel, delayed
import networkx as nx

sys.path.append("./")

from Utils.BundleAdjustmentTool import BundleAdjustmentTool_Triangulation


def ComputeProjectionMatrix(rotationMatrix,translationMatrix,intrinsicMatrix):
    """
    Computes projection matrix from given rotation and translation matrices
    :param rotationMatrix: 3x1 matrix
    :param translationMatrix: 3x1 Matrix
    :param intrinsicMatrix: 3x3 matrix
    :return: 3x4 projection matrix
    """
    
    if rotationMatrix.shape == (3,3):
        cv2.Rodrigues(rotationMatrix)[0]
    
    
    rotationMatrix = cv2.Rodrigues(rotationMatrix)[0]
    RT = np.concatenate((rotationMatrix, translationMatrix), axis=1)
    projectionMatrix = np.dot(intrinsicMatrix, RT)
    return projectionMatrix

def computeExtrinsic(cam1Rotation, cam1Translation, cam2Rotation, cam2Translation):
    """
    compute extrinsic matrix between the given cameras. Rotation and translation parameters supposed to bring points to
    a common coordinate space from camera space.
    Extrinsics can convert points from cam2 space to cam1 space
    Pv = R_cam1. Pc1 + t_cam1 , Pv = R_cam2 . Pc2 + t_cam2
    :param cam1Rotation: rotation ( C -> V) 3x3 matrix
    :param cam1Translation: translation (C -> V) 3x1 matrix
    :param cam2Rotation: rotation (C -> V) 3x3 matrix
    :param cam2Translation: (C -> V) 3x1 matrix
    :return: Rotation and translation ( C2 -> C1)
    """

    # Rotation
    # rotationCam2toCam1 = inverse(cam1Rotation) . cam2Rotation
    rotationCam2toCam1 = np.dot( np.linalg.inv(cam1Rotation), cam2Rotation)

    # translationCam2toCam1 = inverse(cam1Rotation) . (cam2Translation - cam1Translation)
    translationCam2toCam1 = np.dot ( np.linalg.inv(cam1Rotation), (cam2Translation-cam1Translation))

    return rotationCam2toCam1, translationCam2toCam1




def CustomGetObjImgPointAruco(charuco_corners,charuco_ids,board):
    """Built in function was buggy, custom function get coordinates of object coordinate system of calibration board given charuco detections
    
    Output: OutDict:
    ObjPoints = points in object coordinate system
    ImgIDs = Charuco IDs
    ImgPoints = points in Image pixel space
    
    """
    
    BoardCorners = board.chessboardCorners
    
    ObjPoints = []
    PointIDs = []
    ImgPoints = []
    
    
    for i, id in enumerate(charuco_ids):
        PointIDs.append(id[0])
        ObjPoints.append(BoardCorners[id[0]].tolist())
        ImgPoints.append(charuco_corners[i].tolist()[0])
        # import ipdb;ipdb.set_trace()
        
    OutDict = {"ObjPoints": ObjPoints, "ImgIDs":PointIDs, "ImgPoints":ImgPoints}
    return OutDict

def CheckDist(Corner1, Corner2):
    """
    Check the distance between two calibration board corner lists
    to see if the corners are similar. Returns True if similar, returns False if not
    """
    Diff = Corner1[0] - Corner2[0]

    if sum(abs(Diff[0]))/2 < 2:
        #If the mean difference of x and y coordinate of first point is less than 2 pixels, return True
        return True
    else:
        return False    
    

def DetectCornersVideo(VideoPath, CalibBoardDict, IntrinsicDict=None, Undistort=False, FrameDiff=0):
    """
    Given 1 video, detect all of a calibration board and return dictionary
    Undistort: whether to undistort frames during detection
    FrameDiff: video de-synchronization, when saving frame info, will add this number to real frame count
    """
    
    # Create the Charuco board
    aruco_dict = CalibBoardDict["ArucoDict"]
    board = aruco.CharucoBoard_create(
        int(CalibBoardDict["widthNum"]),
        int(CalibBoardDict["lengthNum"]),
        float(CalibBoardDict["squareLen"]),
        float(CalibBoardDict["arucoLen"]),
        aruco_dict
    )
    
    # Open the video file
    cap = cv2.VideoCapture(VideoPath)
    VidLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counter = 0
    OutDict = {}
    FirstLoop = True

    # Loop through each frame in the video
    for i in tqdm(range(VidLength)):
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the frame if required
        if Undistort:
            frame = cv2.undistort(frame, IntrinsicDict["cameraMatrix"], IntrinsicDict["distCoeffs"])

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect Aruco markers in the frame
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        # If enough corners are detected, refine their positions
        if len(corners) > 15:
            SubPixCorners = [cv2.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria) for corner in corners]
            corners = tuple(SubPixCorners)
            
            # Interpolate Charuco corners
            response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )

            # If Charuco corners are detected
            if charuco_corners is not False:
                if FirstLoop:
                    SameBool = False
                    FirstLoop = False
                else:
                    # Check if the detected corners are similar to the previous frame
                    SameBool = CheckDist(OutDict[LastFrame]["ImgPoints"], charuco_corners)

                # If corners are not similar, save the detected points
                if not SameBool:
                    PointsDict = CustomGetObjImgPointAruco(charuco_corners, charuco_ids, board)
                    SyncFrame = counter + FrameDiff
                    LastFrame = SyncFrame
                    OutDict.update({SyncFrame: PointsDict})
        else:
            # If not enough corners are detected, save None for the frame
            SyncFrame = int(counter + FrameDiff)
            OutDict.update({SyncFrame: None})

        counter += 1

    return OutDict


def SteroCalibratePair(CamNamePair, Cam1Points, Cam2Points, IntrinsicDict, imsize):
    """Given detected calibration board points from 2 cameras, do stereo calibration"""

    Cam1Name, Cam2Name = CamNamePair
    Cam1Keys, Cam2Keys = list(Cam1Points.keys()), list(Cam2Points.keys())
    OverlapFrames = [Key for Key in Cam2Keys if Key in Cam1Keys]

    # If no overlapping frames, return a large error and identity matrices
    if len(OverlapFrames) == 0:
        return 100000000000, {Cam2Name: {"R": np.identity(3), "T": np.array([0, 0, 0]).reshape(3, 1), "E": [0, 0, 0], "F": [0, 0, 0]}}

    ObjectPointsArray, Cam1ImgPointsArray, Cam2ImgPointsArray = [], [], []

    # Limit the number of overlapping frames to 50 for calibration
    if len(OverlapFrames) > 50:
        OverlapFrames = np.random.choice(OverlapFrames, 50, replace=False)

    # Loop through each overlapping frame
    for frame in OverlapFrames:
        Cam1IDs, Cam2IDs = Cam1Points[frame]["ImgIDs"], Cam2Points[frame]["ImgIDs"]
        OverlapID = sorted(list(set(Cam1IDs) & set(Cam2IDs)))

        # If less than 4 overlapping IDs, skip this frame
        if len(OverlapID) < 4:
            continue

        ObjectPointsList, Cam1ImgPointsList, Cam2ImgPointsList = [], [], []

        # Loop through each overlapping ID
        for ID in OverlapID:
            Cam1Index, Cam2Index = Cam1Points[frame]["ImgIDs"].index(ID), Cam2Points[frame]["ImgIDs"].index(ID)

            # Ensure object points match between cameras
            if Cam1Points[frame]["ObjPoints"][Cam1Index] != Cam2Points[frame]["ObjPoints"][Cam2Index]:
                raise Exception("Object points don't match, something wrong")

            ObjectPointsList.append(Cam1Points[frame]["ObjPoints"][Cam1Index])
            Cam1ImgPointsList.append(Cam1Points[frame]["ImgPoints"][Cam1Index])
            Cam2ImgPointsList.append(Cam2Points[frame]["ImgPoints"][Cam2Index])

        ObjectPointsArray.append(np.array(ObjectPointsList, dtype=np.float32))
        Cam1ImgPointsArray.append(np.array(Cam1ImgPointsList, dtype=np.float32))
        Cam2ImgPointsArray.append(np.array(Cam2ImgPointsList, dtype=np.float32))

    Cam1Mat, Cam2Mat = IntrinsicDict[Cam1Name]["cameraMatrix"], IntrinsicDict[Cam2Name]["cameraMatrix"]
    Cam1dist, Cam2dist = IntrinsicDict[Cam1Name]["distCoeffs"], IntrinsicDict[Cam2Name]["distCoeffs"]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    try:
        # Perform stereo calibration
        ret, Mat1, dist1, Mat2, dist2, R, T, E, F = cv2.stereoCalibrate(
            ObjectPointsArray, Cam1ImgPointsArray, Cam2ImgPointsArray, Cam1Mat, Cam1dist, Cam2Mat, Cam2dist, imsize, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    except Exception as e:
        print(str(e))
        return 100000000000, {Cam2Name: {"R": np.identity(3), "T": np.array([0, 0, 0]).reshape(3, 1), "E": [0, 0, 0], "F": [0, 0, 0]}}

    # Create dictionary with extrinsic parameters
    ExtrinsicDict = {Cam2Name: {"R": R, "T": T, "E": E, "F": F}}
    return ret, ExtrinsicDict

def GetBestCombination(ReprojectErr, outDictList, CamIndexPairs, CamNames, PrimaryCamera):
    """Given reprojection error of pair camera extrinsics, find best path to primary cam"""

    # Initialize error array and nested parameter list
    ErrorArray = np.full((len(CamNames), len(CamNames)), 0, dtype=np.float32)
    ParamsListNested = [[0] * len(CamNames) for _ in range(len(CamNames))]

    # Populate error array and parameter list with extrinsic parameters
    for x in range(len(CamIndexPairs)):
        pair = CamIndexPairs[x]
        From, To = pair[0], pair[1]
        ErrorArray[From, To] = ReprojectErr[x]
        ErrorArray[To, From] = ReprojectErr[x]
        ExtDict = {k: v for k, v in list(outDictList[x].values())[0].items() if k == "R" or k == "T"}
        ParamsListNested[From][To] = ExtDict

        # Compute reverse extrinsic parameters
        IdentityMat = np.identity(3)
        TranslationOrigin = np.array([0, 0, 0]).reshape(3, 1)
        R, T = ExtDict["R"], ExtDict["T"]
        NewR, NewT = computeExtrinsic(R, T, IdentityMat, TranslationOrigin)
        ReverseDict = {"R": NewR, "T": NewT}
        ParamsListNested[To][From] = ReverseDict

    # Create graph from error array
    Graph = nx.from_numpy_matrix(ErrorArray)
    BestParams = {}

    # Find best path to primary camera
    for i in range(len(CamNames)):
        if i == PrimaryCamera:
            IdentityMat = np.identity(3)
            TranslationOrigin = np.array([0, 0, 0]).reshape(3, 1)
            BestParams.update({CamNames[i]: {"R": IdentityMat, "T": TranslationOrigin}})
            continue

        # Find shortest path using Dijkstra's algorithm
        ShortestPath = nx.dijkstra_path(Graph, PrimaryCamera, i)
        CurrentParams = ParamsListNested[ShortestPath[0]][ShortestPath[1]]
        CurrentR, CurrentT = CurrentParams["R"], CurrentParams["T"]

        # Compute final extrinsic parameters along the path
        if len(ShortestPath) == 2:
            FinalR, FinalT = CurrentR, CurrentT
        else:
            for k in range(len(ShortestPath) - 2):
                From, To = ShortestPath[k + 1], ShortestPath[k + 2]
                NewR, NewT = computeExtrinsic(ParamsListNested[To][From]["R"], ParamsListNested[To][From]["T"], CurrentR, CurrentT)
                CurrentR, CurrentT = NewR.copy(), NewT.copy()
            FinalR, FinalT = CurrentR, CurrentT

        # Update best parameters
        BestParams.update({CamNames[i]: {"R": FinalR, "T": FinalT}})

    return BestParams


def GetCalibParameterDict(DataDir, CamNames, Optimized=False):
    """Get a dictionary of parameters given datadir"""

    # Initialize dictionary to store intrinsic parameters
    IntrinsicDict = {}
    for Cam in CamNames:
        # Construct the path to the intrinsic parameters file
        IntPath = os.path.join(DataDir, f"{Cam}_Intrinsic.p")
        if os.path.exists(IntPath):
            # Load intrinsic parameters if the file exists
            retCal, cameraMatrix, distCoeffs, rvecs, tvecs, pVErr, NewCamMat, roi = pickle.load(open(IntPath, "rb"))
            IntrinsicDict.update({Cam: {"cameraMatrix": cameraMatrix, "distCoeffs": distCoeffs}})
        else:
            # Use default values if the intrinsic parameters file does not exist
            print(f"No Intrinsic for Cam {Cam}, loading default")
            cameraMatrix = np.identity(3)
            distCoeffs = np.array([0, 0, 0, 0, 0])
            IntrinsicDict.update({Cam: {"cameraMatrix": cameraMatrix, "distCoeffs": distCoeffs}})

    # Initialize dictionary to store camera parameters
    CamParamsDict = {}
    for Cam in CamNames:
        if Optimized:
            # Load optimized extrinsic parameters if available
            R, T = pickle.load(open(os.path.join(DataDir, f"{Cam}_Optimized_Extrinsics.p"), "rb"))
            R = cv2.Rodrigues(R)[0]
        else:
            # Load initial extrinsic parameters
            R, T = pickle.load(open(os.path.join(DataDir, f"{Cam}_Initial_Extrinsics.p"), "rb"))

        # Compute the projection matrix
        ProjectionMat = ComputeProjectionMatrix(cv2.Rodrigues(R)[0], T, IntrinsicDict[Cam]["cameraMatrix"])
        
        # Update the camera parameters dictionary
        CamParamsDict.update({
            Cam: {
                "R": R,
                "T": T,
                "cameraMatrix": IntrinsicDict[Cam]["cameraMatrix"],
                "distCoeffs": IntrinsicDict[Cam]["distCoeffs"],
                "ProjectionMat": ProjectionMat
            }
        })

    return CamParamsDict

def AutoExtrinsicCalibrator(VideoPathList, DataDir, VideoName, ObjectDict, CamNames, imsize, PrimaryCamera=0, Undistort=False):
    """
    Read in videos, does extrinsics automatically
    ObjectDict: dictionary for what to detect, different for using board or detecting colour points
    Primary camera is the index of the camera to do stereo calibrate with all other cams
    If Undistort == True, will use intrinsics to undistort as reading the video
    """

    # Load intrinsic parameters for each camera
    IntrinsicDict = {}
    for Cam in CamNames:
        IntPath = os.path.join(DataDir, f"{Cam}_Intrinsic.p")
        if os.path.exists(IntPath):
            retCal, cameraMatrix, distCoeffs, rvecs, tvecs, pVErr, NewCamMat, roi = pickle.load(open(IntPath, "rb"))
            IntrinsicDict.update({Cam: {"cameraMatrix": cameraMatrix, "distCoeffs": distCoeffs}})
        else:
            print(f"No Intrinsic for Cam {Cam}, loading default")
            cameraMatrix = np.identity(3)
            distCoeffs = np.array([0, 0, 0, 0, 0])
            IntrinsicDict.update({Cam: {"cameraMatrix": cameraMatrix, "distCoeffs": distCoeffs}})

    # Initialize list to store detected points
    PointsDictList = []
    FrameDiffList = [0] * len(CamNames) if VideoName is None else pickle.load(open(os.path.join(DataDir, f"{VideoName}_FrameDiffs.p"), "rb"))

    # Detect corners in each video
    for x in range(len(CamNames)):
        VideoPath, Cam, FrameDiff = VideoPathList[x], CamNames[x], FrameDiffList[x]
        OutDict = DetectCornersVideo(VideoPath, ObjectDict, IntrinsicDict[Cam], Undistort=Undistort, FrameDiff=FrameDiff)
        OutDictClean = {k: v for k, v in OutDict.items() if v is not None}
        PointsDictList.append(OutDictClean)

    # Save detected points to file
    pickle.dump(PointsDictList, open(os.path.join(DataDir, f"{VideoName}_Extrinsic2DPoints.p"), "wb"))

    # Generate camera pairs for stereo calibration
    CamNamePairs = list(itertools.combinations(CamNames, 2))
    CamIndexPairs = [(CamNames.index(a), CamNames.index(b)) for a, b in CamNamePairs]
    outDictList, ReprojectErr = [], []

    # Perform stereo calibration for each camera pair
    for x in range(len(CamNamePairs)):
        ret, outDict = SteroCalibratePair(CamNamePairs[x], PointsDictList[CamIndexPairs[x][0]], PointsDictList[CamIndexPairs[x][1]], IntrinsicDict, imsize)
        outDictList.append(outDict)
        ReprojectErr.append(ret)

    # Find the best combination of extrinsic parameters
    CamErrorList, BestParamDictList = []
    for x in range(len(CamNames)):
        BestParamsDict = GetBestCombination(ReprojectErr, outDictList, CamIndexPairs, CamNames, PrimaryCamera=x)
        BestParamDictList.append(BestParamsDict)

        # Save initial extrinsic parameters for each camera
        for Cam in CamNames:
            R, T = (np.identity(3), np.array([0, 0, 0]).reshape(3, 1)) if Cam == CamNames[PrimaryCamera] else (BestParamsDict[Cam]["R"], BestParamsDict[Cam]["T"])
            SavePath = os.path.join(DataDir, f"{Cam}_Initial_Extrinsics.p")
            pickle.dump((R, T), open(SavePath, "wb"))

        # Compute reprojection errors
        MeanErr = GetReprojectErrors(CamNames, DataDir, VideoName, Optimized=False)
        CamErrorList.append(MeanErr)

    # Find the camera with the lowest reprojection error
    LowestIndex = CamErrorList.index(min(CamErrorList))
    print(f"Camera {CamNames[LowestIndex]} is best!")
    BestParamsDict = BestParamDictList[LowestIndex]

    # Save the best extrinsic parameters for each camera
    for Cam in CamNames:
        R, T = BestParamsDict[Cam]["R"], BestParamsDict[Cam]["T"]
        SavePath = os.path.join(DataDir, f"{Cam}_Initial_Extrinsics.p")
        pickle.dump((R, T), open(SavePath, "wb"))

def GetReprojectErrors(CamNames, DataDir, VideoName, Optimized=False):
    """Using detections, do triangulation across cams and do reprojection errors"""

    # Initialize dictionary to store camera parameters
    CamParamsDict = {}
    for Cam in CamNames:
        # Load intrinsic parameters
        IntPath = os.path.join(DataDir, f"{Cam}_Intrinsic.p")
        retCal, cameraMatrix, distCoeffs, rvecs, tvecs, pVErr, NewCamMat, roi = pickle.load(open(IntPath, "rb"))

        # Load extrinsic parameters (optimized or initial)
        if Optimized:
            rvec, tvec = pickle.load(open(os.path.join(DataDir, f"{Cam}_Optimized_Extrinsics.p"), "rb"))
        else:
            rvec, tvec = pickle.load(open(os.path.join(DataDir, f"{Cam}_Initial_Extrinsics.p"), "rb"))
            rvec = cv2.Rodrigues(rvec)[0]
            tvec = np.float32(tvec.reshape(3, 1))

        # Update camera parameters dictionary
        CamParamsDict.update({
            Cam: {
                "R": rvec,
                "T": tvec,
                "cameraMatrix": cameraMatrix,
                "distCoeffs": distCoeffs
            }
        })

    # Load detected points from file
    PointsDictList = pickle.load(open(os.path.join(DataDir, f"{VideoName}_Extrinsic2DPoints.p"), "rb"))
    KeyList = [list(camDict.keys()) for camDict in PointsDictList]
    AllUnqFrames = sorted(list(set().union(*KeyList)))
    ReprojectErrDict = {Cam: [] for Cam in CamNames}

    # Loop through each unique frame
    for frame in tqdm(AllUnqFrames, desc="Computing reprojection error...."):
        # Collect existing frame data for each camera
        ExistFrameData = {CamNames[i]: dict(zip(PointsDictList[i][frame]["ImgIDs"], PointsDictList[i][frame]["ImgPoints"])) for i in range(len(CamNames)) if frame in list(PointsDictList[i].keys())}

        # Skip if less than 2 cameras have data for the frame
        if len(ExistFrameData) < 2:
            continue

        # Prepare input data for triangulation
        TriangTool = BundleAdjustmentTool_Triangulation(CamNames, CamParamsDict)
        if not TriangTool.PrepareInputData(ExistFrameData):
            continue

        # Run triangulation to get 3D points
        Final3DDict = TriangTool.run()

        # Compute reprojection error for each camera
        for Cam in ExistFrameData.keys():
            List3D, List2D = [], []
            for point in Final3DDict.keys():
                if point in list(ExistFrameData[Cam].keys()):
                    List3D.append(Final3DDict[point].tolist())
                    List2D.append(ExistFrameData[Cam][point])

            # Skip if no 3D points
            if len(List3D) == 0:
                continue

            List3D = np.float32(List3D)
            NewPoints2D = cv2.projectPoints(List3D, CamParamsDict[Cam]["R"], CamParamsDict[Cam]["T"], CamParamsDict[Cam]["cameraMatrix"], CamParamsDict[Cam]["distCoeffs"])[0]
            PixelError = NewPoints2D[:, 0, :] - np.array(List2D)
            EucError = np.sqrt(PixelError[:, 0] ** 2 + PixelError[:, 1] ** 2)
            ReprojectErrDict[Cam].extend(EucError)

    # Compute mean reprojection error across all views
    RMSEAvg = [sum(v) / len(v) for v in ReprojectErrDict.values()]
    MeanErr = sum(RMSEAvg) / len(RMSEAvg)
    print(f"Mean Error across views: {MeanErr}")

    return MeanErr
