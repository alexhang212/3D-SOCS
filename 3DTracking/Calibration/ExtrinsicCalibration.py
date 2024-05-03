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
    


def DetectCornersVideo(VideoPath, CalibBoardDict, IntrinsicDict=None,Undistort=False,FrameDiff=0):
    """
    Given 1 video, detect all of a calibration board and return dictionary
    Undistort: whether to undistort frames during detection
    FrameDiff: video de-syncrhonization, when saving frame info, will add this number to real frame count
    
    """
    
    aruco_dict = CalibBoardDict["ArucoDict"]
    board = aruco.CharucoBoard_create(int(CalibBoardDict["widthNum"]),
                                    int(CalibBoardDict["lengthNum"]),
                                    float(CalibBoardDict["squareLen"]),
                                    float(CalibBoardDict["arucoLen"]),aruco_dict)
    
    cap = cv2.VideoCapture(VideoPath)
    VidLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counter = 0
    OutDict = {}

    FirstLoop = True

    # while(cap.isOpened()):
    for i in tqdm(range(VidLength)):
        # print(counter)
        ret, frame = cap.read()
        if ret==True:
            ##ADD UNDISTORT HERE
            if Undistort:
                frame =cv2.undistort(frame, IntrinsicDict["cameraMatrix"],IntrinsicDict["distCoeffs"])


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners,ids,_ = cv2.aruco.detectMarkers(gray,aruco_dict) #Detect aruco markers
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

            if len(corners) >15: #detected board
                #Get subpixel detection for corners
                SubPixCorners = []
                for corner in corners:
                    corner = cv2.cornerSubPix(gray, corner, winSize = (3,3),zeroZone = (-1,-1),
                        criteria = criteria)
                    SubPixCorners.append(corner)
                
                corners = tuple(SubPixCorners)
                    
                #interpolate markers
                response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board)

                if charuco_corners is not False:
                #corners detected, check if corners are more or less the same as previous
                    if FirstLoop: #first loop
                        SameBool = False
                        FirstLoop = False

                    else:
                        SameBool = CheckDist(OutDict[LastFrame]["ImgPoints"], charuco_corners)

                    if SameBool == False: #corners are not the same, append
                        # import ipdb;ipdb.set_trace()
                        #get object
                        PointsDict =CustomGetObjImgPointAruco(charuco_corners,charuco_ids,board)
                        SyncFrame = counter+FrameDiff
                        LastFrame = SyncFrame

                        OutDict.update({SyncFrame:PointsDict})
            else:
                SyncFrame = int(counter+FrameDiff)
                OutDict.update({SyncFrame:None})
                
            counter += 1
        else:
            break
        
    return OutDict

def SteroCalibratePair(CamNamePair, Cam1Points, Cam2Points,IntrinsicDict,imsize):
    """Given detected calibration board points from 2 cameras, do stereo calibration"""

    #Names
    Cam1Name = CamNamePair[0]
    Cam2Name = CamNamePair[1]
    
    #find frames that are overlapped 
    Cam1Keys = list(Cam1Points.keys())
    Cam2Keys = list(Cam2Points.keys())

    OverlapFrames = [Key for Key in Cam2Keys if Key in Cam1Keys]
    
    # import ipdb;ipdb.set_trace()
    if len(OverlapFrames) == 0:
        ###No overlap frame, just return very high error
        return 100000000000, {Cam2Name:{"R" : np.identity(3), "T":np.array([0,0,0]).reshape(3,1)  , "E":[0,0,0], "F":[0,0,0]}}
    
    ObjectPointsArray = []
    Cam1ImgPointsArray = []
    Cam2ImgPointsArray = []


    if len(OverlapFrames) > 50:
        ###sample random 50
        OverlapFrames = np.random.choice(OverlapFrames, 50, replace=False)

    
    for frame in OverlapFrames:
        #find which points are found in both cameras    
        Cam1IDs = Cam1Points[frame]["ImgIDs"]
        Cam2IDs = Cam2Points[frame]["ImgIDs"]
        
        #Find points where detected from both views
        OverlapID = sorted(list(set(Cam1IDs) & set(Cam2IDs)))
        
        if len(OverlapID) < 4: #need at least 4 points
            continue
        
        ObjectPointsList = []
        Cam1ImgPointsList = []
        Cam2ImgPointsList = []

        for ID in OverlapID:
            Cam1Index = Cam1Points[frame]["ImgIDs"].index(ID)
            Cam2Index = Cam2Points[frame]["ImgIDs"].index(ID)
            
            if Cam1Points[frame]["ObjPoints"][Cam1Index] != Cam2Points[frame]["ObjPoints"][Cam2Index]:
                raise Exception("Object points dont match, something wrong")

            ObjectPointsList.append(Cam1Points[frame]["ObjPoints"][Cam1Index])
            Cam1ImgPointsList.append(Cam1Points[frame]["ImgPoints"][Cam1Index])
            Cam2ImgPointsList.append(Cam2Points[frame]["ImgPoints"][Cam2Index])
            
        ObjectPointsArray.append(np.array(ObjectPointsList, dtype=np.float32))
        Cam1ImgPointsArray.append(np.array(Cam1ImgPointsList, dtype=np.float32))
        Cam2ImgPointsArray.append(np.array(Cam2ImgPointsList, dtype=np.float32))
        
    Cam1Mat = IntrinsicDict[Cam1Name]["cameraMatrix"]
    Cam2Mat = IntrinsicDict[Cam2Name]["cameraMatrix"]
    Cam1dist =IntrinsicDict[Cam1Name]["distCoeffs"]
    Cam2dist = IntrinsicDict[Cam2Name]["distCoeffs"]
    
    ##Stereo calibrate:
    #Opencv Flags
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    
    # import ipdb;ipdb.set_trace()
    # print("doing stereo:")
    try:
        ret, Mat1, dist1, Mat2, dist2, R, T, E, F= cv2.stereoCalibrate(ObjectPointsArray, 
                                                                    Cam1ImgPointsArray, 
                                                                    Cam2ImgPointsArray, 
                                                                    Cam1Mat, Cam1dist,
                                                                        Cam2Mat, Cam2dist, 
                                                                        imsize, 
                                                                        criteria = criteria, 
                                                                        flags = cv2.CALIB_FIX_INTRINSIC)
    except Exception as e:
        print(str(e))
        return 100000000000, {Cam2Name:{"R" : np.identity(3), "T":np.array([0,0,0]).reshape(3,1)  , "E":[0,0,0], "F":[0,0,0]}}

    print(ret)
    
    # import ipdb; ipdb.set_trace()
    ExtrinsicDict = {Cam2Name:{"R" : R, "T": T, "E":E, "F":F}}

    return ret,ExtrinsicDict

def GetBestCombination(ReprojectErr,outDictList,CamIndexPairs,CamNames,PrimaryCamera):
    """Given reprojection error of pair camera extrinsics, find best path to primary cam""" 
    
    ErrorArray = np.full((len(CamNames),len(CamNames)),0, dtype=np.float32) #left(rows) is from, top(columns) is to; in terms of pose relationship
    ParamsListNested = [[0]*len(CamNames) for i in range(len(CamNames))]
    
    for x in range(len(CamIndexPairs)):
        pair = CamIndexPairs[x]
        From = pair[0]
        To = pair[1]
        
        ErrorArray[From,To] = ReprojectErr[x]
        ErrorArray[To,From] = ReprojectErr[x]
        ##Initialize params
        # outDictList[x]
        ExtDict = {k:v for k,v in list(outDictList[x].values())[0].items() if k == "R" or k =="T"} 
        ParamsListNested[From][To] = ExtDict
        
        #Get the reverse pose from CamT back to CamF
        IdentityMat = np.identity(3)
        TranslationOrigin = np.array([0,0,0]).reshape(3,1)
        R = ExtDict["R"]
        T = ExtDict["T"]
        
        NewR, NewT = computeExtrinsic(R, T, IdentityMat, TranslationOrigin)
                
        ReverseDict = {"R" : NewR, "T":NewT}
        ParamsListNested[To][From] = ReverseDict
  
    Graph = nx.from_numpy_matrix(ErrorArray)
    BestParams = {}
    # nx.edges(Graph)
    for i in range(len(CamNames)):
        if i == PrimaryCamera:
            IdentityMat = np.identity(3)
            TranslationOrigin = np.array([0,0,0]).reshape(3,1)
            BestParams.update( {CamNames[i] : {"R":IdentityMat,"T":TranslationOrigin}})
            continue
        
    
        ShortestPath = nx.dijkstra_path(Graph, PrimaryCamera,i)
        # import ipdb;ipdb.set_trace()
        # #first, get params of first step inside shortest path
        CurrentParams = ParamsListNested[ShortestPath[0]][ShortestPath[1]]
        CurrentR = CurrentParams["R"]
        CurrentT = CurrentParams["T"]
        
        
        if len(ShortestPath) == 2:
            ##Shortest path only has 1 step
            FinalR = CurrentR
            FinalT = CurrentT
        else:
        # ##shortest path have multiple steps:   
            for k in range(len(ShortestPath)-2): #minus 2 to get total steps needed (e.g if 3 in shortest path, only needs 1 step)
                #Get params for two steps after:
                Original = ShortestPath[k]
                From = ShortestPath[k+1]
                To = ShortestPath[k+2]

                NewR, NewT = computeExtrinsic(ParamsListNested[To][From]["R"], ParamsListNested[To][From]["T"],
                                              CurrentR, CurrentT)
                # #Just use direct:
                # NewR = ParamsListNested[Original][To]["R"]
                # NewT = ParamsListNested[Original][To]["T"]
                
                CurrentR = NewR.copy()
                CurrentT = NewT.copy()
            FinalR = CurrentR
            FinalT = CurrentT
        BestParams.update( {CamNames[i] : {"R":FinalR,"T":FinalT}})
        
        
    return(BestParams)

def GetCalibParameterDict(DataDir, CamNames, Optimized = False):
    """Get a dictionary of parameters given datadir"""
    IntrinsicDict = {}
    for Cam in CamNames:
        IntPath = os.path.join(DataDir,"%s_Intrinsic.p"%Cam)
        if os.path.exists(IntPath):
            retCal, cameraMatrix, distCoeffs, rvecs, tvecs,pVErr,NewCamMat,roi = pickle.load(open(IntPath,"rb"))
            IntrinsicDict.update({Cam: {"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs }})
        else: #No intrinsic:
            print("No Intrinsic for Cam %s, loading default"%Cam)
            cameraMatrix = np.identity(3)
            distCoeffs = np.array([0,0,0,0,0])
            IntrinsicDict.update({Cam: {"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs }})
    
    #Prepare CamParams
    CamParamsDict = {}
    for Cam in CamNames:
        if Optimized:
            R,T = pickle.load(open(os.path.join(DataDir,"%s__Optimized_Extrinsics.p"%Cam),"rb"))
            R = cv2.Rodrigues(R)[0]

        else:
            R,T = pickle.load(open(os.path.join(DataDir,"%s_Initial_Extrinsics.p"%Cam),"rb"))
            
        ProjectionMat = ComputeProjectionMatrix(cv2.Rodrigues(R)[0],T,IntrinsicDict[Cam]["cameraMatrix"])
            
            
        
        CamParamsDict.update({
            Cam:{
             "R":R,
             "T":T,
             "cameraMatrix":IntrinsicDict[Cam]["cameraMatrix"],
             "distCoeffs":IntrinsicDict[Cam]["distCoeffs"],
             "ProjectionMat":ProjectionMat
            }
        })
        
    return CamParamsDict


def AutoExtrinsicCalibrator(VideoPathList,DataDir,VideoName,ObjectDict,CamNames,imsize,PrimaryCamera = 0, Undistort=False):
    """
    Read in videos, does extrinsics automatically 
    ObjectDict: dictionary for what to detect, different for using board or detecting colour points
    Primary camera is the index of the camera to do stero calibrate with all other cams
    If Undistort == True, will use intrinsics to undistort as reading the video
    """

    # ##Read Intrinsic Params
    IntrinsicDict = {}
    for Cam in CamNames:
        IntPath = os.path.join(DataDir,"%s_Intrinsic.p"%Cam)
        if os.path.exists(IntPath):
            retCal, cameraMatrix, distCoeffs, rvecs, tvecs,pVErr,NewCamMat,roi = pickle.load(open(IntPath,"rb"))
            IntrinsicDict.update({Cam: {"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs }})
        else: #No intrinsic:
            print("No Intrinsic for Cam %s, loading default"%Cam)
            cameraMatrix = np.identity(3)
            distCoeffs = np.array([0,0,0,0,0])
            IntrinsicDict.update({Cam: {"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs }})


    #dictionary to save detected points

    PointsDictList = []
    if VideoName is None:
        FrameDiffList = [0]*len(CamNames)
    else:
        FrameDiffList = pickle.load(open(os.path.join(DataDir,"%s_FrameDiffs.p"%VideoName),"rb"))
    # print(FrameDiffList)
    ParamDict = {}

    for x in range(len(CamNames)):
        VideoPath = VideoPathList[x]
        Cam = CamNames[x]
        FrameDiff = FrameDiffList[x]
        # import ipdb;ipdb.set_trace()
        OutDict = DetectCornersVideo(VideoPath, ObjectDict,IntrinsicDict[Cam], Undistort = Undistort,FrameDiff=FrameDiff)

        OutDictClean = {k:v for k,v in OutDict.items() if v is not None}
        PointsDictList.append(OutDictClean)
        
    #Save point detection dict as pickel
    pickle.dump(PointsDictList, open(os.path.join(DataDir,"%s_Extrinsic2DPoints.p"%VideoName),"wb" ))
    # import ipdb;ipdb.set_trace()

    CamNamePairs = list(itertools.combinations(CamNames,2))
    CamIndexPairs = [(CamNames.index(a), CamNames.index(b)) for a,b in CamNamePairs]
    # retList = []
    
    outDictList = []
    ReprojectErr = []
    
    PairwiseExtPath = os.path.join(DataDir,"PairwiseExtrinsics.p")
    # import ipdb;ipdb.set_trace()
    # if os.path.exists(PairwiseExtPath):
        # ReprojectErr,outDictList = pickle.load(open(PairwiseExtPath,"rb" ))
    # else:
    for x in range(len(CamNamePairs)):
        ret, outDict = SteroCalibratePair(CamNamePairs[x],PointsDictList[CamIndexPairs[x][0]],PointsDictList[CamIndexPairs[x][1]],IntrinsicDict,imsize)
        outDictList.append(outDict)
        ReprojectErr.append(ret)


    ##Go through each camera and use the one with lowest error:
    CamErrorList = []
    BestParamDictList = []
    for x in range(len(CamNames)):
        PrimaryCamera=x
        BestParamsDict = GetBestCombination(ReprojectErr,outDictList,CamIndexPairs,CamNames,PrimaryCamera=x)
        BestParamDictList.append(BestParamsDict)
        # import ipdb;ipdb.set_trace()
        ##Save initial estimate
        for Cam in CamNames:
            if Cam == CamNames[PrimaryCamera]:
                R = np.identity(3)
                T = np.array([0,0,0]).reshape(3,1)
            else:
                R = BestParamsDict[Cam]["R"]
                T = BestParamsDict[Cam]["T"]

            SavePath = os.path.join(DataDir,"%s_Initial_Extrinsics.p"%Cam)
            pickle.dump((R,T),open(SavePath,"wb" ))
            
        MeanErr = GetReprojectErrors(CamNames, DataDir,VideoName, Optimized=False)
        CamErrorList.append(MeanErr)
        # import ipdb;ipdb.set_trace()
    ###Save it again:
    # import ipdb;ipdb.set_trace()
    LowestIndex = CamErrorList.index(min(CamErrorList))
    print("Camera %s is best!" %(CamNames[LowestIndex]))
    BestParamsDict = BestParamDictList[LowestIndex]

    # import ipdb;ipdb.set_trace()
    for Cam in CamNames:
        R = BestParamsDict[Cam]["R"]
        T = BestParamsDict[Cam]["T"]

        SavePath = os.path.join(DataDir,"%s_Initial_Extrinsics.p"%Cam)
        pickle.dump((R,T),open(SavePath,"wb" ))
    
def GetReprojectErrors(CamNames,DataDir,VideoName,Optimized=False):
    """Using detections, do triangulation accross cams and do reprojection errors"""
    
    ##Load in params:
    ##Intrinsics:
    CamParamsDict = {}
    for Cam in CamNames:
        IntPath = os.path.join(DataDir,"%s_Intrinsic.p"%Cam)
        retCal, cameraMatrix, distCoeffs, rvecs, tvecs,pVErr,NewCamMat,roi = pickle.load(open(IntPath,"rb"))
        
        if Optimized:
            rvec,tvec= pickle.load(open(os.path.join(DataDir,"%s_Optimized_Extrinsics.p"%Cam), "rb"))

        else:
            rvec, tvec = pickle.load(open(os.path.join(DataDir,"%s_Initial_Extrinsics.p"%Cam),"rb"))
            rvec = cv2.Rodrigues(rvec)[0]
            tvec = np.float32(tvec.reshape(3,1))
        
        CamParamsDict.update({
            Cam:{
             "R":rvec,
             "T":tvec,
             "cameraMatrix":cameraMatrix,
             "distCoeffs":distCoeffs
            }
        })
    #load 2D detections:
    PointsDictList  = pickle.load(open(os.path.join(DataDir,"%s_Extrinsic2DPoints.p"%VideoName),"rb"))

    #Find all frames with detections:
        #Get all unique frames
    KeyList = [list(camDict.keys()) for camDict in PointsDictList]
    
    #get all unique frames:
    AllUnqFrames = sorted(list(set().union(*KeyList)))
    
    ReprojectErrDict = {Cam:[] for Cam in CamNames}
        
    for frame in tqdm(AllUnqFrames, desc= "Computing reprojection errror...."):
        ##get which cameras have the points for this frame:
        # ExistFrameData = {CamNames[i]:PointsDictList[i][frame] for i in range(len(CamNames)) if frame in list(PointsDictList[i].keys())}
        ExistFrameData = {CamNames[i]:dict(zip(PointsDictList[i][frame]["ImgIDs"],PointsDictList[i][frame]["ImgPoints"])) for i in range(len(CamNames)) if frame in list(PointsDictList[i].keys())}

        if len(ExistFrameData) < 2: #if only 1 view saw the board
            continue
        else:
            #triangulate point:
            TriangTool = BundleAdjustmentTool_Triangulation(CamNames, CamParamsDict)
            ret = TriangTool.PrepareInputData(ExistFrameData) ##will return false if no valid points
            
            if ret == False:
                continue
            
            Final3DDict = TriangTool.run()
        
        for Cam in ExistFrameData.keys():
            List3D = []
            List2D = []          
            for point in Final3DDict.keys():
                if point in list(ExistFrameData[Cam].keys()):
                    List3D.append(Final3DDict[point].tolist())
                    List2D.append(ExistFrameData[Cam][point])
                else:
                    continue
                
            if len(List3D) == 0: #if no point for this view
                continue
            ##Reproject Calculate RMSE Err for each point 
            List3D = np.float32(List3D)
            NewPoints2D =cv2.projectPoints(List3D, CamParamsDict[Cam]["R"],
                                           CamParamsDict[Cam]["T"], 
                                           CamParamsDict[Cam]["cameraMatrix"],
                                           CamParamsDict[Cam]["distCoeffs"])
            
            NewPoints2D = NewPoints2D[0]
            
            PixelError = NewPoints2D[:,0,:] - np.array(List2D)
            EucError = np.sqrt(PixelError[:,0]**2 + PixelError[:,1]**2)
            ReprojectErrDict[Cam].extend(EucError)
            
    RMSEAvg = []
    for k, v in ReprojectErrDict.items():
        print("Avg RMSE for %s: %s" %(k, (sum(v)/len(v))))
        RMSEAvg.append(sum(v)/len(v))
        
    MeanErr = sum(RMSEAvg)/len(RMSEAvg)
    print("Mean Error accross views: %s"%MeanErr)
        
    return MeanErr
