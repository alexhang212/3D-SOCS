"""Functions for 3D Tracking"""

import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from ultralytics import YOLO
import math
import sys
import pandas as pd
from itertools import combinations
from collections import Counter


sys.path.append("./Utils")
from BundleAdjustmentTool import BundleAdjustmentTool_Triangulation
import HuangMatching



def DLCInference(InferFrame,box,dlc_liveObj,CropSize):
    """Inference for DLC"""
    box = [0 if val < 0 else val for val in box] #f out of screen, 0


    Crop = InferFrame[round(box[1]):round(box[3]),round(box[0]):round(box[2])]


    if dlc_liveObj.sess == None: #if first time, init
        DLCPredict2D = dlc_liveObj.init_inference(Crop)

    DLCPredict2D= dlc_liveObj.get_pose(Crop)

    return DLCPredict2D


def RunYOLOTrack(YOLOPath,VidPaths,DataDir,CamNames,startFrame,TotalFrames,ScaleBBox, 
                FrameDiffs,Objects=["bird"],Extrinsic = "Initial",Rotate90 = [],Rotate180=[],imgsz = None,
                undistort = False,YOLOThreshold = 0.7):
    """Get YOLO boxes"""

    counter=startFrame

    if undistort:
        CamParamDict = {}
        for cam in CamNames:
            IntPath = os.path.join(DataDir,"%s_Intrinsic.p"%cam)
            retCal, cameraMatrix, distCoeffs, rvecs, tvecs,pVErr,NewCamMat,roi = pickle.load(open(IntPath,"rb"))


            ExtPath = os.path.join(DataDir,"%s_%s_Extrinsics.p"%(cam,Extrinsic))
                ## Initial, Aligned or Optimized


            R,T = pickle.load(open(ExtPath,"rb"))

            # CamDict = {"R":cv2.Rodrigues(R)[0],"T":np.array(T,dtype=np.float64),"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs}
            CamDict = {"R":R,"T":np.array(T,dtype=np.float64),"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs}

            CamParamDict.update({cam:CamDict})

    
    ##Setup video capture objects
    capList = []
    for vid in VidPaths:    
        cap = cv2.VideoCapture(vid)
        capList.append(cap)
        
    for x,cap in enumerate(capList):
        cap.set(cv2.CAP_PROP_POS_FRAMES,counter+FrameDiffs[x]) 

    ##Initialize YOLO objects
    YOLOList = [YOLO(YOLOPath) for x in range(len(CamNames))]
    #YOLOList = [x.to("cuda:1") for x in YOLOList]

    
    OutbboxDict = {}
    BBoxConfDict = {}

    if TotalFrames == -1: #set total frames as max frame of video
        TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # for i in tqdm(range(TotalFrames), desc = "Running YOLO:"):
    for i in tqdm(range(startFrame,startFrame+TotalFrames,1),desc = "Running YOLO:"):


        FrameList = []
        FrameBBoxDict = {}
        FrameConfDict = {}
        #read frames
        for idx, cap in enumerate(capList):
            ret, frame = cap.read()
            if ret == False:
                imsize = (int(cap.get(3)),int(cap.get(4)))
                frame = np.zeros(shape=[imsize[1],imsize[0],3],dtype=np.uint8)

                return OutbboxDict, BBoxConfDict

                # FrameList.append(np.zeros(shape=[imsize[1],imsize[0],3],dtype=np.uint8)) #empty frame if camera produces shorter vid
                # continue
            
            if CamNames[idx] in Rotate90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if CamNames[idx] in Rotate180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            if undistort:
                frame =cv2.undistort(frame, CamParamDict[cam]["cameraMatrix"],CamParamDict[cam]["distCoeffs"])
            
            FrameList.append(frame)


        for x in range(len(CamNames)): #for each cam
            CurrentCam = CamNames[x]
            CamBBoxDict = {}
            CamConfDict = {}

            InferFrame = FrameList[x].copy()
            # InferFrame = InferFrame
            # import ipdb;ipdb.set_trace()
            # cv2.imwrite("temp.jpg",InferFrame)
            if imgsz == None:
                results = YOLOList[x].track(InferFrame, persist=True,verbose = False)
            else:

                results = YOLOList[x].track(InferFrame, imgsz=imgsz, persist=True,verbose = False)

            ##Filter for birds:
            classID = [key for key,val in results[0].names.items() if val in Objects]
            # classID = [key for key,val in results[0].names.items() if val == "head"][0]

            # frame = results[0].plot()
            DetectedClasses = results[0].boxes.cls.cpu().numpy().tolist()
            Confidence = results[0].boxes.conf.cpu().numpy().tolist()
            # print(Confidence)
            
            bbox = results[0].boxes.xywh.cpu().numpy().tolist()
            if len(bbox) == 0:
                FrameBBoxDict[CurrentCam] = {}
                FrameConfDict[CurrentCam] = {}
                continue


            bbox = [[box[0],box[1],box[2]*ScaleBBox,box[3]*ScaleBBox] for box in bbox] #scale width and height

            if results[0].boxes.id == None:
                # import ipdb;ipdb.set_trace()
                # print("no tracking ID")
                FrameBBoxDict[CurrentCam] = {}
                FrameConfDict[CurrentCam] = {}
                continue
            ###Get tracking IDs
            IDList = results[0].boxes.id.cpu().numpy().tolist()

            ###Save detections
            for y,id in enumerate(IDList):
                if DetectedClasses[y] not in classID:
                    continue
                box = bbox[y].copy()
                ##convert back to xyxy:
                box = [box[0]-(box[2]/2), box[1]-(box[3]/2),box[0]+(box[2]/2),box[1]+(box[3]/2)] 
                # import ipdb;ipdb.set_trace()
                DetectedClass = results[0].names[int(DetectedClasses[y])]
                CamBBoxDict["%s-%s-%s"%(CurrentCam,DetectedClass,id)] = box
                CamConfDict["%s-%s-%s"%(CurrentCam,DetectedClass,id)] = Confidence[y]
            FrameBBoxDict[CurrentCam] = CamBBoxDict
            FrameConfDict[CurrentCam] = CamConfDict

        OutbboxDict[counter] = FrameBBoxDict
        BBoxConfDict[counter] = FrameConfDict

        counter += 1


    return OutbboxDict, BBoxConfDict


def RunDLC(dlc_liveObj,OutbboxDict,BBoxConfDict,DataDir, 
               VidPaths,CamNames,CropSize,INFERENCE_BODYPARTS,
               startFrame,TotalFrames,FrameDiffs,Extrinsic = "Initial",
               DLCConfidenceThresh = 0.3, YOLOThreshold = 0.5,Rotate90 = [],Rotate180=[],
               undistort = False):
    """
    Loop through YOLO output and run deeplabcut to get postures
    To be implemented: correspondence matching
    
    """

    counter=startFrame

    CamParamDict = {}
    for cam in CamNames:
        IntPath = os.path.join(DataDir,"%s_Intrinsic.p"%cam)
        retCal, cameraMatrix, distCoeffs, rvecs, tvecs,pVErr,NewCamMat,roi = pickle.load(open(IntPath,"rb"))

        ExtPath = os.path.join(DataDir,"%s_%s_Extrinsics.p"%(cam,Extrinsic))

        R,T = pickle.load(open(ExtPath,"rb"))

        # CamDict = {"R":cv2.Rodrigues(R)[0],"T":np.array(T,dtype=np.float64),"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs}
        CamDict = {"R":R,"T":np.array(T,dtype=np.float64),"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs}

        CamParamDict.update({cam:CamDict})

    ##Setup video capture objects
    capList = []
    for vid in VidPaths:    
        cap = cv2.VideoCapture(vid)
        capList.append(cap)
        
    for x, cap in enumerate(capList):
        cap.set(cv2.CAP_PROP_POS_FRAMES,counter+FrameDiffs[x]) 
    imsize = (int(cap.get(3)),int(cap.get(4)))

    Out2DDict = {}

    if TotalFrames == -1: #set total frames as max frame of video
        TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    for i in tqdm(range(startFrame,startFrame+TotalFrames,1), desc = "Running DLC:"):
        # i = i-startFrame
        PointsDict = {} #Points Dictionary for this frame

        if i not in OutbboxDict or len(OutbboxDict[i]) ==0 or sum([len(val) for val in OutbboxDict[i].values()]) ==0:
            [cap.read() for cap in capList]
            Out2DDict[counter] = PointsDict
            counter += 1
            continue

        # import ipdb;ipdb.set_trace()


        FrameList = []
        #read frames
        for x, cap in enumerate(capList):
            ret, frame = cap.read()
            cam = CamNames[x]
            if ret == True:
                if cam in Rotate90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if cam in Rotate180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                if undistort:
                    frame =cv2.undistort(frame, CamParamDict[cam]["cameraMatrix"],CamParamDict[cam]["distCoeffs"])


                FrameList.append(frame)
            else:
                imsize = (int(cap.get(3)),int(cap.get(4)))
                # continue
                # return Out2DDict
                FrameList.append(np.zeros(shape=[imsize[1],imsize[0],3],dtype=np.uint8)) #empty frame if camera produces shorter vid
                continue
            
        for x in range(len(CamNames)): #for each cam
            InferFrame = FrameList[x].copy()
            InferFrame = InferFrame
            CamPointDict = {}
            # import ipdb;ipdb.set_trace()
            # OutbboxDict[i][CamNames[x]]

            if CamNames[x] not in OutbboxDict[i]:
                PointsDict[CamNames[x]] = CamPointDict

                continue

            for BirdID,BirdBBox in OutbboxDict[i][CamNames[x]].items():
                if BBoxConfDict[i][CamNames[x]][BirdID] < YOLOThreshold:
                    # del OutbboxDict[i][CamNames[x]][BirdID] #delete if below threhold???
                    continue
                DLCPredict2D= DLCInference(InferFrame,BirdBBox,dlc_liveObj,CropSize)
                # import ipdb;ipdb.set_trace()
                # print(DLCPredict2D)
                Dict2D = {"%s_%s"%(BirdID,INFERENCE_BODYPARTS[j]):[DLCPredict2D[j,0]+BirdBBox[0],DLCPredict2D[j,1]+BirdBBox[1]] for j in range(DLCPredict2D.shape[0]) if DLCPredict2D[j,2] >DLCConfidenceThresh }
                # import ipdb;ipdb.set_trace()

                if len(Dict2D) > 0:
                    CamPointDict.update({BirdID:Dict2D})    

            PointsDict[CamNames[x]] = CamPointDict

        Out2DDict[counter] = PointsDict

        counter += 1
        # continue

    for cap in capList:
        cap.release()

    # import ipdb;ipdb.set_trace()
    return Out2DDict
    




def TriangulateFromPointsMulti(Points2D,BBoxConfDict,DataDir, 
            CamNames,INFERENCE_BODYPARTS,FrameDiffs=[], Extrinsic = "Initial",
            TotalFrames=-1,YOLOThreshold=0.7,CombinationThreshold = 0.8,
            DminThresh=50):
    
    """Triangulate from points, version 2: reads in smoothed 2D data, already with SORT tracking IDs"""
    
    if len(FrameDiffs) == 0:
        FrameDiffs = [0]*len(CamNames)

    Out3DDict = {}
    Out2DDict = {}
    counter = 0
    if TotalFrames == -1: #set total frames as max frame of video
        TotalFrames = max(list(Points2D.keys()))

    CamParamDict = {}
    for cam in CamNames:
        IntPath = os.path.join(DataDir,"%s_Intrinsic.p"%cam)
        retCal, cameraMatrix, distCoeffs, rvecs, tvecs,pVErr,NewCamMat,roi = pickle.load(open(IntPath,"rb"))

        ExtPath = os.path.join(DataDir,"%s_%s_Extrinsics.p"%(cam,Extrinsic))


        R,T = pickle.load(open(ExtPath,"rb"))

        CamDict = {"R":R,"T":np.array(T,dtype=np.float64),"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs}

        CamParamDict.update({cam:CamDict})

    ##regenerate master frame dict
    MasterDict = {}
    for i in tqdm(range(TotalFrames),desc="Doing Matching..."):

        # if i not in Points2D:
        #     continue

        MasterFrameDict = {}
        # import ipdb;ipdb.set_trace()
        for cam in CamNames:
            CamDict = {}
            if cam in Points2D[i]:
                for SortIndex in Points2D[i][cam].keys():
                    # import ipdb;ipdb.set_trace()

                    if SortIndex in BBoxConfDict[i][cam]: ##For interpolated 2d data, there might be no bbox
                        BBoxConf = BBoxConfDict[i][cam][SortIndex]
                        if BBoxConf < YOLOThreshold: ##If bigger than bbox confindence
                            continue

                    CamDict[SortIndex] = {"2DKP":Points2D[i][cam][SortIndex],
                                        "SORTindex":SortIndex}

            MasterFrameDict[cam] = CamDict

        ####Do matching per frame
        PointsDict = Points2D[i]

        MatchDict = HuangMatching.MatchingAlgorithm(PointsDict,
                                CamNames,CamParamDict,
                                INFERENCE_BODYPARTS,DminThresh = DminThresh)
        # import ipdb;ipdb.set_trace()
        for cam in MatchDict.keys():
            ##Flip it so that its is {localindex:globalindex}
            FlipedMatchCamDict = {val:key for key,val in MatchDict[cam].items()}
            for SortIndex in MasterFrameDict[cam].keys():
                if SortIndex in FlipedMatchCamDict.keys():
                    MasterFrameDict[cam][SortIndex]["frameMatchedID"]=FlipedMatchCamDict[SortIndex]
                else:
                    MasterFrameDict[cam][SortIndex]["frameMatchedID"]=None

        MasterDict[i] = MasterFrameDict
        
    ###For each frame get combinations of sort IDs, then tally them up?
    GlobalSortCombinationList = []
    for i in range(TotalFrames):
        FrameDict = MasterDict[i]
        UnqFrameMatchIDs = []

        for cam in CamNames:
            if len(FrameDict[cam]) < 1:
                continue
            for val in FrameDict[cam].values():
                if "frameMatchedID" not in val:
                    continue

                if val["frameMatchedID"] is not None:
                    UnqFrameMatchIDs.append(val["frameMatchedID"])

        UnqFrameMatchIDs = list(set(UnqFrameMatchIDs))


        SortLocalCombinations = []
        for cam in CamNames:
            if len(FrameDict[cam]) < 1:
                continue
            for val in FrameDict[cam].values():
                if "frameMatchedID" not in val:
                    continue
                if val["frameMatchedID"] is not None:
                    SortLocalCombinations.append((val["SORTindex"],val["frameMatchedID"]))
        
        if len(UnqFrameMatchIDs) > 0:
            for frameMatchID in UnqFrameMatchIDs:
                SortList = sorted([x[0] for x in SortLocalCombinations if x[1] == frameMatchID])
                GlobalSortCombinationList.append(SortList)

    ##For each unique combination, tally up the number of times it appears
    FlattenIDList = [x for subList in GlobalSortCombinationList for x in subList ]
    SortIDTally = Counter(FlattenIDList)
    UnqIDs = list(set(FlattenIDList))
    # import ipdb;ipdb.set_trace()


    UnqCombinations = list(combinations(UnqIDs,2))
    CombinationTallyDict = {comb:0 for comb in UnqCombinations}

    for comb in GlobalSortCombinationList:
        combUnqCombs = list(combinations(comb,2)) #unique combinations within each frame matched combinations
        for unqComb in combUnqCombs:
            if unqComb not in CombinationTallyDict: #flip it if not in dict
                unqComb = (unqComb[1],unqComb[0])

            CombinationTallyDict[unqComb] += 1 

    ###compute count of unique combination as a percentage of total frames the sort id is present
    CombinationPercentDict = {}
    for comb in CombinationTallyDict.keys():
        CombinationPercentDict[comb] = (CombinationTallyDict[comb]/SortIDTally[comb[0]],CombinationTallyDict[comb]/SortIDTally[comb[1]])

    GlobalTrackList = []

    for id in UnqIDs:
        ##Find the name and percentage overlap of the OTHER ID with current ID
        idTally = {comb[1-comb.index(id)]:val[comb.index(id)] for comb,val in CombinationPercentDict.items() if id in comb}

        MatchedIDList = [MatchedID for MatchedID in idTally.keys() if idTally[MatchedID] > CombinationThreshold]
        # if id == 80.0:
        #     import ipdb;ipdb.set_trace()
        # print(id)


        if len(MatchedIDList) <1:
            continue

        ##Check if ID is already in some set
        IsIDinSet = [GlobalTrackList.index(x) for x in GlobalTrackList if bool(set([id]+MatchedIDList) & x)]
        if len(IsIDinSet) <1: #initialize new set
            newTrackSet = set([id]+MatchedIDList)

            GlobalTrackList.append(newTrackSet)
        else:
            ##Add to existing set
            GlobalTrackList[IsIDinSet[0]].update(set([id]+MatchedIDList))            

    #########Assign global id based on GlobalTrackList
    GlobalIDDict = {}
    for id in UnqIDs:
        GlobalID = [GlobalTrackList.index(x) for x in GlobalTrackList if id in x]
        if len(GlobalID) > 0:
            GlobalIDDict[id] = GlobalID[0]

    NewMasterDict = {}
    for i in range(TotalFrames):
        FrameDict = MasterDict[i]

        for cam in CamNames:
            for localIndex in FrameDict[cam].keys():
                if FrameDict[cam][localIndex]["SORTindex"] is not None and FrameDict[cam][localIndex]["SORTindex"] in GlobalIDDict:
                    FrameDict[cam][localIndex]["globalID"] = GlobalIDDict[FrameDict[cam][localIndex]["SORTindex"]]
                else:
                    FrameDict[cam][localIndex]["globalID"] = np.nan

        NewMasterDict[i] = FrameDict

    print(GlobalTrackList)

    ####Triangulate 3d points:
    Out3DDict = {}
    Out2DDict = {}

    for i in tqdm(range(TotalFrames), desc="Triangulating..."):
        FrameDict = NewMasterDict[i]
        PointsDict = {}
        FrameSave2DDict = {}
        for cam in CamNames:
            CamDictSave = {} ##for saving, have one more level of id
            CamDict = {}
            for localID in FrameDict[cam].keys():
                if np.isnan(FrameDict[cam][localID]["globalID"]):
                    continue
                else:
                    globalID = FrameDict[cam][localID]["globalID"]
                    IDDict = {}
                    for kp in FrameDict[cam][localID]["2DKP"].keys():
                        NewName = str(globalID)+"_" + "_".join(kp.split("_")[1:len(kp.split("_"))])
                        IDDict[NewName] = FrameDict[cam][localID]["2DKP"][kp]
                        # import ipdb;ipdb.set_trace()
                        CamDict[NewName] = FrameDict[cam][localID]["2DKP"][kp]
                    CamDictSave[globalID] = IDDict
                    
            PointsDict[cam] = CamDict
            FrameSave2DDict[cam] = CamDictSave
        # import ipdb;ipdb.set_trace()
        Out2DDict[i] = FrameSave2DDict.copy()
        # import ipdb;ipdb.set_trace()
        if len(PointsDict) > 1:
            ###TRIANGULATE
            TriangTool = BundleAdjustmentTool_Triangulation(CamNames,CamParamDict)
            valid = TriangTool.PrepareInputData(PointsDict)

            if not valid:
                continue

            Point3DDict = TriangTool.run()

            Out3DDict[i] = Point3DDict
            NewMasterDict[i]["Point3D"] = Point3DDict

    return Out3DDict,Out2DDict





def Filter2D(Points2D,CamNames, INFERENCE_BODYPARTS,WindowSize=3):
    """ filter 2D points + rolling average"""

    TotalFrames = max(list(Points2D.keys()))

    FinalOutPoints2D = {i:{cam:{} for cam in CamNames} for i in range(TotalFrames+1) }

    for cam in CamNames:
        # import ipdb;ipdb.set_trace()
        CamDict = {frame:frameDict[cam] for frame,frameDict in Points2D.items() if cam in frameDict}

        AllIDList = []
        [AllIDList.extend(list(frameDict.keys())) for frameDict in list(CamDict.values())]
        AllIDList = list(set(AllIDList))

        for sortIndex in AllIDList:
            ###Get all points with this SORT index
            KP2D = {frame:frameDict[sortIndex] for frame,frameDict in CamDict.items() if sortIndex in frameDict.keys()}
            KPDict = {}


            for kp in INFERENCE_BODYPARTS:
                KPDict[kp] = {frame:KP2D[frame][key] for frame in KP2D.keys() for key in KP2D[frame].keys() if kp in key}

                kpDF = pd.DataFrame.from_dict(KPDict[kp], orient="index", columns = ["x","y"])
                if len(kpDF.index) ==0:
                    continue

                kpDF = kpDF.reindex(list(range(int(kpDF.index.min()),int(kpDF.index.max()+1))),fill_value=np.nan)
                
        

                RollingAverage = kpDF.rolling(WindowSize, min_periods=math.ceil(WindowSize/2), center=True).mean()
                # import ipdb;ipdb.set_trace()


                RollingDict = RollingAverage.to_dict(orient="index")

                ###Save back to correct format
                for frame,val in RollingDict.items():
                    # import ipdb;ipdb.set_trace()
                    if not np.isnan(val["x"]):
                        if sortIndex not in FinalOutPoints2D[frame][cam]:
                            FinalOutPoints2D[frame][cam][sortIndex] = {}

                        FinalOutPoints2D[frame][cam][sortIndex]["%s_%s"%(str(sortIndex),kp)] = [val["x"],val["y"]]

    return FinalOutPoints2D