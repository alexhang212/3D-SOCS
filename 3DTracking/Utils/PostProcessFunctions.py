"""Functions for post processing"""

import os
import pickle
import numpy as np
import pandas as pd
import cv2
import itertools

from natsort import natsorted

import sys
sys.path.append("./Utils")
import FindPose
import Transformations as tf


def IsPointValid(Dim, point):
    """Check if a point is valid, i.e within image frame"""
    #get dimension of screen from config
    Valid = False
    if 0 <= point[0] <= Dim[0] and 0 <= point[1] <= Dim[1]:
        Valid = True
    else:
        return Valid
    return Valid


def FilterbyReproject(Out3DDict,DataDir,CamNames,imsizeDict,Ratio = 1,Extrinsic = "Aligned"):
    CamParamDict = {}
    for cam in CamNames:
        IntPath = os.path.join(DataDir,"%s_Intrinsic.p"%cam)
        retCal, cameraMatrix, distCoeffs, rvecs, tvecs,pVErr,NewCamMat,roi = pickle.load(open(IntPath,"rb"))
        ExtPath = os.path.join(DataDir,"%s_%s_Extrinsics.p"%(cam,Extrinsic))


        R,T = pickle.load(open(ExtPath,"rb"))

        # CamDict = {"R":cv2.Rodrigues(R)[0],"T":np.array(T,dtype=np.float64),"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs}
        CamDict = {"R":R,"T":np.array(T,dtype=np.float64),"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs}

        CamParamDict.update({cam:CamDict})


    NewOutDict = {}

    for i in range(max(Out3DDict.keys())):
        if i not in Out3DDict or len(Out3DDict[i]) == 0:
            continue

        frame3D = Out3DDict[i]
        OutFrameDict = {}

        ValidDict = {}

        ##Reproject 3D:
        for cam in CamNames:
            CamParams = CamParamDict[cam]
            Points3DArr = np.array(list(frame3D.values()))
            PointsNames = list(frame3D.keys())
            Allimgpts, jac = cv2.projectPoints(Points3DArr, CamParams["R"], CamParams["T"],CamParams["cameraMatrix"], CamParams["distCoeffs"])

            for x, pointName in enumerate(PointsNames):
                if pointName in ValidDict:
                    ValidDict[pointName].append(IsPointValid(imsizeDict[cam],Allimgpts[x][0]))
                else:
                    ValidDict[pointName] = [IsPointValid(imsizeDict[cam],Allimgpts[x][0])]


        for kp in ValidDict.keys():
            if sum(ValidDict[kp]) <len(CamNames)*Ratio:
                ###filter out if all cam reprojected is out of screen
                ##Changed to 70% of cameras
                continue
            else:
                OutFrameDict[kp] = frame3D[kp]
        
        NewOutDict[i] = OutFrameDict


    return NewOutDict


def FilterPoints(Out3DDict,ImportantPoints):
    """Compute difference between each point with important points, then filter them out"""

    Filtered3DDict = Out3DDict.copy()

    ###Get unique individuals
    Out3DDictKeys = [list(subDict.keys()) for subDict in Out3DDict.values()]
    Out3DDictKeys = list(itertools.chain.from_iterable(Out3DDictKeys))
    UnqIndividuals = sorted(list(set([x.split("_")[0] for x in list(set(Out3DDictKeys))])))


    for ind in UnqIndividuals:
        ##Filter only one individual:
        IndsubDict = {}
        for frame,subDict in Out3DDict.items():
            IndsubDict[frame] = {key:val for key,val in subDict.items() if key.startswith("%s_"%ind)} 
        # FilteredIndsubDict = copy.deepcopy(IndsubDict)

        for FocalPoint in ImportantPoints:
            ###Get difference of point from mean head point
            PointDiffList = []

            for k,subDict in IndsubDict.items():

                AllHeadpt = []
                for kp,vals in subDict.items():
                    if "_".join(kp.split("_")[1:4]) in ImportantPoints and not kp.endswith(FocalPoint):
                        AllHeadpt.append(vals)
                
                if len(AllHeadpt) == 0:
                    PointDiffList.append(np.nan)
                    continue
                    

                AllHeadArr = np.array(AllHeadpt)
                # MeanPT = [np.nanmean(AllHeadArr[:,x])for x in range(AllHeadArr.shape[1])]
                MeanPT = [np.nanmedian(AllHeadArr[:,x])for x in range(AllHeadArr.shape[1])]

                #is focal point in dictionary?
                FocalinDict = [key for key in subDict.keys() if key.endswith(FocalPoint)]
                if len(FocalinDict) > 0:
                    PointDiffList.append(tf.GetEucDist(MeanPT,subDict[FocalinDict[0]]))
                else:
                    PointDiffList.append(np.nan)

            DiffMean = np.nanmean(np.array(PointDiffList))
            DiffSD = np.nanstd(np.array(PointDiffList))
            DiffMedian = np.nanmedian(np.array(PointDiffList))

            #set hard threshold as 1 standard deviation
            HardThreshold = DiffMean+DiffSD


            #List of frames, might not start with frame 0
            FrameList = list(Out3DDict.keys())

            for i in range(len(PointDiffList)):
                frameNum = FrameList[i]

                ##Get kp name with bird ID
                PointName = [key for key in IndsubDict[frameNum].keys() if FocalPoint in key]

                if len(PointName) > 0 and PointDiffList[i]>HardThreshold:
                    Filtered3DDict[frameNum].pop(PointName[0]) #remove it if its above threshold
                else:
                    continue

    return Filtered3DDict


def PreprocessData(InputDict):
    """Preprocess data to split x y z"""

    ProcessedDict = {}
    ###
    for frame, frameDict in InputDict.items():
        outFrameDict = {}
        for key, val in frameDict.items():
            outFrameDict["%s_x"%key] = val[0]
            outFrameDict["%s_y"%key] = val[1]
            outFrameDict["%s_z"%key] = val[2]

        ProcessedDict[frame] = outFrameDict
    

    return ProcessedDict


def PostprocessData(outDict,colNames):
    """Put output back into dictionary format"""

    UnqNames = list(set(["_".join(x.split("_")[:-1]) for x in colNames]))

    ProcessedDict = {}

    for frame,frameDict in outDict.items():
        outFrameDict = {}
        for key in UnqNames:
            if np.isnan(frameDict["%s_x"%key]):
                continue
            else:
                outFrameDict[key] = np.array([frameDict["%s_x"%key],frameDict["%s_y"%key],frameDict["%s_z"%key]])

        ProcessedDict[frame] = outFrameDict
            

    return ProcessedDict

def InterpolateData(InputDict,Type="spline",interval = 1, show=True):

    # import ipdb;ipdb.set_trace()

    ProcessedDict = PreprocessData(InputDict)

    df = pd.DataFrame.from_dict(ProcessedDict, orient="index")
    df.sort_index(inplace=True)

    if len(df) == 0:
        return {}

    # import ipdb;ipdb.set_trace()
    ###fill missing index
    df = df.reindex(range(0,df.index[-1]+1))

    #Check if columns have enough values
    interpolatedf = df.apply(lambda x: x.interpolate(method=Type,limit = interval, limit_direction='both',limit_area="inside") if x.count() > 1 else x,axis = 0)


    colNames = list(interpolatedf.columns)
    outDict = interpolatedf.to_dict(orient="index")

    FinalDict = PostprocessData(outDict,colNames)

    return FinalDict


def FilterbyObject(Out3DDict,ObjectCoord,Keypoints, IDName,ObjFilterThresh =20):
    """
    get frame to frame RT, then fill outliers
    ObjFilterThresh: Threshold for treating point as outlier
    
    """
    # import ipdb;ipdb.set_trace()

    FilteredDict = Out3DDict.copy()
    # RotationFilterDict = {}

    PreviousFrameDict = ObjectCoord.copy() #first frame use object coord as previous frame
    FilteredTally = 0

    for i in natsorted(list(Out3DDict.keys())):
        # import ipdb;ipdb.set_trace()

        frameDict = Out3DDict[i]

        HeadDict = {"_".join(k.split("_")[1:4]):v for k,v in frameDict.items() if "_".join(k.split("_")[1:4]) in Keypoints and k.split("_")[0] == IDName}

        OverlapKP = set(HeadDict.keys()) & set(PreviousFrameDict.keys())

        PrevPoints = []
        CurrentPoints = []
        ObjectPoints = []
        PointNames = []

        for kp in Keypoints:
            if kp in OverlapKP:
                PrevPoints.append(PreviousFrameDict[kp])
                CurrentPoints.append(HeadDict[kp])
                ObjectPoints.append(ObjectCoord[kp])
                PointNames.append(kp)

        PrevPoints = np.array(PrevPoints)
        CurrentPoints = np.array(CurrentPoints)
        ObjectPoints = np.array(ObjectPoints)

        if len(ObjectPoints) < 3: #not enough detected points, filter
            # import ipdb;ipdb.set_trace()
            for kp in Keypoints:
                if "%s_%s"%(IDName,kp) in FilteredDict[i]:
                    FilteredDict[i].pop("%s_%s"%(IDName,kp))

            FilteredTally += 1
            PreviousFrameDict = ObjectCoord.copy() 
            continue


        ##First compare with object coordinate, see if low errors
        R, T = FindPose.findPoseFromPoints(ObjectPoints.T,CurrentPoints.T)
        transObjPoints = tf.transformPoints(ObjectPoints.T, R, T).T
        ObjErrorList = [tf.GetEucDist(transObjPoints[x].tolist()[0],CurrentPoints[x]) for x in range(transObjPoints.shape[0])]
        
        # WhereError = np.where(np.array(ObjErrorList)>0)[0]
        WhereError = np.where(np.array(ObjErrorList)>ObjFilterThresh)[0]


        if  len(ObjErrorList) - len(WhereError) < 3: #remaining not enough to get pose, filter
            for kp in Keypoints:
                if "%s_%s"%(IDName,kp) in FilteredDict[i]:
                    FilteredDict[i].pop("%s_%s"%(IDName,kp))

            FilteredTally += 1
            PreviousFrameDict = ObjectCoord.copy() 
            continue

        ###Remove all points defined as outlier then recompute rotation
        KPNamesFiltered = [PointNames[x] for x in range(len(PointNames)) if x not in WhereError]
        ObjPointsFiltered = [ObjectPoints.tolist()[x] for x in range(len(ObjectPoints)) if x not in WhereError ]
        CurrentPointsFiltered = [CurrentPoints.tolist()[x] for x in range(len(CurrentPoints)) if x not in WhereError ]

        ObjPointsFiltered = np.array(ObjPointsFiltered)
        CurrentPointsFiltered = np.array(CurrentPointsFiltered)

        ##Recompute new pose:
        NewR, NewT = FindPose.findPoseFromPoints(ObjPointsFiltered.T,CurrentPointsFiltered.T)

        ###Replace using object:
        AllObjPoints = np.array(list(ObjectCoord.values()))
        AllObjNames = list(ObjectCoord.keys())

        transPrevPoints = tf.transformPoints(AllObjPoints.T, NewR, NewT).T
        TransDict = {kp:transPrevPoints.tolist()[x] for x,kp in enumerate(AllObjNames)}
        
        #Just use object coordinate:
        for kp in Keypoints:
            if kp in TransDict:
                FilteredDict[i]["%s_%s"%(IDName,kp)] = TransDict[kp]
        
        # import ipdb;ipdb.set_trace()
        PreviousFrameDict = TransDict.copy()

    return FilteredDict