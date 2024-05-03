"""
This script:
Huang et al tracking algorithm

"""

import numpy as np
import math
import sys
sys.path.append("./")
from Utils.BundleAdjustmentTool import BundleAdjustmentTool_Triangulation

import itertools
import Utils.HungarianAlgorithm as HungarianAlgorithm

from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist 


def GetEucDist(Point1,Point2):
    """Get euclidian error, both 2D and 3D"""
    
    if len(Point1) ==3 & len(Point2) ==3:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2) + ((Point1[2] - Point2[2]) ** 2) )
    elif len(Point1) ==2 & len(Point2) ==2:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2))
    else:
        import ipdb;ipdb.set_trace()
        Exception("point input size error")
    
    return EucDist

    
def IsPointValid(Dim, point):
    """Check if a point is valid, i.e within image frame"""
    #get dimension of screen from config
    Valid = False
    if 0 <= point[0] <= Dim[0] and 0 <= point[1] <= Dim[1]:
        Valid = True
    else:
        return Valid
    return Valid


def FindCommonPoint(CamPairDict):
    """Given pair camera points dictionary, find the most frequent point that is seen accross cameras"""

    TallyDict = {}
    for pair, pairVal in CamPairDict.items():
        for PointDict in pairVal["3DPoints"]:
            for k in PointDict.keys():
                if k in TallyDict:
                    TallyDict[k] += 1
                else:
                    TallyDict[k] = 0

    CommonPoint = [k for k,v in TallyDict.items() if TallyDict[k] == max(list(TallyDict.values()))]

    if len(CommonPoint) == 0:
        return False
    elif TallyDict[CommonPoint[0]] == 1: #if max point only present in 1 case
        return False
    else:
        return CommonPoint[0]

def euclidean_distance_fro(Y1, Y2):
    """
    Compute the Euclidean distance between two 3D poses using the Frobenius norm.
    
    Parameters:
    Y1 -- first pose, a numpy array of shape (n, m)
    Y2 -- second pose, a numpy array of shape (n, m)
    
    Returns:
    dist -- the Euclidean distance between the two poses
    """
    dist = np.linalg.norm(Y1 - Y2, 'fro')
    return dist

def GetDiffbetween3DPoses(Pose1, Pose2):
    ###Find common points:
    CommonPoints = list(set(Pose1.keys()) & set(Pose2.keys()))

    if len(CommonPoints) == 0:
        return np.inf
    
    ###Get array of common points
    Pose1Arr = np.array([Pose1[point] for point in CommonPoints])
    Pose2Arr = np.array([Pose2[point] for point in CommonPoints])

    Dist = euclidean_distance_fro(Pose1Arr, Pose2Arr)

    return Dist



def MatchingAlgorithm(MatchingDict,CamNames,CamParamDict,KEYPOINT_NAMES,DminThresh = 200,confidence_threshold=0.5):
    """

    Input: dictionary of {CamName:{"Keypoints":np.array(KPoutput)}:}

    Matching algorithm to get correspondeces based on reprojection
    Dmin: based on Huang et al 2020, minimum threshold, after that finish matching
    """
    # import ipdb;ipdb.set_trace()
    ##Get all combinations of cameras
    CamNamePairs = list(itertools.combinations(list(MatchingDict.keys()),2))
    # CamIndexPairs = [(CamNames.index(a), CamNames.index(b)) for a,b in CamNamePairs]
    Point3DDict = {}

    #For each possible camera pair and for each possible detection pair, triangulate to get 3d pose subspace

    for CamPair in CamNamePairs:#For each possible camera pair
        Cam1KP = MatchingDict[CamPair[0]]
        Cam2KP = MatchingDict[CamPair[1]]

        if len(Cam1KP) == 0 or len(Cam2KP) ==0:
            continue

        TempCamNames = [CamPair[0],CamPair[1]]

        ObjectPairs = list(itertools.product(Cam1KP.keys(),Cam2KP.keys()))

        # ErrorArray = np.full((len(Cam1KP),len(Cam2KP)),100000000000, dtype=np.float32) #left(rows) is KP1, top(columns) is KP2

        # Point3DDict = {}


        for IndexPair in ObjectPairs:
            #Prepare data:
            Cam1Dict = {"_".join(key.split("_")[1:len(key.split("_"))]):val for key,val in Cam1KP[IndexPair[0]].items()}
            Cam2Dict = {"_".join(key.split("_")[1:len(key.split("_"))]):val for key,val in Cam2KP[IndexPair[1]].items()}

            # if len(Cam1Dict) ==0 or len(Cam2Dict) == 0:
            #     continue

            Point2DDict = {TempCamNames[0]:Cam1Dict, TempCamNames[1]:Cam2Dict}

            TriangTool = BundleAdjustmentTool_Triangulation(TempCamNames,CamParamDict)

            Out = TriangTool.PrepareInputData(Point2DDict)
            if Out == False: #no common points for triangulation
                Point3DDict.update({IndexPair:{}})

                continue

            Points3Dsubdict = { TriangTool.PointsNameList[x]:TriangTool.Points3DArr[x] for x in range(len(TriangTool.PointsNameList))}
            NewIndexPair = ((TempCamNames[0],IndexPair[0]),(TempCamNames[1],IndexPair[1]))
            # import ipdb;ipdb.set_trace()
            Point3DDict[NewIndexPair] = Points3Dsubdict


    # import ipdb;ipdb.set_trace()
    if len(Point3DDict) == 0: ##Nothing
        return {cam:{id:None for id in MatchingDict[cam].keys()} for cam in list(MatchingDict.keys())}
    # FinalCamDict = {cam:{0:PairMatches[x]} for x,cam in enumerate(PairCamNames)}

    ### Implementing Huang et al 2020 for dynamic matching
    ### First get matrix of differences between all 3d point pairs (with corresponding camera and index pairs)
    ### Find the min in that matrix, then start a global matched set
    ### if any set already has any camera/index pair, just add to the set, if not start a new set
    ### if a camera already exist in a set (with different index), use the existing one and throw this out

    ErrorArray = np.full((len(Point3DDict.keys()),len(Point3DDict.keys())),np.inf, dtype=np.float32) 

    ###Matrix for the names:
    PairPointNames = list(Point3DDict.keys())
    PointNamesMatrix = [[[x,y] for x in PairPointNames] for y in PairPointNames]

    AllNamePairs = list(itertools.product(Point3DDict.keys(),Point3DDict.keys()))

    for namepair in AllNamePairs:
        if namepair[0] == namepair[1]:
            continue
        Dist = GetDiffbetween3DPoses(Point3DDict[namepair[0]],Point3DDict[namepair[1]])
        Index0 = PairPointNames.index(namepair[0])
        Index1 = PairPointNames.index(namepair[1])

        # import ipdb;ipdb.set_trace()
        ErrorArray[Index0,Index1] = Dist


    # np.fill_diagonal(ErrorArray, np.inf)
    Dmin = 0
    GlobalMatchedList = []

    while Dmin < DminThresh:
        MinIndex = np.unravel_index(ErrorArray.argmin(),ErrorArray.shape )
        Dmin = ErrorArray[MinIndex[0],MinIndex[1]]
        # PointCamPairs = set(IndexList[MinIndex[0]]+ IndexList[MinIndex[1]])
        PairNames = PointNamesMatrix[MinIndex[0]][MinIndex[1]]
        PairNameUnravel = [PairNames[0][0],PairNames[0][1],PairNames[1][0],PairNames[1][1]]
        PointCamPairs = set(PairNameUnravel)


        existSetIndex = set([x for x,Subset in enumerate(GlobalMatchedList) for pair in Subset if pair in PointCamPairs ])
        #indexes in global list where there is overlap cam index pair

        if len(existSetIndex)==0: ##no set already matched, create new set
            GlobalMatchedList.append(PointCamPairs)
        else: 
            if len(existSetIndex)>1: #both already matched
                ErrorArray[MinIndex[0],MinIndex[1]] = np.inf #remove the already matched pair
                continue
            
            MatchedSet = GlobalMatchedList[list(existSetIndex)[0]]
            PresentCamNames = [pair[0] for pair in PointCamPairs]
            for pair in MatchedSet:
                if pair[0] in PresentCamNames: ##cam already matched here
                    ##to find which one of the pairs shares cam name:
                    MatchedPair = [subPair for subPair in PointCamPairs if subPair[0] == pair[0]]
                    [PointCamPairs.discard(subPair) for subPair in MatchedPair] #if already matched, no new matching

            GlobalMatchedList[list(existSetIndex)[0]].update(PointCamPairs)
                    
        ErrorArray[MinIndex[0],MinIndex[1]] = np.inf #remove the already matched pair

    # import ipdb;ipdb.set_trace()

    ##Convert Global Matching list back to dictionary format, with an arbitiary index for each individual

    FinalCamDict = {key:{} for key in CamNames}

    # import ipdb;ipdb.set_trace()
    #Go through each cluster
    for x in range(len(GlobalMatchedList)): 
        for cam in CamNames:
            IndexList = [pair[1] for pair in GlobalMatchedList[x] if pair[0] == cam]
            if len(IndexList) == 0:
                CamIndex = None
            else:
                CamIndex = IndexList[0]

            FinalCamDict[cam].update({x:CamIndex})
    # import ipdb;ipdb.set_trace()
    # FinalCamDict["Cam4"].values()
    return FinalCamDict
