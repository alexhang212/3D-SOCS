"""Get head coordinate system from 3D tracking """
"""Part of gap filling pipeline, go through predictions and get object definition per individual per frame"""

import numpy as np
import math
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *
import scipy.stats as st


def GetMedianFromTrial(Out3DDict,AllObjNames,ObjectDefNames, show= False):
    """Given a trial, get median object of all birds in the trial"""
    
    RDictHead, TDictHead, HeadCoordDict  = GetFrameObjectDict(Out3DDict,AllObjNames,ObjectDefNames)

    AllHeadCoord = []
    [AllHeadCoord.extend(HeadCoordDict[ID]) for ID in HeadCoordDict.keys()]

    ###Get Median:
    KPDict = {}
    for kp in AllObjNames:
        kpList = [valDict[kp] for valDict in AllHeadCoord if kp in valDict]
        KPDict[kp] = np.array(kpList)

    MedianObjHead = {}
    for k,v in KPDict.items():
        if len(v) == 0:
            continue
        MedX = np.median(v[:,0])
        MedY = np.median(v[:,1])
        MedZ = np.median(v[:,2])
        MedianObjHead[k] = np.array([MedX,MedY,MedZ])


    if show:
        VisualizePlot(KPDict,MedianObjHead)


    return MedianObjHead



def PreprocessDict(Points3Dlist,Keypoints):
    """Preprocess 3d dictionary, organize by bird ID"""

    BirdMasterDict = {} #master dictionary to store all keypoints

    for i in Points3Dlist.keys():
        frameDict = Points3Dlist[i]

        ##get all unique bird IDs in this frame:
        UnqIDs = list(set([k.split("_")[0] for k in frameDict.keys()]))

        for ID in UnqIDs:
            IDdict = {"_".join(k.split("_")[1:]):pt for k,pt in frameDict.items() if k.startswith(ID) and "_".join(k.split("_")[1:]) in Keypoints}
            IDdict["Index"] = i

            if ID not in list(BirdMasterDict.keys()):
                BirdMasterDict.update({ID:[IDdict]})
            else:
                BirdMasterDict[ID].append(IDdict)
    
    return BirdMasterDict

def GetMagnitude(point):
    """Return magnitute of vector"""
    return math.sqrt((point[0]**2+point[1]**2+point[2]**2))

def GetMidPoint(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2,(p1[2]+p2[2])/2]

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

def DefineObj(beak,pt1,pt2):
    """Define head object coordinate from 3 points: beak and 2 eyes"""
    
    ##Get vector from eye to beak
    Vec1 = pt1-beak
    Vec2 = pt2-beak
    PlaneNormal = np.cross(Vec1,Vec2)
    NormalUnit = PlaneNormal/GetMagnitude(PlaneNormal)

    ###get vector of between eye to beak (as the y axis)
    BetweenEye = GetMidPoint(pt1,pt2)
    ForwardVec = BetweenEye-beak
    ForwardUnit = ForwardVec/GetMagnitude(ForwardVec)
    
    #get horizontal axis (x), normal is (z), mid eye to beak is y
    HorizontalAxis = np.cross(PlaneNormal,ForwardVec)
    HorizontalUnit = HorizontalAxis/GetMagnitude(HorizontalAxis)

    ###Calc rotation matrix against principle axes
    Xaxis = np.array([1,0,0])
    Yaxis = np.array([0,1,0])
    Zaxis = np.array([0,0,1])

    ##Rotate matrix:
    R = np.array([[np.dot(Xaxis,HorizontalUnit),-np.dot(Xaxis,ForwardUnit),np.dot(Xaxis,NormalUnit)],
                 [np.dot(Yaxis,HorizontalUnit),-np.dot(Yaxis,ForwardUnit),np.dot(Yaxis,NormalUnit)],
                 [np.dot(Zaxis,HorizontalUnit),-np.dot(Zaxis,ForwardUnit),np.dot(Zaxis,NormalUnit)]])
    
    T = BetweenEye

    return R,T

def GetObjCoords(Point3D, R,T):
    """Given R and T and 3d points in world coordinate, get object coordinate"""

    ###Get points in object coordinate system:
    # import ipdb;ipdb.set_trace()
    # ObjPointsObj = np.dot((Point3D-T),R)

    Point3D = Point3D.T
    T = np.array(T).reshape(3,1)

    Translated = Point3D-T

    #test:
    ObjPointsObj = np.dot(np.linalg.inv(R),Translated).T


    return ObjPointsObj

def GlobaltoLocal(Point3D,R,T):
    """Test Global to Local"""   
    # import ipdb;ipdb.set_trace() 
    Point3D = Point3D.T

    #test:
    ObjPointsObj = np.dot(np.linalg.inv(R),Point3D).T

    Translated = ObjPointsObj.reshape(3,1)-np.array(T).reshape(3,1)


    return Translated.reshape(1,3)

def GetObjectsBird(BirdMasterDict,ObjectDefinitionList):
    """Get object definition for each bird"""
    AllBirds = list(BirdMasterDict.keys())    
    RDictHead = {} #R and T for each bird per frame
    TDictHead = {}
    HeadCoordListDict = {} #list of dictionaries of head coordinates per frame


    for birdID in AllBirds:
        birdDictList = BirdMasterDict[birdID]
        HeadCoordList = []

        BirdHeadRDict = {}
        BirdHeadTDict = {}
        
        for frameDict in birdDictList:
            FrameIndex = frameDict["Index"]

            ##Head coord system:
            if ObjectDefinitionList[0] not in frameDict or ObjectDefinitionList[1] not in frameDict or ObjectDefinitionList[2] not in frameDict:
                continue
            R, T =  DefineObj(frameDict[ObjectDefinitionList[0]],frameDict[ObjectDefinitionList[1]],frameDict[ObjectDefinitionList[2]])

            BirdHeadRDict.update({FrameIndex:R})
            BirdHeadTDict.update({FrameIndex:T})

            # HeadObjDict = {k:v for k,v in frameDict.items() if k.startswith("hd")}
            HeadObjDict = frameDict.copy()
            HeadObjDict.pop("Index")
            HeadObjPointsWorld = np.array(list(HeadObjDict.values()))

            # import ipdb;ipdb.set_trace()
            HeadObjPointsObj = GetObjCoords(HeadObjPointsWorld,R,T)

            HeadObjDictBird = {list(HeadObjDict.keys())[x]:HeadObjPointsObj[x] for x in range(HeadObjPointsObj.shape[0])}
            HeadObjDictBird.update({"index" : FrameIndex})
            HeadCoordList.append(HeadObjDictBird)
            
        
        RDictHead.update({birdID:BirdHeadRDict})
        TDictHead.update({birdID:BirdHeadTDict})
        HeadCoordListDict.update({birdID:HeadCoordList})

        # pickle.dump(HeadCoordList,open(os.path.join("./Data/Evaluation/SubjectObjects","%s_hd.p"%birdID), "wb"))
        # pickle.dump(BackpackCoordList,open(os.path.join("./Data/Evaluation/SubjectObjects","%s_bp.p"%birdID), "wb"))

    return RDictHead, TDictHead, HeadCoordListDict

def GetObjectFromPolar(R,T ,Azimuth=30,Elevation = 30, Magnitude = 7):
    """
    Given polar coordinates of an object, get object coordinate in world coordinate system
    """

    # import ipdb;ipdb.set_trace()
    ####Get new points based on points in obj coord system
    PrinciplePoint = np.array([0,0,1]) * Magnitude

    AzmRad = math.radians(Azimuth)
    RotationAzm = np.array([[math.cos(AzmRad),math.sin(AzmRad),0],
                            [-math.sin(AzmRad),math.cos(AzmRad),0],[0,0,1]])

    ##Rotate around x axis:
    ElvRad = math.radians(Elevation)
    RotationElv = np.array([[1,0,0],[0,math.cos(ElvRad),-math.sin(ElvRad)],
                            [0,math.sin(ElvRad),math.cos(ElvRad)]])
    
    TransformedPoint = np.dot(PrinciplePoint,RotationAzm)
    TransformedPoint = np.dot(TransformedPoint,RotationElv)


    ###Transform back to world coordinate
    inverse_R = np.linalg.inv(R)
    # inverse_R = R

    NewVec = np.add(np.dot(TransformedPoint,inverse_R ),T)

    return NewVec

def GetVisualFieldPoints(R,T ,Angle=75, ElvAngle = 0,Magnitude = 7, pigeon=False):
    """Get visual field of a bird
    """

    # import ipdb;ipdb.set_trace()
    ####Get new points based on points in obj coord system
    PrinciplePoint = np.array([0,1,0]) * Magnitude*100
    
    AngleRadDown = math.radians(ElvAngle)

    # ##Old:
    AngleRadRight = math.radians(-Angle)

    #rotate down

    rotation_matrix_down = np.array([
        [1, 0, 0],
        [0, np.cos(AngleRadDown), -np.sin(AngleRadDown)],
        [0, np.sin(AngleRadDown), np.cos(AngleRadDown)]
    ])

    VSRight = np.dot(rotation_matrix_down,PrinciplePoint)

    rotation_matrix_right = np.array([
        [np.cos(AngleRadRight), -np.sin(AngleRadRight), 0],
        [np.sin(AngleRadRight), np.cos(AngleRadRight), 0],
        [0, 0, 1]
    ])

    VSRight = np.dot(rotation_matrix_right,VSRight)


    AngleRadLeft = math.radians(Angle)

    #rotate down

    rotation_matrix_down = np.array([
        [1, 0, 0],
        [0, np.cos(AngleRadDown), -np.sin(AngleRadDown)],
        [0, np.sin(AngleRadDown), np.cos(AngleRadDown)]
    ])

    VSLeft = np.dot(rotation_matrix_down,PrinciplePoint)

    rotation_matrix_left = np.array([
        [np.cos(AngleRadLeft), -np.sin(AngleRadLeft), 0],
        [np.sin(AngleRadLeft), np.cos(AngleRadLeft), 0],
        [0, 0, 1]
    ])

    VSLeft = np.dot(rotation_matrix_left,VSLeft)

    ##Frontal visual field
    AngleRadFront = math.radians(ElvAngle) #-45 down along x axis
    # ShortPrinciplePoint = PrinciplePoint/5
    ShortPrinciplePoint = PrinciplePoint

    rotation_matrix_down = np.array([
        [1, 0, 0],
        [0, np.cos(AngleRadFront), -np.sin(AngleRadFront)],
        [0, np.sin(AngleRadFront), np.cos(AngleRadFront)]
    ])

    VSFront = np.dot(rotation_matrix_down,ShortPrinciplePoint)

    ###Transform back to world coordinate
    inverse_R = np.linalg.inv(R)
    # inverse_R = R

    VSRightWorld = np.add(np.dot(VSRight,inverse_R ),T)
    VSLeftWorld = np.add(np.dot(VSLeft,inverse_R ),T)
    VSFrontWorld = np.add(np.dot(VSFront,inverse_R ),T)

    return VSRightWorld, VSLeftWorld ,VSFrontWorld

def VisualizePlot(KeypointArrDict,MeanObjDict):
    """plot 3d heatmap of all points"""
    # import ipdb;ipdb.set_trace()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize axis limits with values that will be updated as we process the points
    x_min, x_max, y_min, y_max, z_min, z_max = float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')

    for point_type, points in KeypointArrDict.items():
        points = np.array(points)  # Convert the list of points to a NumPy array
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        ax.scatter(x, y, z, label=point_type)


        ##plot mean:
        meanPoint = MeanObjDict[point_type]
        ax.plot([meanPoint[0]], [meanPoint[1]], [meanPoint[2]], marker='^', markersize=8, color='black', label='Mean ({})'.format(point_type))
        ax.text(meanPoint[0], meanPoint[1], meanPoint[2], f'Mean ({point_type}): ({meanPoint[0]:.2f}, {meanPoint[1]:.2f}, {meanPoint[2]:.2f})', fontsize=12, color='black')



        # Update axis limits based on the current point type
        x_min = min(x_min, np.min(x))
        x_max = max(x_max, np.max(x))
        y_min = min(y_min, np.min(y))
        y_max = max(y_max, np.max(y))
        z_min = min(z_min, np.min(z))
        z_max = max(z_max, np.max(z))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Heatmap')

    plt.legend()
    plt.show()

def GetOutliers(BirdID, Keypoints,HeadCoordList):
    """For each bird, get outlier frames based on object coordinate definition"""

    OutlierTally = 0

    CoordDict = {dict["index"]:dict for dict in HeadCoordList}


    KeypointListDict = {"_".join(key.split("_")[1:3]) :[] for key in Keypoints}

    #load object definition points
    for i in range(len(HeadCoordList)):
        for key in KeypointListDict.keys():
            KeypointListDict[key].append(HeadCoordList[i][key])

    KeypointArrDict = {key:np.array(val) for key,val in KeypointListDict.items()}


    # import ipdb;ipdb.set_trace()

    ##Mean object definition:
    MeanObjDict = {key: val.mean(axis=0) for key,val in KeypointArrDict.items()}
    VisualizePlot(KeypointArrDict,MeanObjDict)


    pickle.dump(CoordDict,open(os.path.join("./Data/Evaluation/SubjectObjects","GapFilled_%s_hd.p"%BirdID), "wb"))


    return CoordDict, ErrorIndex , OutlierTally

def ReGenerate3DPoint(HeadCoordDictAll, ErrorIndexAll,Points3Dlist,RDict, TDict,Keypoints):
    """Given new object coordinates, with outlier gaps filled, regenerate coordinates in world coordinate system"""

    NewPoints3Dlist = []

    for i in range(len(Points3Dlist)):
        FrameDict = Points3Dlist[i]
        ###Find all BirdIDs in this frame:
        AllIDs = list(set(["_".join(k.split("_")[0:2]) for k in FrameDict.keys()]))
        NoDateIDs = [val.split("_")[0] for val in AllIDs] #ids without the date of trial
        NewFrameDict = {}

        for x in range(len(NoDateIDs)):
            ID = NoDateIDs[x]
            ObjPoints = HeadCoordDictAll[ID][i]
            ##Rotate back to world:
            R = RDict[ID][i]
            inverse_R = np.linalg.inv(R)

            T = TDict[ID][i]

            TempObjPoints = ObjPoints.copy()
            TempObjPoints.pop("index")
            Points3D = np.array(list(TempObjPoints.values()))

            WorldPoints3D = np.add(np.dot(Points3D,inverse_R ),T) ##New world points:
            NewWorldPointDict = {k:WorldPoints3D[index] for index,k in enumerate(list(TempObjPoints.keys()))}


            ##Create fake dict for this frame with same format as Points3Dlist
            for key in Keypoints:
                if key in list(NewWorldPointDict.keys()):

                    NewFrameDict.update({"%s_%s"%(AllIDs[x],key):NewWorldPointDict[key] })
                else:
                    NewFrameDict.update({"%s_%s"%(AllIDs[x],key):FrameDict["%s_%s"%(AllIDs[x],key)]})

        NewPoints3Dlist.append(NewFrameDict)

    return NewPoints3Dlist
            


def GapFillandFilter(Points3Dlist,Keypoints, model):
    """Get object definition of head and bp, filter out points outside of 95% interval, gap fill using mean object definition"""


    BirdMasterDict = PreprocessDict(Points3Dlist,Keypoints)
    RDictHead, TDictHead, HeadCoordList = GetObjectsBird(BirdMasterDict) #R/T dict is organized as: RDictHead[BirdID][ImgIndex]

    AllOutlierTallyHead = 0
    AllFilteredTallyHead = 0
    AllOutlierTallyBack = 0
    AllFilteredTallyBack = 0

    ##Filter/ gap fill by head coord:
    HeadCoordDictAll = {}
    ErrorIndexAll = {}
    for BirdID in RDictHead.keys():
        HeadCoordDict, ErrorIndex, OutlierTally = GetOutliers(BirdID, Keypoints,HeadCoordList)
        HeadCoordDictAll.update({BirdID:HeadCoordDict})
        ErrorIndexAll.update({BirdID:ErrorIndex})

        AllOutlierTallyHead += OutlierTally
        AllFilteredTallyHead += len(ErrorIndex)

    NewPoints3Dlist = ReGenerate3DPoint(HeadCoordDictAll, ErrorIndexAll,Points3Dlist,RDictHead, TDictHead,Keypoints)



    pickle.dump(NewPoints3Dlist,open(os.path.join("./Data/Evaluation/","%s.p"), "wb"))

    print("Total instance gap filled in head: %s"%AllOutlierTallyHead)
    print("Total instance gap filled in Backpack: %s"%AllOutlierTallyBack)
    print("Total instance filtered in head: %s"%AllFilteredTallyHead)
    print("Total instance filtered in head: %s"%AllFilteredTallyBack)

def GetFrameObjectDict(Points3Dlist,Keypoints,ObjectDefinitionList):
    """Master function to get head coordinate of all frames within a sequence"""

    BirdMasterDict = PreprocessDict(Points3Dlist,Keypoints)
    RDictHead, TDictHead, HeadCoordList = GetObjectsBird(BirdMasterDict,ObjectDefinitionList) #R/T dict is organized as: RDictHead[BirdID][ImgIndex]

    return RDictHead, TDictHead, HeadCoordList 

if __name__ == "__main__":
    Points3DPath = "/home/michael/Greti_2023/processed_data/2023-02-24/3B0018C1C4/start_1677247383710228/Data/FilterSmooth_Out3DDict.p"
    Points3Dlist = pickle.load(open(Points3DPath, "rb"))
    Keypoints  =  ["hd_eye_right","hd_eye_left","hd_cheek_left","hd_cheek_right","hd_bill_tip","hd_bill_base"]

    GapFillandFilter(Points3Dlist,Keypoints, model = "muppet3DGapFill")



