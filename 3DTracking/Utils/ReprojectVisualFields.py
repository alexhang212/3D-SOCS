"""Visualize 3D dict, with visual fields"""
import cv2 
from tqdm import tqdm
import numpy as np

import sys
sys.path.append("./")


import math
import os
import pickle

from Utils import GetHeadCoordinate


    
def getColor(keyPoint,ColourDictionary):
    for k,v in ColourDictionary.items():
        if keyPoint.endswith(k):
            return v
        else:
            continue

    return (0,165,255)


def IsPointValid(Dim, point):
    """Check if a point is valid, i.e within image frame"""
    #get dimension of screen from config
    Valid = False
    if 0 <= point[0] <= Dim[0] and 0 <= point[1] <= Dim[1]:
        Valid = True
    else:
        return Valid
    return Valid


def PlotLine(MarkerDict,Key1Short,Key2Short,Colour,img):
    """
    Plot a line in opencv between Key1 and Key2
    Input is a dictionary of points
    
    """
    # import ipdb;ipdb.set_trace()
    imsize = (img.shape[1],img.shape[0])


    if Key1Short in MarkerDict and Key2Short in MarkerDict:
        ###no need find name
        Key1Name = [Key1Short]
        Key2Name = [Key2Short]
    else:
        #Get index for given keypoint
        Key1Name = [k for k in list(MarkerDict.keys()) if Key1Short in k]
        Key2Name = [k for k in list(MarkerDict.keys()) if Key2Short in k]

    if len(Key1Name) >0 and len(Key2Name) > 0:
        # Key1Name = Key1Name[0]
        # Key2Name = Key2Name[0]
        Key1Name = Key1Name
        Key2Name = Key2Name
    else:
        return None 
    
    for x in range(len(Key1Name)):
        pt1 = MarkerDict[Key1Name[x]]

        if len(Key2Name) < x+1:
            continue
        pt2 = MarkerDict[Key2Name[x]]

        if not IsPointValid(imsize, pt1) or not IsPointValid(imsize, pt2):
            continue

        if np.isnan(pt1[0]) or math.isinf(pt1[0]) or math.isinf(pt1[1]):
            return None
        elif np.isnan(pt2[0]) or math.isinf(pt2[0]) or math.isinf(pt2[1]):
            return None

        point1 = (round(pt1[0]),round(pt1[1]))
        point2 = (round(pt2[0]),round(pt2[1]))

        cv2.line(img,point1,point2,Colour,2 )


def Process_Crop(Crop, CropSize):
    """Crop image and pad, if too big, will scale down """
    # import ipdb;ipdb.set_trace()
    if Crop.shape[0] > CropSize[0] or Crop.shape[1] > CropSize[1]: #Crop is bigger, scale down
        ScaleProportion = min(CropSize[0]/Crop.shape[0],CropSize[1]/Crop.shape[1])
        
        width_scaled = int(Crop.shape[1] * ScaleProportion)
        height_scaled = int(Crop.shape[0] * ScaleProportion)
        Crop = cv2.resize(Crop, (width_scaled,height_scaled), interpolation=cv2.INTER_LINEAR)  # resize image

        # Points2D = {k:[v[0]*ScaleProportion,v[1]*ScaleProportion] for k,v in Points2D.items()}
    else:
        ScaleProportion = 1
        
    if Crop.shape[0] %2 ==0:
        #Shape is even number
        YPadTop = int((CropSize[1] - Crop.shape[0])/2)
        YPadBot = int((CropSize[1] - Crop.shape[0])/2)
    else:
        YPadTop = int( ((CropSize[1] - Crop.shape[0])/2)-0.5)
        YPadBot = int(((CropSize[1] - Crop.shape[0])/2)+0.5)
    ##Padding:
    if Crop.shape[1] %2 ==0:
        #Shape is even number
        XPadLeft = int((CropSize[0] - Crop.shape[1])/2)
        XPadRight= int((CropSize[0] - Crop.shape[1])/2)
    else:
        XPadLeft =  int(((CropSize[0] - Crop.shape[1])/2)-0.5)
        XPadRight= int(((CropSize[0] - Crop.shape[1])/2)+0.5)



    OutImage = cv2.copyMakeBorder(Crop, YPadTop,YPadBot,XPadLeft,XPadRight,cv2.BORDER_CONSTANT,value=[0,0,0])
    
    return OutImage,ScaleProportion, YPadTop,XPadLeft



def VizualizeAll(frame, counter,CamParams,VisualizeIndex,Boxes,Point3DDict,imsize,ColourDictionary,lines = False):
    """Visualize all on visualize cam"""
    ###Viszualize all
    ##Reproject points

    # import ipdb;ipdb.set_trace()
    PointsDict = {}

    Points3DArr = np.array(list(Point3DDict.values()))
    PointsNames = list(Point3DDict.keys())
    Allimgpts, jac = cv2.projectPoints(Points3DArr, CamParams["R"], CamParams["T"],CamParams["cameraMatrix"], CamParams["distCoeffs"])

    # import ipdb;ipdb.set_trace()
    for i in range(len(Allimgpts)):
        pts = Allimgpts[i]
        if np.isnan(pts[0][0]) or math.isinf(pts[0][0]) or math.isinf(pts[0][1]):
            continue
        #######
        point = (round(pts[0][0]),round(pts[0][1]))
        if IsPointValid(imsize,point):
            colour = getColor(PointsNames[i],ColourDictionary)
            cv2.circle(frame,point,1,colour, 5)
        
        PointsDict.update({PointsNames[i]:point})

    ##Plot BBox:
    if Boxes is not None:
        for box in Boxes:
            # import ipdb;ipdb.set_trace()
            cv2.rectangle(frame,(round(box[0]),round(box[1])),(round(box[2]),round(box[3])),[255,0,0],3)
      
    # import ipdb;ipdb.set_trace()
    IndIDs = list(set([key.split("_")[0] for key in Point3DDict.keys()]))

    if lines:
        for id in IndIDs:
            PlotLine(PointsDict, "%s_hd_eye_left"%id, "%s_hd_eye_right"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_bill_base"%id, "%s_hd_eye_right"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_bill_base"%id, "%s_hd_eye_left"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_bill_base"%id, "%s_hd_bill_tip"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_eye_left"%id, "%s_hd_cheek_left"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_eye_right"%id, "%s_hd_cheek_right"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_cheek_right"%id, "%s_hd_cheek_left"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_beak_base"%id, "%s_hd_eye_right"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_beak_base"%id, "%s_hd_eye_left"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_beak_base"%id, "%s_hd_beak_tip"%id,[0,255,255], frame)

            ##greti:
            PlotLine(PointsDict, "%s_hd_neck_left"%id, "%s_hd_eye_left"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_neck_right"%id, "%s_hd_eye_right"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_neck_left"%id, "%s_hd_cheek_left"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_neck_right"%id, "%s_hd_cheek_right"%id,[0,255,255], frame)
            PlotLine(PointsDict, "%s_hd_neck_right"%id, "%s_hd_neck_left"%id,[0,255,255], frame)

    return frame
    

    
def IsPointValid(Dim, point):
    """Check if a point is valid, i.e within image frame"""
    #get dimension of screen from config
    Valid = False
    if 0 <= point[0] <= Dim[0] and 0 <= point[1] <= Dim[1]:
        Valid = True
    else:
        return Valid
    return Valid


def Visualize3D_VS(Smooth3DDict,DataDir, VidPaths,CamNames,FrameDiffs,startFrame,TotalFrames,VisualizeIndex,ColourDictionary,Extrinsic = "Initial",show=True,save=False,VSAngle=60,ElvAngle=0, Magnitude = 10):
    """
    Given a 3d dictionary output, reprojected back to a view
    Dictionary format:
    {frame: {kp:[x,y,z]}}
    
    """

    # import ipdb;ipdb.set_trace()
    if show:
        # cv2.namedWindow("Window",cv2.WINDOW_NORMAL)
        cv2.namedWindow("Window",cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    counter=startFrame + FrameDiffs[VisualizeIndex]
    # import ipdb;ipdb.set_trace()
    CamParamDict = {}
    for cam in CamNames:
        IntPath = os.path.join(DataDir,"%s_Intrinsic.p"%cam)
        retCal, cameraMatrix, distCoeffs, rvecs, tvecs,pVErr,NewCamMat,roi = pickle.load(open(IntPath,"rb"))

        ExtPath = os.path.join(DataDir,"%s_%s_Extrinsics.p"%(cam,Extrinsic))

        R,T = pickle.load(open(ExtPath,"rb"))

        # CamDict = {"R":cv2.Rodrigues(R)[0],"T":np.array(T,dtype=np.float64),"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs}
        CamDict = {"R":R,"T":np.array(T,dtype=np.float64),"cameraMatrix":cameraMatrix,"distCoeffs":distCoeffs}

        CamParamDict.update({cam:CamDict})

    ##Setup video capture object
    cap = cv2.VideoCapture(VidPaths[VisualizeIndex])
    cap.set(cv2.CAP_PROP_POS_FRAMES,counter) 

    if TotalFrames == -1: #set total frames as max frame of video
        TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    imsize = (int(cap.get(3)),int(cap.get(4)))
    if save:
        TrialName = VidPaths[VisualizeIndex].split("/")[-3]
        out = cv2.VideoWriter(filename="%s_VisualField_sample.mp4"%TrialName, apiPreference=cv2.CAP_FFMPEG, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=15, frameSize = imsize)

    Out3DDict = {}

    for i in tqdm(range(TotalFrames)):
        i = i+startFrame
        # print(i)
        PointsDict = {} #Points Dictionary for this frame

        FrameList = []
        #read frames
        ret, frame = cap.read()
        frame =cv2.undistort(frame, CamParamDict[CamNames[VisualizeIndex]]["cameraMatrix"],CamParamDict[CamNames[VisualizeIndex]]["distCoeffs"])

        # import ipdb;ipdb.set_trace()

        keyList = [list(val.keys()) for val in PointsDict.values()]
        # CombinedList = sum(keyList,[])
        # counts = [CombinedList.count(key) for key in CombinedList]

        if i in Smooth3DDict:
            Point3DDict = Smooth3DDict[i]
            if len(Point3DDict) >0:

                # import ipdb;ipdb.set_trace()
                CamParams = CamParamDict[CamNames[VisualizeIndex]]
            
                IndIDs = list(set([key.split("_")[0] for key in Point3DDict.keys()]))

                for id in IndIDs:
                    if "%s_hd_bill_tip"%id not in Point3DDict or "%s_hd_eye_left"%id not in Point3DDict or "%s_hd_eye_right"%id not in Point3DDict:
                        continue

                    beakpt = np.array(Point3DDict["%s_hd_bill_tip"%id])
                    leftEye = np.array(Point3DDict["%s_hd_eye_left"%id])
                    rightEye = np.array(Point3DDict["%s_hd_eye_right"%id])


                    R,T = GetHeadCoordinate.DefineObj(beakpt,leftEye,rightEye)
                    VSRightWorld, VSLeftWorld, VSFrontWorld= GetHeadCoordinate.GetVisualFieldPoints(R,T ,Angle=VSAngle,ElvAngle = ElvAngle,Magnitude = Magnitude, pigeon=False)
                    PlotVS(frame, CamParams,VSRightWorld,VSLeftWorld,VSFrontWorld,Point3DDict,id)

                frame = VizualizeAll(frame, counter,CamParams,VisualizeIndex,None,Point3DDict,imsize,ColourDictionary,lines=True)


        if show:
            cv2.imshow('Window',frame)
        # import ipdb;ipdb.set_trace()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if save:
            out.write(frame)
        counter += 1

    cap.release()
    if save:
        out.release()
    cv2.destroyAllWindows()

    return Out3DDict


def GetMidPoint(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2,(p1[2]+p2[2])/2]

def PlotVS(frame, CamParams,VSRightWorld,VSLeftWorld,VSFrontWorld,Point3DDict,bird):
    """Plot VS points"""

    MidEye = GetMidPoint(Point3DDict["%s_hd_eye_right"%bird],Point3DDict["%s_hd_eye_left"%bird])
    Points3DArr = np.array([VSRightWorld,Point3DDict["%s_hd_eye_right"%bird],VSLeftWorld,Point3DDict["%s_hd_eye_left"%bird],VSFrontWorld,MidEye])
    Allimgpts, jac = cv2.projectPoints(Points3DArr, CamParams["R"], CamParams["T"], 
                                       CamParams["cameraMatrix"], CamParams["distCoeffs"])

    # import ipdb;ipdb.set_trace()

    points = [[round(pt[0][0]), round(pt[0][1])] for pt in Allimgpts]

    # print(points)

    try:
        cv2.line(frame, points[0], points[5], [0,255,0], 2)
        cv2.line(frame, points[2], points[5], [0,0,255], 2) 
        cv2.line(frame, points[4], points[5], [255,0,0], 2) 
    except:
        pass
