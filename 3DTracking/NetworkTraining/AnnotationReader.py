"""Object to Read JSON output from label-studio"""

import sys
import os
sys.path.append("./")
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def getColor(keyPoint):
    return (0,255,0)

def IsPointInBox(bbox, point):
    """Check if a point is within bounding box, i.e within image frame"""
    #get dimension of screen from config
    Valid = False
    if bbox[0] <= point[0] <= bbox[0]+bbox[2] and bbox[1] <= point[1] <= bbox[1]+bbox[3]:
        Valid = True
    else:
        return Valid
    return Valid

class LS_JSONReader:
    
    def __init__(self, DatasetPath,JSONPath):
        """
        Initialize JSON reader object
        JSONPath: path to json file
        DatasetPath: Path to dataset root directory to read images
        Type: 2D or 3D, based om which type was read     
        """
        with open(JSONPath) as f:
            self.data = json.load(f)
        # import ipdb;ipdb.set_trace()

        self.DatasetPath = DatasetPath

    def GetBirdIDs(self,index):
        """Get arbitiary IDs for multiple birds in frame"""
        BBoxLabels = self.data[index]["label"]
        NumInd = len(BBoxLabels) #number of individuals

        GretiIndex = 0
        BlutiIndex = 0
        HeadIndex = 0

        IndList = []
        for x in range(NumInd):
            label = BBoxLabels[x]["rectanglelabels"][0]
            if label == "greti":
                IndList.append(label+"_"+str(GretiIndex))
                GretiIndex  += 1
            elif label == "bluti":
                IndList.append(label+"_"+str(BlutiIndex))
                BlutiIndex  += 1
            elif label == "head":
                IndList.append(label+"_"+str(HeadIndex))
                HeadIndex  += 1

        return IndList

    def Extract2D(self,index):
        """Extract 2D data"""

        BirdIDs = self.GetBirdIDs(index)
        # import ipdb;ipdb.set_trace()
        KPData = self.data[index]["kp-1"]

        KeypointDict = {}
        # import ipdb;ipdb.set_trace()

        if len(BirdIDs)>1:
            BBoxDict = self.ExtractBBox(index)

            for bird in BirdIDs:
                BirdDict = {}
                for kp in KPData:
                    point = [(kp["x"]/100)*kp["original_width"],(kp["y"]/100)*kp["original_height"]]
                    if IsPointInBox(BBoxDict[bird],point):
                        BirdDict.update({kp["keypointlabels"][0]:point})
                    else:
                        continue
                KeypointDict.update({bird:BirdDict})
        else:
            for bird in BirdIDs:
                BirdDict = {}
                for kp in KPData:
                    BirdDict.update({kp["keypointlabels"][0]:[(kp["x"]/100)*kp["original_width"],
                                                                (kp["y"]/100)*kp["original_height"]]})

                KeypointDict.update({bird:BirdDict})
            
        return KeypointDict

    def ExtractBBox(self,index):
        """Extract bbox, xyxy format"""
        # import ipdb;ipdb.set_trace()
        BirdIDs = self.GetBirdIDs(index)
        BBoxData = self.data[index]["label"]
        
        BBoxDict = {}
        for x in range(len(BirdIDs)):
            XRatio = BBoxData[x]["original_width"]/100
            YRatio = BBoxData[x]["original_height"]/100

            bbox = [BBoxData[x]["x"]*XRatio,BBoxData[x]["y"]*YRatio,
                    BBoxData[x]["x"]*XRatio+BBoxData[x]["width"]*XRatio,BBoxData[x]["y"]*YRatio+BBoxData[x]["height"]*YRatio]
            BBoxDict.update({BirdIDs[x]:bbox})
        
        return BBoxDict
    
    def ExtractHeadBBox(self,index, scale = 1.5):
        """Extract bbox,but only for the head xyxy format"""
        # import ipdb;ipdb.set_trace()
        BirdIDs = self.GetBirdIDs(index)
        Points2D = self.Extract2D(index)
        BBoxData = self.data[index]["label"]
        # import ipdb;ipdb.set_trace()

        BBoxDict = {}
        for x in range(len(BirdIDs)):

            HeadPoints = [pt for k,pt in Points2D[BirdIDs[x]].items() if k.startswith("hd") or k.startswith("bd_keel_top")]
            if len(HeadPoints) < 3:
                continue

            XVals = [pt[0] for pt in HeadPoints]
            YVals = [pt[1] for pt in HeadPoints]

            LeftMost = min(XVals)
            RightMost = max(XVals)
            TopMost = min(YVals)
            BotMost = max(YVals)
            width = RightMost-LeftMost
            height = BotMost-TopMost
            NewWidth = width*scale
            NewHeight = height*scale

            if LeftMost-((NewWidth-width)/2) <0:
                x1=0
            else:
                x1= LeftMost-((NewWidth-width)/2)
            
            if TopMost-((NewHeight-height)/2) <0:
                y1=0
            else:
                y1=TopMost-((NewHeight-height)/2)

            if RightMost+((NewWidth-width)/2) > BBoxData[x]["original_width"]:
                x2 = BBoxData[x]["original_width"]
            else:
                x2 = RightMost+((NewWidth-width)/2)

            if BotMost+((NewHeight-height)/2) > BBoxData[x]["original_height"]:
                y2 = BBoxData[x]["original_height"]
            else:
                y2 = BotMost+((NewHeight-height)/2)

            bbox = [x1,y1,x2,y2]

            BBoxDict.update({BirdIDs[x]:bbox})
        
        return BBoxDict
    
    def GetImagePath(self,index):
        # import ipdb;ipdb.set_trace()
        if "img" in self.data[index]:
            Outpath = self.data[index]["img"][1:]#remove first slash
        elif "image" in self.data[index]:
            Outpath = self.data[index]["image"][1:]

        return Outpath 

    def CheckBoxAnnotations(self, index, show=True):
        ImgPath = self.GetImagePath(index)
        Key2D = self.Extract2D(index)
        # BBox = self.ExtractHeadBBox(index)
        BBox = self.ExtractBBox(index)
        # import ipdb;ipdb.set_trace()
        if len(BBox) == 0:
            return False
        
        RealImgPath = os.path.join(self.DatasetPath,ImgPath) #removes first slash in ImgPath

        img = cv2.imread(RealImgPath)
        
        ##Draw keypoints:
        # import ipdb;ipdb.set_trace()
        for BirdID,Key2DDict in Key2D.items():
            for key, pts in Key2DDict.items():
                point = (round(pts[0]),round(pts[1]))
                colour = getColor(key)
                cv2.circle(img,point,3,colour, -1)
            BBoxData = BBox[BirdID]
            cv2.rectangle(img,(round(BBoxData[0]),round(BBoxData[1])),(round(BBoxData[2]),round(BBoxData[3])),(255,0,0),3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, BirdID,(round(BBoxData[0]),round(BBoxData[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            cv2.putText(img, BirdID,(round(pts[0]),round(pts[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
 
        if show:
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img

    
    def CheckAnnotations(self, index, show=True):
        ImgPath = self.GetImagePath(index)
        Key2D = self.Extract2D(index)
        # BBox = self.ExtractHeadBBox(index)
        BBox = self.ExtractBBox(index)

        # import ipdb;ipdb.set_trace()
        if len(BBox) == 0:
            return False
        
        RealImgPath = os.path.join(self.DatasetPath,ImgPath) #removes first slash in ImgPath

        img = cv2.imread(RealImgPath)
        
        ##Draw keypoints:
        # import ipdb;ipdb.set_trace()
        for BirdID,Key2DDict in Key2D.items():
            for key, pts in Key2DDict.items():
                point = (round(pts[0]),round(pts[1]))
                colour = getColor(key)
                cv2.circle(img,point,3,colour, -1)
            BBoxData = BBox[BirdID]
            cv2.rectangle(img,(round(BBoxData[0]),round(BBoxData[1])),(round(BBoxData[2]),round(BBoxData[3])),(255,0,0),3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, BirdID,(round(BBoxData[0]),round(BBoxData[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            cv2.putText(img, BirdID,(round(pts[0]),round(pts[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if show:
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img




if __name__ == "__main__":
    DatasetPath = "../SampleDataset/Dataset"
    JSONPath = "../SampleDataset/Dataset/FullDataset.json"

    Reader = LS_JSONReader(DatasetPath,JSONPath)
    print(len(Reader.data))

    ###10 as an example:
    print("Dataset Path:")
    print(os.path.join(DatasetPath,Reader.GetImagePath(10)))
    print("\n2D Keypoints:")
    print(Reader.Extract2D(10))
    print("\nBounding Box:")
    print(Reader.ExtractBBox(10))

    ##Visualize anntations:
    for i in range(0,len(Reader.data)):
        print(i)
        Reader.CheckAnnotations(i)
