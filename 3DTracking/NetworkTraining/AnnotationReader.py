"""Object to Read JSON output from label-studio"""

import sys
import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Add the current directory to the system path
sys.path.append("./")

def getColor(keyPoint):
    """Return color for keypoints"""
    return (0, 255, 0)

def IsPointInBox(bbox, point):
    """Check if a point is within bounding box, i.e within image frame"""
    return bbox[0] <= point[0] <= bbox[0] + bbox[2] and bbox[1] <= point[1] <= bbox[1] + bbox[3]

class LS_JSONReader:
    
    def __init__(self, DatasetPath, JSONPath):
        """
        Initialize JSON reader object
        JSONPath: path to json file
        DatasetPath: Path to dataset root directory to read images
        """
        with open(JSONPath) as f:
            self.data = json.load(f)
        self.DatasetPath = DatasetPath

    def GetBirdIDs(self, index):
        """Get arbitrary IDs for multiple birds in frame"""
        BBoxLabels = self.data[index]["label"]
        NumInd = len(BBoxLabels)  # number of individuals

        GretiIndex = 0
        BlutiIndex = 0
        HeadIndex = 0

        IndList = []
        for x in range(NumInd):
            label = BBoxLabels[x]["rectanglelabels"][0]
            if label == "greti":
                IndList.append(label + "_" + str(GretiIndex))
                GretiIndex += 1
            elif label == "bluti":
                IndList.append(label + "_" + str(BlutiIndex))
                BlutiIndex += 1
            elif label == "head":
                IndList.append(label + "_" + str(HeadIndex))
                HeadIndex += 1

        return IndList

    def Extract2D(self, index):
        """Extract 2D data"""
        BirdIDs = self.GetBirdIDs(index)
        KPData = self.data[index]["kp-1"]

        KeypointDict = {}

        if len(BirdIDs) > 1:
            BBoxDict = self.ExtractBBox(index)

            for bird in BirdIDs:
                BirdDict = {}
                for kp in KPData:
                    point = [(kp["x"] / 100) * kp["original_width"], (kp["y"] / 100) * kp["original_height"]]
                    if IsPointInBox(BBoxDict[bird], point):
                        BirdDict.update({kp["keypointlabels"][0]: point})
                KeypointDict.update({bird: BirdDict})
        else:
            for bird in BirdIDs:
                BirdDict = {}
                for kp in KPData:
                    BirdDict.update({kp["keypointlabels"][0]: [(kp["x"] / 100) * kp["original_width"],
                                                               (kp["y"] / 100) * kp["original_height"]]})
                KeypointDict.update({bird: BirdDict})
            
        return KeypointDict

    def ExtractBBox(self, index):
        """Extract bounding box in xyxy format"""
        BirdIDs = self.GetBirdIDs(index)
        BBoxData = self.data[index]["label"]
        
        BBoxDict = {}
        for x in range(len(BirdIDs)):
            XRatio = BBoxData[x]["original_width"] / 100
            YRatio = BBoxData[x]["original_height"] / 100

            bbox = [BBoxData[x]["x"] * XRatio, BBoxData[x]["y"] * YRatio,
                    BBoxData[x]["x"] * XRatio + BBoxData[x]["width"] * XRatio,
                    BBoxData[x]["y"] * YRatio + BBoxData[x]["height"] * YRatio]
            BBoxDict.update({BirdIDs[x]: bbox})
        
        return BBoxDict
    
    def ExtractHeadBBox(self, index, scale=1.5):
        """Extract bounding box for the head in xyxy format"""
        BirdIDs = self.GetBirdIDs(index)
        Points2D = self.Extract2D(index)
        BBoxData = self.data[index]["label"]

        BBoxDict = {}
        for x in range(len(BirdIDs)):
            HeadPoints = [pt for k, pt in Points2D[BirdIDs[x]].items() if k.startswith("hd") or k.startswith("bd_keel_top")]
            if len(HeadPoints) < 3:
                continue

            XVals = [pt[0] for pt in HeadPoints]
            YVals = [pt[1] for pt in HeadPoints]

            LeftMost = min(XVals)
            RightMost = max(XVals)
            TopMost = min(YVals)
            BotMost = max(YVals)
            width = RightMost - LeftMost
            height = BotMost - TopMost
            NewWidth = width * scale
            NewHeight = height * scale

            x1 = max(0, LeftMost - ((NewWidth - width) / 2))
            y1 = max(0, TopMost - ((NewHeight - height) / 2))
            x2 = min(BBoxData[x]["original_width"], RightMost + ((NewWidth - width) / 2))
            y2 = min(BBoxData[x]["original_height"], BotMost + ((NewHeight - height) / 2))

            bbox = [x1, y1, x2, y2]
            BBoxDict.update({BirdIDs[x]: bbox})
        
        return BBoxDict
    
    def GetImagePath(self, index):
        """Get image path from JSON data"""
        if "img" in self.data[index]:
            Outpath = self.data[index]["img"][1:]  # remove first slash
        elif "image" in self.data[index]:
            Outpath = self.data[index]["image"][1:]
        return Outpath 

    def CheckBoxAnnotations(self, index, show=True):
        """Check and visualize bounding box annotations"""
        ImgPath = self.GetImagePath(index)
        Key2D = self.Extract2D(index)
        BBox = self.ExtractBBox(index)

        if len(BBox) == 0:
            return False
        
        RealImgPath = os.path.join(self.DatasetPath, ImgPath)
        img = cv2.imread(RealImgPath)
        
        for BirdID, Key2DDict in Key2D.items():
            for key, pts in Key2DDict.items():
                point = (round(pts[0]), round(pts[1]))
                colour = getColor(key)
                cv2.circle(img, point, 3, colour, -1)
            BBoxData = BBox[BirdID]
            cv2.rectangle(img, (round(BBoxData[0]), round(BBoxData[1])), (round(BBoxData[2]), round(BBoxData[3])), (255, 0, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, BirdID, (round(BBoxData[0]), round(BBoxData[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, BirdID, (round(pts[0]), round(pts[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
 
        if show:
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img

    def CheckAnnotations(self, index, show=True):
        """Check and visualize annotations"""
        ImgPath = self.GetImagePath(index)
        Key2D = self.Extract2D(index)
        BBox = self.ExtractBBox(index)

        if len(BBox) == 0:
            return False
        
        RealImgPath = os.path.join(self.DatasetPath, ImgPath)
        img = cv2.imread(RealImgPath)
        
        for BirdID, Key2DDict in Key2D.items():
            for key, pts in Key2DDict.items():
                point = (round(pts[0]), round(pts[1]))
                colour = getColor(key)
                cv2.circle(img, point, 3, colour, -1)
            BBoxData = BBox[BirdID]
            cv2.rectangle(img, (round(BBoxData[0]), round(BBoxData[1])), (round(BBoxData[2]), round(BBoxData[3])), (255, 0, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, BirdID, (round(BBoxData[0]), round(BBoxData[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, BirdID, (round(pts[0]), round(pts[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if show:
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img

if __name__ == "__main__":
    DatasetPath = "../SampleDataset/Dataset"
    JSONPath = "../SampleDataset/Dataset/FullDataset.json"

    Reader = LS_JSONReader(DatasetPath, JSONPath)
    print(len(Reader.data))

    # Example usage with index 10
    print("Dataset Path:")
    print(os.path.join(DatasetPath, Reader.GetImagePath(10)))
    print("\n2D Keypoints:")
    print(Reader.Extract2D(10))
    print("\nBounding Box:")
    print(Reader.ExtractBBox(10))

    # Visualize annotations
    for i in range(len(Reader.data)):
        print(i)
        Reader.CheckAnnotations(i)
