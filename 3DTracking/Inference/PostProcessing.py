"""Do Post processing"""

import os
import pickle
import numpy as np
import pandas as pd

import sys
sys.path.append("Utils/")
import PostProcessFunctions as PP



if __name__ == "__main__":
    InputDir = "../SampleDataset/3DTracking/bird_3B00186CDA_trial_32_time_2023-11-05 10_30_28.117871/"
    DataDir = "../SampleDataset/Calibration/data"
    SubDataDir = os.path.join(InputDir,"data")

    CamNames = ["mill1","mill2","mill3","mill4","mill5","mill6"]
    imsizeDict = {"mill1":(1920,1080),"mill2":(1280,720),"mill3":(1280,720),
                  "mill4":(1280,720),"mill5":(1280,720),"mill6":(1280,720)}

    ###Load inference files
    Out3DDict = pickle.load(open(os.path.join(SubDataDir,"Out3DDict.p"),"rb"))
    Out2DDict = pickle.load(open(os.path.join(SubDataDir,"Filtered2DDict.p"),"rb"))

    ###Filter reproject out
    FilterReprojectOut = PP.FilterbyReproject(Out3DDict,DataDir,CamNames,imsizeDict,Ratio = 0.5,Extrinsic = "Initial")

    ###Filter by distance
    HeadPoints = ["hd_eye_right","hd_eye_left","hd_cheek_left","hd_cheek_right","hd_bill_tip","hd_bill_base", "hd_neck_left", "hd_neck_right"]
    BodyPoints = [ "bd_bar_left_front", "bd_bar_left_back","bd_bar_right_front","bd_bar_right_back", "bd_tail_base","bd_keel_top","bd_tail_tip","bd_shoulder_left","bd_shoulder_right"]
    Filtered_3DDict = PP.FilterPoints(FilterReprojectOut,HeadPoints)
    Filtered_3DDict = PP.FilterPoints(Filtered_3DDict,BodyPoints)
    
    #Interpolate 1
    Linear3DDict = PP.InterpolateData(Filtered_3DDict,Type="pchip",interval = 1, show=False)

    pickle.dump(Linear3DDict,open(os.path.join(SubDataDir,"MultiFiltered_Out3DDict.p"),"wb"))

    ####Object Filter
    ###Dumping original to iteratively update it for multi individual
    pickle.dump(Linear3DDict,open(os.path.join(SubDataDir,"MultiObjectFiltered_Out3DDict.p"),"wb"))
    Out3DDictKeys = [list(subDict.keys()) for subDict in Linear3DDict.values()]
    Out3DDictKeysFlat = []
    [Out3DDictKeysFlat.extend(x) for x in Out3DDictKeys]
    UnqIDs = sorted(list(set([x.split("_")[0] for x in list(set(Out3DDictKeysFlat))])))
    SpeciesObjectDict = pickle.load(open(os.path.join(InputDir,"MedianSpeciesObjects.p"),"rb"))


    # import ipdb;ipdb.set_trace()
    for IndID in UnqIDs:
        Linear3DDict = pickle.load(open(os.path.join(SubDataDir,"MultiObjectFiltered_Out3DDict.p"), "rb"))

        HeadFilteredDictRotation = PP.FilterbyObject(Linear3DDict,SpeciesObjectDict["GRETI"]["hd"],HeadPoints,IndID, ObjFilterThresh =50)
        BodyFilteredDictRotation = PP.FilterbyObject(HeadFilteredDictRotation,SpeciesObjectDict["GRETI"]["bd"],BodyPoints,IndID, ObjFilterThresh =50)

        pickle.dump(BodyFilteredDictRotation,open(os.path.join(SubDataDir,"MultiObjectFiltered_Out3DDict.p"),"wb"))


    FinalOut3DDict = PP.InterpolateData(BodyFilteredDictRotation,Type="pchip",interval = 1, show=False)

    pickle.dump(FinalOut3DDict,open(os.path.join(SubDataDir,"Out3DDictProcessed.p"),"wb"))
