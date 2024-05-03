"""Train DLC Model"""

import sys
sys.path.append("Repositories/DeepLabCut/")

import deeplabcut as dlc
import sys
import os
import numpy as np
import argparse

def Train_DLC(DLC_ConfigPath, BatchSize = 32, Optimizer = "adam"):

    ###Use mergeandsplit to create train val split
    trainIndex, testIndex = dlc.mergeandsplit(DLC_ConfigPath, trainindex=0, uniform=False)
    
    TrainOut = dlc.create_training_dataset(DLC_ConfigPath,Shuffles = [1],trainIndices=[trainIndex],testIndices=[testIndex], augmenter_type='imgaug')
    dlc.auxiliaryfunctions.edit_config(DLC_ConfigPath,{"TrainingFraction": [TrainOut[0][0]]})



    train_pose_config, _, _ = dlc.return_train_network_path(DLC_ConfigPath)
    
    MultiStep = [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 30000],  [1e-5, 100000]] ##multistep from jonathan
    
    dlc.auxiliaryfunctions.edit_config(train_pose_config, {'batch_size': BatchSize,
                                                           "optimizer":Optimizer,
                                                           'multi_step':MultiStep}, )



    dlc.train_network(DLC_ConfigPath, saveiters=1000,max_snapshots_to_keep=1000)

    dlc.auxiliaryfunctions.edit_config(DLC_ConfigPath, {'snapshotindex': "all"}, )

    dlc.evaluate_network(DLC_ConfigPath,Shuffles=[1])

    dlc.export_model(DLC_ConfigPath)


if __name__ == "__main__":
    ##Specify absolute path to DLC folder and batch size!!
    Path = "/home/alexchan/Documents/3D-SOCS/SampleDataset/Dataset/DLC_Dataset"
    Batch = 32

    DLC_ConfigPath = os.path.join(Path,"config.yaml")
    Train_DLC(DLC_ConfigPath, BatchSize = Batch, Optimizer = "adam")