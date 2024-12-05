"""Train DLC Model for 3D-SOCS"""

import sys
sys.path.append("Repositories/DeepLabCut/")  # Add DeepLabCut repository to the system path

import deeplabcut as dlc
import os
import numpy as np
import argparse

def Train_DLC(DLC_ConfigPath, BatchSize=32, Optimizer="adam"):
    """
    Function to train a DeepLabCut model.
    
    Parameters:
    DLC_ConfigPath (str): Path to the DeepLabCut config file.
    BatchSize (int): Batch size for training.
    Optimizer (str): Optimizer to use for training.
    """

    # Use mergeandsplit to create train/val split
    trainIndex, testIndex = dlc.mergeandsplit(DLC_ConfigPath, trainindex=0, uniform=False)
    
    # Create training dataset
    TrainOut = dlc.create_training_dataset(DLC_ConfigPath, Shuffles=[1], trainIndices=[trainIndex], testIndices=[testIndex], augmenter_type='imgaug')
    dlc.auxiliaryfunctions.edit_config(DLC_ConfigPath, {"TrainingFraction": [TrainOut[0][0]]})

    # Get the path to the training configuration file
    train_pose_config, _, _ = dlc.return_train_network_path(DLC_ConfigPath)
    
    # Define multi-step learning rate schedule
    MultiStep = [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 30000], [1e-5, 100000]]  # Multi-step from Jonathan
    
    # Edit the training configuration with batch size, optimizer, and multi-step schedule
    dlc.auxiliaryfunctions.edit_config(train_pose_config, {'batch_size': BatchSize,
                                                           "optimizer": Optimizer,
                                                           'multi_step': MultiStep})

    # Train the network
    dlc.train_network(DLC_ConfigPath, saveiters=1000, max_snapshots_to_keep=1000)

    # Edit the configuration to evaluate all snapshots
    dlc.auxiliaryfunctions.edit_config(DLC_ConfigPath, {'snapshotindex': "all"})

    # Evaluate the trained network
    dlc.evaluate_network(DLC_ConfigPath, Shuffles=[1])

    # Export the trained model
    dlc.export_model(DLC_ConfigPath)


if __name__ == "__main__":
    # Specify absolute path to DLC folder and batch size
    Path = "/home/alexchan/Documents/3D-SOCS/SampleDataset/KPDataset/DLC_Dataset"
    Batch = 32

    # Construct the path to the DLC config file
    DLC_ConfigPath = os.path.join(Path, "config.yaml")
    
    # Train the DLC model
    Train_DLC(DLC_ConfigPath, BatchSize=Batch, Optimizer="adam")