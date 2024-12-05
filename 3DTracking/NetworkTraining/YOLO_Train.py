"""Train YOLO for 3D-SOCS"""

from ultralytics import YOLO
import argparse

def TrainYOLO(PreTrained, Config):
    """
    Function to train the YOLO model.
    
    Parameters:
    PreTrained (str): Path to the pre-trained YOLO model.
    Config (str): Path to the configuration file for training.
    """
    model = YOLO(PreTrained)  # Load the pre-trained YOLO model

    # Train the model with the given configuration
    model.train(data=Config, batch=-1, epochs=100)
    
    # Validate the model
    model.val()
    
    # Export the trained model
    model.export()

if __name__ == "__main__":
    # Path to the pre-trained YOLO model
    PreTrained = "./yolov8l.pt"
    
    # Path to the configuration file for training
    Config = "./NetworkTraining/Greti_YOLO.yaml"
    
    # Call the training function
    TrainYOLO(PreTrained, Config)