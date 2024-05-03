"""Train YOLO for pigeons"""

from ultralytics import YOLO
import argparse


def TrainYOLO(PreTrained, Config):
    model = YOLO(PreTrained)

    model.train(data=Config, batch=-1,epochs=100)  # train the model
    model.val()
    model.export()


if __name__ == "__main__":
    PreTrained = "./yolov8l.pt"
    Config = "./NetworkTraining/Greti_YOLO.yaml"
    TrainYOLO(PreTrained, Config)