import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf

from imageai.Detection import ObjectDetection

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(
    execution_path, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(
    # input_image=os.path.join(execution_path, "1990_0223_cp", "_Color_4381.png"),
    input_image=os.path.join(execution_path, "pig.png",),
    output_image_path=os.path.join(execution_path, "imagenew.jpg"))

detections


for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])
