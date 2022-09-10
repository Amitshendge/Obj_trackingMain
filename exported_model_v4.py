import argparse
import pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow

PROB_THRESHOLD = 0.7  # Minimum probably to show results.
dir_path="V4TensorFlowModel"
label_file=open(r"V4TensorFlowModel\labels.txt","r")
labels=label_file.read().split("\n")
# image=Image.open("frame.jpg")
# h,w,ch=np.array(image).shape
# print(labels)
class Model:
    def __init__(self):
        model = tensorflow.saved_model.load(str(dir_path))
        self.serve = model.signatures['serving_default']
        self.input_shape = self.serve.inputs[0].shape[1:3]

    def predict(self,param):
        input_tensor = tensorflow.convert_to_tensor(param)
        outputs = self.serve(input_tensor)
        return(print_outputs({k: v[np.newaxis, ...] for k, v in outputs.items()}))


def print_outputs(outputs):
    boxes=[]
    classes=[]
    scores=[]
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
        if score > PROB_THRESHOLD:
            # print(f"Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
            boxes.append((float(box[0]), float(box[1]),float(box[2]), float(box[3])))
            classes.append(labels[int(class_id)])
            scores.append(float(score))
    return(boxes,classes,scores)
