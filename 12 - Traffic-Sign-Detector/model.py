import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import CLASSES

def create_model(num_classes: int):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes = num_classes
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # return the model
    return model