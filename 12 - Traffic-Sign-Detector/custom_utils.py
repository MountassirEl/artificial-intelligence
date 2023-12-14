import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from config import CLASSES, OUT_DIR



# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer
        ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            """
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(OUT_DIR, 'best_model.pth'))
            """
            torch.save(model, os.path.join(OUT_DIR, 'best_model.pth'))


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# define the training tranforms
def get_train_transform():
    return A.Compose([
        # A.Flip(0.5),
        # A.RandomRotate90(0.5),
        # A.MotionBlur(p=0.2),
        # A.MedianBlur(blur_limit=3, p=0.1),
        # A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


# define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })


def show_tranformed_image(train_loader):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            for image, target in zip(images, targets):
                im = image.cpu().numpy().transpose(1, 2, 0)
                im = np.array(im * 255, dtype = np.uint8)
                boxes = target["boxes"].to(torch.int).numpy()
                labels = target["labels"].numpy()
                visualize(im, boxes, labels, CLASSES)
                break; # visualize one picture
            
def show_inference_results(image, model):
    """
    """
    inputs = [np.array(image).transpose(2, 0, 1) / 255.0]
    inputs = [torch.FloatTensor(im) for im in inputs]
    inputs = [im.to("cpu") for im in inputs]
    outputs = model(inputs)
    for output in outputs:
        boxes = output["boxes"].detach().to(torch.int).numpy()
        scores = output["scores"].detach().numpy()
        labels = output["labels"].detach().numpy()
        visualize(image, boxes, labels, CLASSES)


def visualize_bbox(img, bbox, class_name, thickness=1):
    """Visualizes a single bounding box on the image"""
    BOX_COLOR = (255, 0, 0)      # Red
    TEXT_COLOR = (255, 255, 255) # White
    x_min, y_min, x_max, y_max = np.array(bbox, dtype = int)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.25 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.3, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(OUT_DIR, 'last_model.pth'))
    """
    torch.save(model, os.path.join(OUT_DIR, 'last_model.pth'))


def save_loss_plot(OUT_DIR, train_loss_list, val_loss_list):
    fig = plt.figure()
    plt.plot(range(1, len(train_loss_list)+1), train_loss_list, color='tab:blue', label = "train")
    plt.plot(range(1, len(val_loss_list)+1), val_loss_list, color='tab:red', label = "val")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    fig.savefig(f"{OUT_DIR}/loss.png")
    plt.show()
    plt.close('all')
    print('SAVING PLOTS COMPLETE...')
    
def annot_lines_to_dict(lines: list):
    
    file_names = list()
    annotations = list()
    
    for line in lines:
        file_name, xmin, ymin, xmax, ymax, label = line.split(";")
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        if file_name in file_names:
            idx = file_names.index(file_name)
            annotations[idx]["boxes"].append([xmin, ymin, xmax, ymax])
            annotations[idx]["labels"].append(int(label) + 1) # +1 because we don't have a background (0) class 
        else:
            file_names.append(file_name)
            annotations.append({
                "file_name": file_name,
                "boxes": [[xmin, ymin, xmax, ymax]],
                "labels": [int(label)+1] # +1 because we don't have a background (0) class 
            })
    return annotations
    
