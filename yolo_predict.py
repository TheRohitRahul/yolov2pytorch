from ImagePredictLoader import ImagePredictLoader
from yolo_network import YoloDarknet
from torch.utils.data import DataLoader
from tqdm import tqdm
from yolo_head import yolo_head
from anchors_helper import read_anchors
from yolo_utils import boxes_to_corners, scale_boxes, yolo_filter_boxes
from yolo_loss import YoloLoss
import cv2
import numpy as np
import torch.optim as optim
import torch
import os
import shutil

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")

def predict(image_folder, model_path, batch_size = 1, prediction_folder = ""):
  dataset = ImagePredictLoader(image_folder)
  actual_classes = len(dataset.classes)  
  
  # Reading in the anchors
  anchors = read_anchors("anchors.txt")
  num_anchors = len(anchors)

  num_classes = (actual_classes + 5)*num_anchors

  yolo_net = torch.load(model_path)
  predict_loader = DataLoader(dataset=dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = 4)
  
  length_of_dataset = len(predict_loader)
  yolo_net.eval()
  for i, an_image in tqdm(enumerate(predict_loader), total = length_of_dataset):
      output = None
      with torch.no_grad():
        output = yolo_net(an_image.to(device))
      conv_output = output.permute(0,2,3,1).reshape(-1,an_image.size()[2]//32,an_image.size()[3]//32,num_anchors, actual_classes + 5)
      box_confidence, box_xy, box_wh, box_class_prob = yolo_head(conv_output, anchors, actual_classes)
      
      boxes = boxes_to_corners(box_xy, box_wh)
      scores, classes, boxes = yolo_filter_boxes(box_confidence, box_class_prob, boxes, score_threshold=0.5)

      named_classes = []
      for a_class in classes:
        named_classes.append(dataset.classes[int(a_class.cpu().data.numpy())])
      image_shape = an_image.size()[2:4]
      boxes = scale_boxes(boxes, image_shape)
      draw_on_image(an_image, boxes, named_classes, i, prediction_folder)
    

def draw_on_image(an_image, boxes, named_classes, name_image, prediction_folder):
  if not(os.path.exists(prediction_folder)):
    os.makedirs(prediction_folder)
  an_image = an_image.permute( 0, 2, 3, 1)
  image_to_show = np.interp(an_image[0], (an_image.min(), an_image.max()), (0, 255))
  all_bounding_boxes = []  
  
  with open(os.path.join(prediction_folder, "{}.txt".format(name_image)), "w") as f:
    f.write("")

  for box, a_class in zip(boxes, named_classes):
    y1, x1, y2, x2 = box

    y1,x1, y2,x2 = int(y1), int(x1), int(y2), int(x2)
    with open(os.path.join(prediction_folder, "{}.txt".format(name_image)), "a") as f:
      f.write("{},{},{},{},{}".format(x1, y1, x2, y2, a_class))

    cv2.rectangle(image_to_show,(x1, y1), (x2, y2), (0,0,255), 2)
    # Writing the name of the class
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
  
    # Blue color in BGR 
    color = (255, 0, 0) 
      
    # Line thickness of 2 px 
    thickness = 2
      
    # Using cv2.putText() method 
    image = cv2.putText(image_to_show, a_class, (x1 - 10, y1 -10), font,  
                      fontScale, color, thickness, cv2.LINE_AA)  
    
  cv2.imwrite(os.path.join(prediction_folder, "{}.jpg".format(name_image)), image_to_show)


if __name__ == "__main__":
  prediction_folder = './prediction_folder'
  image_folder = ""
  model_path = ""
  
  if os.path.exists(prediction_folder):
    shutil.rmtree(prediction_folder)
  predict(image_folder, model_path, 1, prediction_folder)