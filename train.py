from VOCDataloader import VOCLoader
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
from tensorboardX import SummaryWriter
import shutil
import os

dataset_name = "voc"
model_save_location = "saved_models"

# Creating path to save model summary
log_path = "./tensorboard_summary"
if os.path.exists(log_path):
  shutil.rmtree(log_path)

os.makedirs(log_path)  

if not(os.path.exists(model_save_location)):
  os.makedirs(model_save_location)

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")

def train(train_image_folder, train_annotation_folder, valid_image_folder, valid_annotation_folder,  resume = False, model_path = "", batch_size =  8, num_epochs=1000000000):

  train_dataset = VOCLoader(train_image_folder, train_annotation_folder)
  valid_dataset = VOCLoader(valid_image_folder, valid_annotation_folder)

  actual_classes = len(train_dataset.classes)
  
  
  # Reading in the anchors
  anchors = read_anchors("anchors.txt")
  num_anchors = len(anchors)

  num_classes = (actual_classes + 5)*num_anchors

  # object for writing summary
  writer = SummaryWriter(log_path)

  # See if we have to resume the training or 
  # if we should start from scratch
  yolo_net  = None
  if resume:
    yolo_net = torch.load(model_path)
  else:
    yolo_net = YoloDarknet(num_classes).to(device)
  
  # Dataloader for getting training images
  train_loader = DataLoader(dataset=train_dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = 4)

  # Dataloader for getting training images
  valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = 4)
  
  length_of_training = len(train_loader)
  length_of_validation = len(valid_loader)

  loss_function = YoloLoss()
  opt = optim.Adam(yolo_net.parameters())#0.00025)

  min_train_loss = np.inf
  min_valid_loss = np.inf
  
  for an_epoch in range(num_epochs):
    train_epoch_loss = []
    yolo_net.train()
    for i, (an_image, detector_mask, true_boxes, box_xywh) in tqdm(enumerate(train_loader), total = length_of_training):
      opt.zero_grad()
      output = yolo_net(an_image.to(device))
      conv_output = output.permute(0,2,3,1).reshape(-1,an_image.size()[2]//32,an_image.size()[3]//32,num_anchors, actual_classes + 5)
      '''
      shape of box xy : ?, 7, 7, 5 ,2
      shape of box wh : ?, 7, 7, 5 ,2
      shape of box confidence : ?, 7, 7, 5, 1
      shape of box class_prob : ?, 7, 7, 5, num_classes
      '''
      box_confidence, box_xy, box_wh, box_class_prob = yolo_head(conv_output, anchors, actual_classes)
      train_loss = loss_function(detector_mask.to(device), true_boxes.to(device), conv_output, (box_confidence, box_xy, box_wh, box_class_prob), box_xywh.to(device))
      train_epoch_loss.append(train_loss.item())
      writer.add_scalar('Train/Loss', train_loss.item(), an_epoch * length_of_training + i)
      train_loss.backward()
      opt.step()

    
    print("Training loss is : {}\nMininmum Training Loss is : {}".format(np.mean(train_epoch_loss), min_train_loss))
    if np.mean(train_epoch_loss) < min_train_loss:
      min_train_loss = np.mean(train_epoch_loss)
      torch.save(yolo_net, os.path.join( model_save_location, "yolo_net_train_best_{}_{}.pt".format(dataset_name, min_train_loss)))
    # boxes will be [y1, x1, y2, x2]
    # shape of boxes would be : 1, 7, 7, 5, 4
    # boxes = boxes_to_corners(box_xy, box_wh)
    # scores, classes, boxes = yolo_filter_boxes(box_confidence, box_class_prob, boxes, score_threshold=0.15)
    # image_shape = an_image.size()[2:4]
    # boxes = scale_boxes(boxes, image_shape)
    # print(boxes)
    # draw_on_image(an_image, boxes)

    valid_epoch_loss = []
    yolo_net.eval()
    for i, (an_image, detector_mask, true_boxes, box_xywh) in tqdm(enumerate(valid_loader), total = length_of_validation):
      output = None
      with torch.no_grad():
        output = yolo_net(an_image.to(device))
      conv_output = output.permute(0,2,3,1).reshape(-1,an_image.size()[2]//32,an_image.size()[3]//32,num_anchors, actual_classes + 5)
      '''
      shape of box xy : ?, 7, 7, 5 ,2
      shape of box wh : ?, 7, 7, 5 ,2
      shape of box confidence : ?, 7, 7, 5, 1
      shape of box class_prob : ?, 7, 7, 5, num_classes
      '''
      box_confidence, box_xy, box_wh, box_class_prob = yolo_head(conv_output, anchors, actual_classes)
      valid_loss = loss_function(detector_mask.to(device), true_boxes.to(device), conv_output, (box_confidence, box_xy, box_wh, box_class_prob), box_xywh.to(device))
      valid_epoch_loss.append(valid_loss.item())
      writer.add_scalar('Valid/Loss', valid_loss.item(), an_epoch * length_of_validation + i)
      
    print("validation loss is : {}\nMinimum Validation Loss is {}".format(np.mean(valid_epoch_loss), min_valid_loss ))
    if np.mean(valid_epoch_loss) < min_valid_loss:
      min_valid_loss = np.mean(valid_epoch_loss)
      torch.save(yolo_net, os.path.join(model_save_location, "yolo_net_valid_best_{}_{}.pt".format(dataset_name, min_valid_loss)))
        

def draw_on_image(an_image, boxes):
  an_image = an_image.permute( 0, 2, 3, 1)
  image_to_show = np.interp(an_image[0], (an_image.min(), an_image.max()), (0, 255))
    
  for box in boxes:
    y1, x1, y2, x2 = box
    y1,x1, y2,x2 = int(y1), int(x1), int(y2), int(x2)
    print(y1,x1,y2,x2)
    print(image_to_show.shape)
    cv2.rectangle(image_to_show,(x1, y1), (x2, y2), (0,0,255), 2)
  cv2.imwrite("im.jpg", image_to_show)

if __name__ == "__main__":
  train_image_folder = ""
  train_annotation_folder = ""
  
  valid_image_folder = ""
  valid_annotation_folder = ""
  
  resume = False
  model_path = ""

  train(train_image_folder, train_annotation_folder, valid_image_folder, valid_annotation_folder, resume, model_path)