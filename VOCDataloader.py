from parse_voc_dataset import parse_xml
import os
import cv2
import numpy as np
import torch
from preprocess_gt import preprocess_label
from anchors_helper import read_anchors

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

class VOCLoader(object):
  
  def __init__(self, images_folder, annotation_folder):
    all_image_names = os.listdir(images_folder)
    
    self.classes = ["aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor", "person", "bird", "cat", "cow", "dog", "horse", "sheep"]
    self.anchors = read_anchors("anchors.txt")
    self.image_size = 416
    self.grid_size = 13
    

    self.one_hot = np.identity(len(self.classes))
    self.annotation_files = []
    self.image_files = []
    self.num_allowed_objects = 80

    for image_name in all_image_names:
      if image_name.endswith(".jpg") or image_name.endswith(".png"):
        annotation_path = os.path.join(annotation_folder, os.path.splitext(image_name)[0] + ".xml")
        if os.path.exists(annotation_path):
          self.image_files.append(os.path.join(images_folder, image_name))
          self.annotation_files.append(annotation_path)
  
  def __len__(self):
    return len(self.image_files)
  
  def __getitem__(self, index):
    image_path = self.image_files[index]
    image = cv2.imread(image_path)

    annotation_path = self.annotation_files[index]
    annotations = parse_xml(annotation_path)
    
    original_height, original_width = image.shape[:2]

    if original_height > self.image_size or original_width > self.image_size:
      if original_height > original_width:
        image = image_resize(image, width = None, height = self.image_size, inter = cv2.INTER_AREA)
      else:
        image = image_resize(image, width = self.image_size, height = None , inter = cv2.INTER_AREA)

      # Now we have to change the xml annotations also as the size of the image is changed
      mod_height, mod_width = image.shape[:2]

      for i in range(len(annotations)):
        x1, y1, x2, y2, classname = annotations[i]
        # print(x1, y1, x2, y2)
        x1_new = int((x1/original_width)*mod_width)
        y1_new = int((y1/original_height)*mod_height)
        x2_new = int((x2/original_width)*mod_width)
        y2_new = int((y2/original_height)*mod_height)
        annotations[i] = [x1_new, y1_new, x2_new, y2_new, classname]

    height, width = image.shape[:2]

    difference_height = self.image_size - height
    difference_width = self.image_size - width

    # Padding the image
    if difference_height > 0 or difference_width > 0:
      image = cv2.copyMakeBorder(image, 0, difference_height, 0, difference_width, borderType=cv2.BORDER_CONSTANT ,value = (255,255,255))
    

    # Used for debugging
    debug_image = image.copy()

    # Normalizing image 
    image = image/255.0
    image = torch.Tensor(image)

    # Making the image channel first
    image = image.permute(2,0,1)

    # Used for debugging
    counter = 0
    for j in range(int(self.image_size/self.grid_size)):
      counter += int(self.image_size/self.grid_size)
      cv2.line(debug_image, (counter, 0), (counter, self.image_size), (255,0, 0), 2)
      cv2.line(debug_image, (0, counter),(self.image_size, counter), (255, 0, 0), 2)

    # Back to normal flow
    boxes = []
    #TODO currently assuming that there won't be more than 10 objects in the image
    # think of a way to accomodate any number of objects
    # I have initialized it with -600 assuming that it won't get a IOU > 0.6
    boxes_xywh = torch.ones((self.num_allowed_objects, 4), requires_grad=False)*(-600)
    
    for a, annotation in enumerate(annotations):
      x1, y1, x2, y2, classname = annotation
      midpoint_x = ((x2 + x1)/2.0)/self.image_size
      midpoint_y = ((y2 + y1)/2.0)/self.image_size
      height = (y2 - y1)/self.image_size
      width = (x2 - x1)/self.image_size
      boxes.append([midpoint_x, midpoint_y, width, height, self.classes.index(classname)])
      boxes_xywh[a, :] = torch.Tensor([midpoint_x, midpoint_y, width, height])
      
    detector_mask, true_boxes = preprocess_label(np.array(boxes), self.anchors, [self.image_size, self.image_size], len(self.classes))
    boxes_xywh = np.array(boxes_xywh)
    # print(boxes_xywh.size())

    return image, detector_mask, true_boxes, boxes_xywh

if __name__ == "__main__":
  pass