from anchors_helper import read_anchors
import cv2
import os
import torch

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

class ImagePredictLoader(object):
  
  def __init__(self, images_folder):
    all_image_names = os.listdir(images_folder)
    
    self.classes = ["aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor", "person", "bird", "cat", "cow", "dog", "horse", "sheep"]
    self.anchors = read_anchors("anchors.txt")
    self.image_size = 416
    self.grid_size = 13
    
    self.image_files = []
    
    for image_name in all_image_names:
      if image_name.endswith(".jpg") or image_name.endswith(".png"):
        self.image_files.append(os.path.join(images_folder, image_name))
        
  def __len__(self):
    return len(self.image_files)
  
  def __getitem__(self, index):
    image_path = self.image_files[index]
    image = cv2.imread(image_path)

    original_height, original_width = image.shape[:2]

    if original_height > self.image_size or original_width > self.image_size:
      if original_height > original_width:
        image = image_resize(image, width = None, height = self.image_size, inter = cv2.INTER_AREA)
      else:
        image = image_resize(image, width = self.image_size, height = None , inter = cv2.INTER_AREA)

     
    height, width = image.shape[:2]

    difference_height = self.image_size - height
    difference_width = self.image_size - width

    # Padding the image
    if difference_height > 0 or difference_width > 0:
      image = cv2.copyMakeBorder(image, 0, difference_height, 0, difference_width, borderType=cv2.BORDER_CONSTANT ,value = (255,255,255))
    

    # Normalizing image 
    image = image/255.0
    image = torch.Tensor(image)

    # Making the image channel first
    image = image.permute(2,0,1)

    return image
