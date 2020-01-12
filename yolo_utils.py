import torch

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")

# converts the boxes from xy and wh to 
# y1,x1,y2,x2 --> this is the output
def boxes_to_corners(box_xy, box_wh):
  box_mins = box_xy - box_wh/2
  box_maxs = box_xy + box_wh/2

  return torch.cat([
    box_mins[:,:,:,:,1:2], # y1
    box_mins[:,:,:,:,0:1], # x1
    box_maxs[:,:,:,:,1:2], # y2
    box_maxs[:,:,:,:,0:1]  # x2    
  ], dim=-1)

def scale_boxes(boxes, image_shape):
  height, width = image_shape
  image_dims = torch.Tensor([height, width, height, width]).to(device)
  image_dims = image_dims.reshape((1,4))
  boxes = boxes*image_dims
  return boxes

def yolo_filter_boxes(box_confidence, box_class_probs, boxes, score_threshold=0.5):
  box_scores = box_confidence*box_class_probs
  # Getting the max elements
  box_class_max = torch.max(box_scores, axis=-1)
  box_class_scores, box_classes = box_class_max.values, box_class_max.indices
  batch, im_height, im_width, num_anchors = box_class_scores.size()

  box_class_scores, box_classes = box_class_scores, box_classes
  prediction_mask = box_class_scores.ge(score_threshold)
  boxes = boxes[prediction_mask]
  scores = box_class_scores[prediction_mask]
  classes = box_classes[prediction_mask]
  return scores, classes, boxes

  