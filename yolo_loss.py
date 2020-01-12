import torch
import torch.nn as nn
import numpy as np

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")

class YoloLoss(nn.Module):
  def __init__(self):
    super(YoloLoss, self).__init__()
    self.coords_scale = 1
    self.class_scale = 1
    self.rescore_confidence = False
    self.object_scale = 5
    self.no_object_scale = 1
    self.num_classes = 20
    self.one_hot_matrix = np.identity(self.num_classes)
  
  def forward(self, detector_mask, true_boxes, yolo_output, yolo_head_output, box_xywh):
    box_confidence, box_xy, box_wh, box_class_prob = yolo_head_output

    # box_xywh = box_xywh.to(device)
    batch_size, num_boxes, box_params = box_xywh.size()
    box_xywh = torch.reshape(box_xywh, (batch_size, 1, 1, 1, num_boxes, box_params))
    real_xy = box_xywh[..., 0:2]
    real_wh = box_xywh[..., 2:4]

    real_wh_half = real_wh / 2.
    real_mins = real_xy - real_wh_half
    real_maxes = real_xy + real_wh_half

    box_xy = box_xy.unsqueeze(4)
    box_wh = box_wh.unsqueeze(4)

    box_wh_half = box_wh / 2.
    box_mins = box_xy - box_wh_half
    box_maxes = box_xy + box_wh_half

    intersect_mins = torch.min(box_mins, real_mins)
    intersect_maxs = torch.max(box_maxes, real_maxes)
    intersect_wh = torch.max(intersect_maxs - intersect_mins , 0).values
    intersect_area = intersect_wh[..., 0]*intersect_wh[..., 1]

    box_area = box_wh[...,0]* box_wh[...,1]
    real_area = real_wh[...,0]*real_wh[...,1]

    union_areas = box_area + real_area - intersect_area
    iou_scores = intersect_area / union_areas
    best_iou = torch.max(iou_scores, axis = 4).values

    iou_scores = iou_scores.unsqueeze(-1)

    # detector_mask = detector_mask.squeeze(-1)

    object_detections = torch.ge(best_iou, 0.6)

    
    not_object_detections =  torch.logical_not(object_detections.float()).float()
    not_detection_mask = torch.logical_not(detector_mask.squeeze()).float()

    no_object_mask = self.no_object_scale* torch.mul(not_object_detections,not_detection_mask)
    no_object_loss = torch.sum( no_object_mask.unsqueeze(-1) * torch.pow(-box_confidence, 2))
   

    objects_loss = 0
    if self.rescore_confidence:
      objects_loss = torch.sum(self.object_scale * detector_mask * (torch.pow( torch.sub(best_iou, box_confidence), 2)))

    else:
      objects_loss = torch.sum(self.object_scale * detector_mask * torch.pow( 1 - box_confidence, 2))

    bbox_predictions = yolo_output[...,0:4]
    bbox_predictions = torch.cat((nn.Sigmoid()(yolo_output[...,:2]), yolo_output[...,2:4]), dim=-1)
    bbox_gt = true_boxes[...,0:4]
    
    coords_loss= torch.sum(self.coords_scale* detector_mask * torch.pow(torch.sub(bbox_gt, bbox_predictions), 2))
    
    all_classes_in_batch = true_boxes[:,:,:,:, -1].cpu().data.numpy().astype(np.int32)
    
    one_hot_batch_classes = torch.Tensor(self.one_hot_matrix[all_classes_in_batch])
    one_hot_batch_classes = one_hot_batch_classes.to(device)
    class_loss = torch.sum(self.class_scale* detector_mask * torch.pow(torch.sub(one_hot_batch_classes, box_class_prob), 2))

    total_loss = (coords_loss + class_loss + objects_loss + no_object_loss)/batch_size

    return total_loss
