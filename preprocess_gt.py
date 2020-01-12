import numpy as np
import torch



def preprocess_label(boxes, anchors, image_size, num_classes):
  # print("num classes : {}".format(num_classes))
  image_height, image_width = image_size
  num_anchors = anchors.shape[0]
  
  # Image should be a perfect multiple of 32 otherwise consider padding
  assert image_height % 32 == 0, "Image height should be a multiple of 32"
  assert image_width % 32 == 0, "Image width should be a multiple of 32"
  
  # The size of the final map would be image size / 32
  conv_height, conv_width = image_height//32, image_width//32
  num_box_params = boxes.shape[1]

  detector_mask = torch.zeros( conv_height, conv_width, num_anchors, 1)
  gt_boxes = torch.zeros(conv_height, conv_width, num_anchors, num_box_params, requires_grad = False)
  
  for box in boxes:
    box_class = int(box[4:5])
    box = box[0:4]*np.array([conv_width, conv_height, conv_width, conv_height])
    
    i = int(np.floor(box[1]))
    j = int(np.floor(box[0]))


    best_iou = 0
    best_anchor = 0
    for k, an_anchor in enumerate(anchors):
      box_maxes = box[2:4]/2
      box_mins = - box_maxes

      anchor_maxes = an_anchor/2
      anchor_min = -anchor_maxes
      inter_min = np.maximum(box_mins, anchor_min)
      inter_max = np.minimum(box_maxes, anchor_maxes)
      inter_wh = np.maximum(inter_max-inter_min, 0)
      inter_area = inter_wh[0]*inter_wh[1]

      box_area = box[2]* box[3]
      anchor_area = an_anchor[0]* an_anchor[1]
      union = box_area*anchor_area - inter_area

      iou = inter_area/union

      if iou > best_iou:
        best_iou  = iou
        best_anchor = k

    if best_iou > 0:
      detector_mask[i, j, best_anchor] = 1.0
      gt_made = torch.Tensor([
        box[0] - j,
        box[1] - i,
        np.log(box[2]/anchors[best_anchor][0]),
        np.log(box[3]/anchors[best_anchor][1]),
        box_class        
      ])
      gt_boxes[i,j, best_anchor : ] = gt_made
  return detector_mask, gt_boxes
    