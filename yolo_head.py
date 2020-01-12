import torch
import torch.nn as nn

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")

# This assumes channel last
def yolo_head(conv_output, anchors, num_classes):
  num_anchors = len(anchors)
  anchors = anchors.reshape((1, 1, 1, num_anchors, 2)).to(device)
  conv_dims = conv_output.size()[1:3]
  # TODO 
  # Currently assuming that the feature map would be a square 
  # Add code to handle various dimensions
  conv_index = torch.arange(0, conv_dims[1])
  conv_index = torch.flip(torch.cartesian_prod(conv_index, conv_index), dims=(-1,)).reshape((1,conv_dims[0], conv_dims[1],1,2)).to(device)
  
  # changing the shape of conv_dims in order to 
  conv_dims = torch.Tensor([conv_dims[0], conv_dims[1]]).expand((1,1,1,1,2)).to(device)
  
  
  # We want the confidence and xy to be between 0 and 1 
  box_confidence = nn.Sigmoid()(conv_output[:,:,:,:, 4:5])
  box_xy = nn.Sigmoid()(conv_output[:,:,:,:, :2])
  box_wh = torch.exp(conv_output[:,:,:,:, 2:4])
  box_class_prob = nn.Softmax()(conv_output[:,:,:,:, 5:])

  box_xy = (box_xy + conv_index)/conv_dims
  box_wh = box_wh*anchors/conv_dims
  
  return box_confidence, box_xy, box_wh, box_class_prob


if __name__ == "__main__":
  anchors = torch.Tensor([[0.57273,  0.677385],
                      [1.87446,  2.06253 ],
                      [3.33843,  5.47434 ],
                      [7.88282,  3.52778 ],
                      [9.77052,  9.16828 ]])
  conv_output = torch.randn((1,5,6,7,7))
  num_classes = 3
  conv_output = conv_output.permute(0,3,4,1,2)
  print(yolo_head(conv_output, anchors, num_classes))