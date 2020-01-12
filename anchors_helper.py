import torch

def read_anchors(file_path):
  with open(file_path, "r") as f:
    anchor_line = f.readline()
    anchors = [ float(x) for x in anchor_line.split(",") ]
    anchors = torch.Tensor(anchors).reshape((-1,2))
  return anchors

if __name__ == "__main__":
  file_path = "anchors.txt"
  print(read_anchors(file_path))

# input to yolo head :
# 1. tensor (m, 19, 19, 5, 85)
# 2. tensor (5, 2)
# 3. int == 80