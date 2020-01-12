import torch.nn as nn
import torch
import numpy as np

# Note that this function assumes channel first and 
# shape should be (batch_size, channels, height, width)
def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

class YoloConvBlock(nn.Module):
  def __init__(self, input_channels, output_channels, kernel_size, padding=True, negative_slope=0.01):
    super(YoloConvBlock, self).__init__()
    self.conv = nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size, padding=padding)
    self.conv_bn = nn.BatchNorm2d(output_channels)
    self.activation = nn.LeakyReLU(negative_slope)

  def forward(self, input_tensor):
    X = self.conv(input_tensor)
    X = self.conv_bn(X)
    X = self.activation(X)
    return X

class YoloDarknet(nn.Module):
  def __init__(self, num_classes):
    super(YoloDarknet, self).__init__()
    self.conv1 = YoloConvBlock(3, 32, 3)
    self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.conv2 = YoloConvBlock(32, 64, 3)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv3 = YoloConvBlock(64, 128, 3)
    self.conv4 = YoloConvBlock(128, 64, 1, padding=False)
    self.conv5 = YoloConvBlock(64, 128, 3)
    self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv6 = YoloConvBlock(128, 256, 3)
    self.conv7 = YoloConvBlock(256, 128, 1, padding=False)
    self.conv8 = YoloConvBlock(128, 256, 3)
    self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv9 = YoloConvBlock(256, 512, 3)
    self.conv10 = YoloConvBlock(512, 256, 1, padding=False)
    self.conv11 = YoloConvBlock(256, 512, 3)
    self.conv12 = YoloConvBlock(512, 256, 1, padding=False)
    self.conv13 = YoloConvBlock(256, 512, 3)
    self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv14 = YoloConvBlock(512, 1024, 3)
    self.conv15 = YoloConvBlock(1024, 512, 1, padding=False)
    self.conv16 = YoloConvBlock(512, 1024, 3)
    self.conv17 = YoloConvBlock(1024, 512, 1, padding=False)
    self.conv18 = YoloConvBlock(512, 1024, 3)
    self.conv19 = YoloConvBlock(1024, 1024, 3)
    self.conv20 = YoloConvBlock(1024, 1024, 3)
    self.conv21 = YoloConvBlock(512, 64, 1, padding=False)
    self.conv22 = YoloConvBlock(1280, 1024, 3)
    self.conv23 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=False)

  
  def forward(self, input_tensor):
    conv1_out = self.conv1.forward(input_tensor)
    maxp1_out = self.maxpool1(conv1_out)
    
    conv2_out = self.conv2(maxp1_out)
    maxp2_out = self.maxpool2(conv2_out)

    conv3_out = self.conv3(maxp2_out)
    conv4_out = self.conv4(conv3_out)
    conv5_out = self.conv5(conv4_out)
    maxp3_out = self.maxpool3(conv5_out)

    conv6_out = self.conv6(maxp3_out)
    conv7_out = self.conv7(conv6_out)
    conv8_out = self.conv8(conv7_out)
    maxp4_out = self.maxpool4(conv8_out)

    conv9_out = self.conv9(maxp4_out)
    conv10_out = self.conv10(conv9_out)
    conv11_out = self.conv11(conv10_out)
    conv12_out = self.conv12(conv11_out)
    conv13_out = self.conv13(conv12_out)
    maxp5_out = self.maxpool5(conv13_out)

    conv14_out = self.conv14(maxp5_out)
    conv15_out = self.conv15(conv14_out)
    conv16_out = self.conv16(conv15_out)
    conv17_out = self.conv17(conv16_out)
    conv18_out = self.conv18(conv17_out)
    conv19_out = self.conv19(conv18_out)
    conv20_out = self.conv20(conv19_out)
    conv21_out = self.conv21(conv13_out)
    std_out = space_to_depth(conv21_out, 2)
    cat1 = torch.cat((std_out, conv20_out),1)
    conv22_out = self.conv22(cat1)
    conv23_out = self.conv23(conv22_out)

    return conv23_out

if __name__ == "__main__":
  random_tensor = np.random.random((1, 3, 224, 224))
  yolov2_obj = YoloDarknet()
  output = yolov2_obj(torch.Tensor(random_tensor))
  print(output.shape)