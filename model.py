import numpy as np
import torch

# Squeeze and excitation block
class SE_Block(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE_Block, self).__init__()
        # Adaptive Average Pooling: 
        # This reduces the spatial dimensions (H x W) to 1x1, effectively summarizing 
        # the feature maps across spatial dimensions into a single value per channel.
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        # Fully connected layers for recalibrating channel importance
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction, bias=False),  # Reduction step: Compress the channel dimension
            torch.nn.ReLU(inplace=True),  # Non-linearity to introduce non-linear transformations
            torch.nn.Linear(channels // reduction, channels, bias=False),  # Restore the channel dimension
            torch.nn.Sigmoid()  # Outputs attention weights in the range [0,1] to reweight feature maps
        )
    def forward(self, x):
        batch, channels, w, h = x.size() # Extract batch size, channel, width, and height

        # Squeeze step: Apply global average pooling and reshape to (batch, channels)
        y = self.avg_pool(x).view(batch, channels)

        # Excitation step: Pass through the fully connected layers to learn channel importance
        y = self.fc(y).view(batch, channels, 1, 1) # Reshape to (batch, channels, 1, 1) for broadcasting

        # Scale the original input feature maps by learned channel-wise weights
        return x * y.expand_as(x)# Broadcasting ensures each channel is multiplied by its learned weight

class ResNet_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None, use_se_block=False):
        super(ResNet_Block, self).__init__()

        # Architechture (Basic)
        # CONV -> BN -> ReLU -> CONV -> BN -> Addition -> ReLU
        # Architechture (With SE)
        # CONV -> BN -> ReLU -> CONV -> BN -> SE Block -> Addition -> ReLU

        # First convolutional layer:
        # - Uses a 3x3 kernel with possible downsampling (stride != 1 for size reduction)
        # - Applies Batch Normalization to stabilize training
        # - Uses ReLU activation for non-linearity
        self.convolution1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding=1), # Here the stride is not 1 because we might be coming from a bigger image size and we might need to downsample
                                torch.nn.BatchNorm2d(out_channels),
                                torch.nn.ReLU())
        # Second convolutional layer:
        # - Uses a 3x3 kernel with a fixed stride of 1 (no downsampling here)
        # - Applies Batch Normalization
        # - No activation function applied before the skip connection addition
        self.convolution2 = torch.nn.Sequential(
                                torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1), # Here the stride is always 1 because this is the second conv layer is not downsampled
                                torch.nn.BatchNorm2d(out_channels))
        # Downsampling layer:
        # - Used when the input and output dimensions differ (e.g., due to stride > 1)
        # - Ensures the skip connection can match the new feature map dimensions
        self.downsample = downsample  
        self.relu = torch.nn.ReLU()
        # Option to use a Squeeze-and-Excitation (SE) block 
        self.use_se_block = use_se_block
        self.se_block = SE_Block(out_channels) 


    def forward(self, x):
        res = x # Residual
        if self.downsample is not None: # If downsampling
            res = self.downsample(x) # The input is downsampled

        out = self.convolution1(x) # Conv1
        out = self.convolution2(out) # Conv2

        #SE Block
        if self.use_se_block:
          out = self.se_block(out)

        z = out + res # Identity mapping
        z = self.relu(z) # Final ReLU

        return z


class PreActivation_ResNet_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None, use_se_block=False):
        super(PreActivation_ResNet_Block, self).__init__()

        # Architechture (Basic)
        # BN -> ReLU -> CONV ->  BN -> ReLU -> CONV -> Addition
        # Architechture (With SE)
        # BN -> ReLU -> CONV ->  BN -> ReLU -> CONV -> SE Block -> Addition
        
        # Batch Normalization
        self.bn1 = torch.nn.BatchNorm2d(in_channels)

        # ReLU
        self.relu1 = torch.nn.ReLU(inplace=True)

        #Uses a 3x3 kernel with possible downsampling (stride != 1 for size reduction)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # Batch Normalization
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        # ReLU
        self.relu2 = torch.nn.ReLU(inplace=True)

        #Uses a 3x3 kernel with a fixed stride of 1 (no downsampling here)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        self.use_se_block = use_se_block
        self.se_block = SE_Block(out_channels)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu1(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.use_se_block:
          out = self.se_block(out)

        out += residual
        return out

class ResNet(torch.nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.block = block
        # Input = 3 channels, 32 * 32
        # Initial Convolution that uses a 5 * 5 kernel with stride = 1 and padding = 2. 
        # Creates 64 channels
        # Batch Normalization
        # ReLU
        self.convolution1 = torch.nn.Sequential(
                                torch.nn.Conv2d(3, 64, kernel_size = 5, stride = 1, padding = 2),
                                torch.nn.BatchNorm2d(64),
                                torch.nn.ReLU()) # Output size => 32*32
        # First ResNet Block with input 64 channels and output 64 channels. Image size does not change
        self.layer0 = self.add_res_net_block(64, 64, layers[0], first_layer_stride = 1) # 32*32

        # Second ResNet Block with input 64 channels and output 128 channels. Image size = 16 * 16
        self.layer1 = self.add_res_net_block(64, 128, layers[1], first_layer_stride = 2)# 16*16

         # Second ResNet Block with input 128 channels and output 256 channels. Image size = 8 * 8
        self.layer2 = self.add_res_net_block(128, 256, layers[2], first_layer_stride = 2)#8*8

        # Average pool. Reduces the spatial dimensions of every channel to 1 node
        self.avgpool = torch.nn.AvgPool2d(8, stride=1)

        # Final connected layer with 10 output
        self.fc = torch.nn.Linear(256, 10)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d): # Initialize from kaming normal, if it is a convolution layer.
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d): # If batch normalization ==> weights = 1 and biases = 0
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    # Function to add a block 
    '''
        Arguments
            Input channels
            output channels
            number of layers
            initial stride (to check for downsampling)
    '''
    def add_res_net_block(self, in_channels, out_channels, layers, first_layer_stride):
        downsample = None
        num_layers, use_se_block = layers # un pack layers (num_layers:int, use_se_block:bool)

        # Check if downsampling is needed:
        # - If stride > 1, the spatial dimensions will shrink, requiring downsampling.
        # - If input and output channels differ, a 1x1 convolution is used to match dimensions.
        if first_layer_stride != 1 or in_channels != out_channels: 

            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=first_layer_stride),
                torch.nn.BatchNorm2d(out_channels)
            )
        block_layers = []
        # First layer of the residual block:
        # - Uses downsampling if needed
        block_layers.append(self.block(in_channels, out_channels, first_layer_stride, downsample))
        for i in range(num_layers-1):
            # Remaining layers of the residual block:
            # - Stride is always 1 (no further downsampling)
            # - Uses SE block if enabled
            if use_se_block:
                block_layers.append(self.block(out_channels, out_channels, 1, None, True))
            else:
                block_layers.append(self.block(out_channels, out_channels, 1, None, False))
        return torch.nn.Sequential(*block_layers) # return sequence of all the layers as a block

    def forward(self, x):
        x = self.convolution1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x