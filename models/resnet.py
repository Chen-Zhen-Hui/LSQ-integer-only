import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import QConv2d, QLinear, QAdd


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

    def fuse_bn(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'bn2'], inplace=True)
        if self.downsample is not None and isinstance(self.downsample, nn.Sequential):
            if len(self.downsample) == 2 and isinstance(self.downsample[0], nn.Conv2d) and isinstance(self.downsample[1], nn.BatchNorm2d):
                torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)

    def quantize(self, w_num_bits, a_num_bits, precision):
        self.conv1 = QConv2d(self.conv1, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True, precision=precision)
        self.conv2 = QConv2d(self.conv2, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True, precision=precision)
        if self.downsample is not None and isinstance(self.downsample, nn.Sequential) and len(self.downsample) > 0:
            if isinstance(self.downsample[0], nn.Conv2d):
                original_conv_module = self.downsample[0]
                self.downsample[0] = QConv2d(original_conv_module, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True, precision=precision)
        self.add = QAdd(num_bits=a_num_bits, qi=False, q_shortcut=False, qo=True, precision=precision)

    def quantize_forward(self, x, qi=None):
        identity = x
        out = self.relu(self.conv1(x, qi))
        out = self.conv2(out, self.conv1.qo)

        if self.downsample is not None:
            if isinstance(self.downsample[0], QConv2d):
                identity = self.downsample[0](x, qi)
                if len(self.downsample) > 1:
                     for i in range(1, len(self.downsample)):
                        identity = self.downsample[i](identity)
            else:
                identity = self.downsample(x)
        
        q_shortcut_param = qi
        if self.downsample is not None and isinstance(self.downsample[0], QConv2d):
            q_shortcut_param = self.downsample[0].qo 

        out = self.add(out, identity)
        out = self.relu(out)
        return out

    def freeze(self, qi):
        qo_conv1 = self.conv1.freeze(qi)
        qo_conv2 = self.conv2.freeze(qo_conv1)
        
        q_shortcut = qi
        if self.downsample is not None and isinstance(self.downsample, nn.Sequential) and len(self.downsample) > 0:
            if isinstance(self.downsample[0], QConv2d):
                q_shortcut = self.downsample[0].freeze(qi)
        
        self.qo = self.add.freeze(qi=qo_conv2, q_shortcut=q_shortcut)
        return self.qo

    def quantize_inference(self, x):
        identity = x
        out = F.relu(self.conv1.quantize_inference(x))
        out = self.conv2.quantize_inference(out)

        if self.downsample is not None:
            if isinstance(self.downsample[0], QConv2d):
                identity = self.downsample[0].quantize_inference(x)
                if len(self.downsample) > 1:
                    for i in range(1, len(self.downsample)):
                        identity = self.downsample[i](identity)
            else:
                identity = self.downsample(x)
        
        out = self.add.quantize_inference(out, identity)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, image_size, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = 64
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.fc = nn.Linear(512 * block.expansion * (image_size//32) ** 2, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18(image_size, num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], image_size, num_classes=num_classes)

def ResNet34(image_size, num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], image_size, num_classes=num_classes)
