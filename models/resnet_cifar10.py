import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import QConv2d, QLinear, QAdd

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu6(out)
        return out

    def fuse_bn(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'bn2'], inplace=True)

        # Fuse shortcut conv and bn if they exist
        if isinstance(self.shortcut, nn.Sequential) and len(self.shortcut) == 2:
            if isinstance(self.shortcut[0], nn.Conv2d) and isinstance(self.shortcut[1], nn.BatchNorm2d):
                torch.quantization.fuse_modules(self.shortcut, ['0', '1'], inplace=True)

    def quantize(self, w_num_bits, a_num_bits):
        self.conv1 = QConv2d(self.conv1, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True)
        self.conv2 = QConv2d(self.conv2, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True)
        if isinstance(self.shortcut, nn.Sequential) and len(self.shortcut) > 0 and isinstance(self.shortcut[0], nn.Conv2d):
            original_conv_module = self.shortcut[0]
            self.shortcut[0] = QConv2d(original_conv_module, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True)
        self.add = QAdd(qi=False, q_shortcut=False, qo=True, num_bits=a_num_bits)

    def quantize_forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.add(out, self.shortcut(x))
        out = F.relu(out)
        return out
    
    def freeze(self, qi):
        qo=self.conv1.freeze(qi)
        qo=self.conv2.freeze(qo)
        if isinstance(self.shortcut, nn.Sequential) and len(self.shortcut) > 0 and isinstance(self.shortcut[0], nn.Conv2d):
            q_shortcut=self.shortcut[0].freeze(qi)
        else:
            q_shortcut = qi
        self.qo=self.add.freeze(qo, q_shortcut)
        return self.qo
    
    def quantize_inference(self, x):
        main_path_out = F.relu(self.conv1.quantize_inference(x))
        main_path_out = self.conv2.quantize_inference(main_path_out)

        if isinstance(self.shortcut, nn.Sequential) and len(self.shortcut) > 0 and isinstance(self.shortcut[0], QConv2d):
            shortcut_out = self.shortcut[0].quantize_inference(x)
        else: # Identity shortcut
            shortcut_out = x
        out = self.add.quantize_inference(main_path_out, shortcut_out)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        fc_in_features = 512 * 16
        self.linear = nn.Linear(fc_in_features, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def fuse_bn(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1'], inplace=True)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.fuse_bn()

    def quantize(self, w_num_bits, a_num_bits):
        if isinstance(self.conv1, nn.Conv2d):
            self.conv1 = QConv2d(self.conv1, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True)
        
        for layer_group_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer_group = getattr(self, layer_group_name)
            if isinstance(layer_group, nn.Sequential):
                for i in range(len(layer_group)):
                    block_module = layer_group[i]
                    if isinstance(block_module, BasicBlock):
                        block_module.quantize(w_num_bits=w_num_bits, a_num_bits=a_num_bits)
        
        if isinstance(self.linear, nn.Linear):
            self.linear = QLinear(self.linear, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True)

    def quantize_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        current_input = out
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer_sequential = getattr(self, layer_name)
            for block in layer_sequential:
                current_input = block.quantize_forward(current_input)
        out = current_input
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def freeze(self):
        qo = self.conv1.qo
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer_sequential = getattr(self, layer_name)
            for block in layer_sequential:
                qo = block.freeze(qi=qo)
        self.qo = self.linear.freeze(qi=qo)
        return self.qo

    def quantize_inference(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv1.qo.quantize_tensor(out)
        current_input = out
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer_sequential = getattr(self, layer_name)
            for block in layer_sequential:
                current_input = block.quantize_inference(current_input)
        out = current_input
        out = out.view(out.size(0), -1)
        out = self.linear.quantize_inference(out)
        return out

def ResNet18_CIFAR10(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

if __name__ == '__main__':
    model = ResNet18_CIFAR10()
    print("Floating-point ResNet-18 for CIFAR-10 created successfully.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    dummy_input = torch.randn(2, 3, 32, 32) # Batch size 2, 3 channels, 32x32 image
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    model.quantize(num_bits=4)
    q_output = model(dummy_input)
    print(f"Output shape: {q_output.shape}")