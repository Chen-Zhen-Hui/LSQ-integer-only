import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import QConv2d, QLinear, QParam, QAdd


bit=2

n_channels = [3, 32, 64, 128, 256]
fc_channels = 1024
n_layers = [3, 3, 4, 3]
image_size = 128

class TinyVGG(nn.Module):
    def __init__(self,image_size=image_size,num_classes=1000):
        super(TinyVGG, self).__init__()

        self.stage1 = self._make_stage(n_channels[0], n_channels[1], n_layers[0])
        self.stage2 = self._make_stage(n_channels[1], n_channels[2], n_layers[1])
        self.stage3 = self._make_stage(n_channels[2], n_channels[3], n_layers[2])
        self.stage4 = self._make_stage(n_channels[3], n_channels[4], n_layers[3])
        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros((1, n_channels[0], image_size, image_size))
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0]
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.relu1 = nn.ReLU6(inplace=True)
        self.fc2 = nn.Linear(512, 1024)
        self.relu2 = nn.ReLU6(inplace=True)
        self.fc3 = nn.Linear(1024, num_classes)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_channels, out_channels, n_blocks):
        stage = nn.Sequential()
        for index in range(n_blocks):
            if index == 0:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            else:
                conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

            stage.add_module(f'conv{index}', conv)
            stage.add_module(f'bn{index}', nn.BatchNorm2d(out_channels))
            stage.add_module(f'relu{index}', nn.ReLU6(inplace=True))
        return stage

    def _forward_conv(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = torch.flatten(x, 1)
        x = self.dp1(self.relu1(self.fc1(x)))
        x = self.dp2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def fuse_bn(self):
        for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
            stage = getattr(self, stage_name)
            if isinstance(stage, nn.Sequential):
                temp_list = list(stage.named_children())
                
                i = 0
                while i < len(temp_list):
                    name, module = temp_list[i]
                    if isinstance(module, nn.Conv2d):
                        if (i + 1) < len(temp_list) and isinstance(temp_list[i+1][1], nn.BatchNorm2d):
                            conv_name = name
                            bn_name = temp_list[i+1][0]
                            torch.quantization.fuse_modules(stage, [conv_name, bn_name], inplace=True)
                    i += 1

    def quantize(self, w_num_bits, a_num_bits):
        for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
            stage = getattr(self, stage_name)
            if isinstance(stage, nn.Sequential):
                for i in range(len(stage)):
                    module = stage[i]
                    if isinstance(module, nn.Conv2d):
                        stage[i] = QConv2d(module, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True)
        
        if isinstance(self.fc1, nn.Linear):
            self.fc1 = QLinear(self.fc1, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True)
        if isinstance(self.fc2, nn.Linear):
            self.fc2 = QLinear(self.fc2, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True)
        if isinstance(self.fc3, nn.Linear):
            self.fc3 = QLinear(self.fc3, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True)

    def quantize_forward(self, x):
        x = F.relu(self.stage1.conv0(x))
        x = F.relu(self.stage1.conv1(x, qi=self.stage1.conv0.qo))
        x = F.relu(self.stage1.conv2(x, qi=self.stage1.conv1.qo))
        x = F.relu(self.stage2.conv0(x, qi=self.stage1.conv2.qo))
        x = F.relu(self.stage2.conv1(x, qi=self.stage2.conv0.qo))
        x = F.relu(self.stage2.conv2(x, qi=self.stage2.conv1.qo))
        x = F.relu(self.stage3.conv0(x, qi=self.stage2.conv2.qo))
        x = F.relu(self.stage3.conv1(x, qi=self.stage3.conv0.qo))
        x = F.relu(self.stage3.conv2(x, qi=self.stage3.conv1.qo))
        x = F.relu(self.stage3.conv3(x, qi=self.stage3.conv2.qo))
        x = F.relu(self.stage4.conv0(x, qi=self.stage3.conv3.qo))
        x = F.relu(self.stage4.conv1(x, qi=self.stage4.conv0.qo))
        x = F.relu(self.stage4.conv2(x, qi=self.stage4.conv1.qo))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x, qi=self.stage4.conv2.qo))
        x = F.relu(self.fc2(x, qi=self.fc1.qo))
        x = self.fc3(x, qi=self.fc2.qo)
        return x

    def freeze(self):
        qo_s1c0 = self.stage1.conv0.qo
        qo_s1c1 = self.stage1.conv1.freeze(qi=qo_s1c0)
        qo_s1c2 = self.stage1.conv2.freeze(qi=qo_s1c1)

        qo_s2c0 = self.stage2.conv0.freeze(qi=qo_s1c2)
        qo_s2c1 = self.stage2.conv1.freeze(qi=qo_s2c0)
        qo_s2c2 = self.stage2.conv2.freeze(qi=qo_s2c1)

        qo_s3c0 = self.stage3.conv0.freeze(qi=qo_s2c2)
        qo_s3c1 = self.stage3.conv1.freeze(qi=qo_s3c0)
        qo_s3c2 = self.stage3.conv2.freeze(qi=qo_s3c1)
        qo_s3c3 = self.stage3.conv3.freeze(qi=qo_s3c2)

        qo_s4c0 = self.stage4.conv0.freeze(qi=qo_s3c3)
        qo_s4c1 = self.stage4.conv1.freeze(qi=qo_s4c0)
        qo_s4c2 = self.stage4.conv2.freeze(qi=qo_s4c1)

        qo_fc1 = self.fc1.freeze(qi=qo_s4c2)
        qo_fc2 = self.fc2.freeze(qi=qo_fc1)
        qo_fc3 = self.fc3.freeze(qi=qo_fc2)
        self.qo = qo_fc3

    def quantize_inference_integer(self, x):
        # stage1
        x = self.stage1.conv0(x)
        x = F.relu(x)
        x = self.stage1.conv0.qo.quantize_tensor(x)
        x = self.stage1.conv1.quantize_inference_integer(x)
        x = F.relu(x)
        x = self.stage1.conv2.quantize_inference_integer(x)
        x = F.relu(x)
        
        # stage2
        x = self.stage2.conv0.quantize_inference_integer(x)
        x = F.relu(x)
        x = self.stage2.conv1.quantize_inference_integer(x)
        x = F.relu(x)
        x = self.stage2.conv2.quantize_inference_integer(x)
        x = F.relu(x)

        # stage3
        x = self.stage3.conv0.quantize_inference_integer(x)
        x = F.relu(x)
        x = self.stage3.conv1.quantize_inference_integer(x)
        x = F.relu(x)
        x = self.stage3.conv2.quantize_inference_integer(x)
        x = F.relu(x)
        x = self.stage3.conv3.quantize_inference_integer(x)
        x = F.relu(x)

        # stage4
        x = self.stage4.conv0.quantize_inference_integer(x)
        x = F.relu(x)
        x = self.stage4.conv1.quantize_inference_integer(x)
        x = F.relu(x)
        x = self.stage4.conv2.quantize_inference_integer(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        
        x = self.fc1.quantize_inference_integer(x)
        x = F.relu(x)
        
        x = self.fc2.quantize_inference_integer(x)
        x = F.relu(x)
        
        x = self.fc3.quantize_inference_integer(x)
        x = self.qo.dequantize_tensor(x)
        return x

if __name__ == '__main__':
    model = TinyVGG(image_size=64, num_classes=200)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(y.shape)