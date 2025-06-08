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
        self.fc1 = nn.Linear(self.feature_size, fc_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc_channels, fc_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(fc_channels, num_classes)

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
            stage.add_module(f'relu{index}', nn.ReLU(inplace=True))
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

    def quantize(self, w_num_bits, a_num_bits, precision):
        for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
            stage = getattr(self, stage_name)
            if isinstance(stage, nn.Sequential):
                for i in range(len(stage)):
                    module = stage[i]
                    if isinstance(module, nn.Conv2d):
                        # Replace nn.Conv2d with QConv2d
                        stage[i] = QConv2d(module, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True, precision=precision)
        
        if isinstance(self.fc1, nn.Linear):
            self.fc1 = QLinear(self.fc1, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True, precision=precision)
        if isinstance(self.fc2, nn.Linear):
            self.fc2 = QLinear(self.fc2, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True, precision=precision)
        if isinstance(self.fc3, nn.Linear):
            self.fc3 = QLinear(self.fc3, w_num_bits=w_num_bits, a_num_bits=a_num_bits, qi=False, qo=True, precision=precision)

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
        last_qo = None # This will track the output quantizer of the previous layer
        # Initial input quantizer (if inputs are meant to be quantized, otherwise None or a default QParam for fp inputs)
        # For now, let's assume the very first conv layer's qi will be initialized based on its own settings (e.g. if qi=True in QConv2d)
        # or it expects floating point input. If the model input itself needs quantization, a global qi should be passed and managed.
        
        # Freeze convolutional stages
        for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
            stage = getattr(self, stage_name)
            if isinstance(stage, nn.Sequential):
                for i in range(len(stage)):
                    module = stage[i]
                    if isinstance(module, QConv2d):
                        # The first QConv in a chain might not have a last_qo if it's the first layer overall.
                        # QConv2d.freeze expects the input quantizer `qi`.
                        # If module.qi is already set (e.g. qi=True in constructor), it might use that.
                        # Otherwise, we need to provide it. Let's assume qi is passed if not first layer.
                        current_module_qi = last_qo
                        if last_qo is None and not hasattr(module, 'qi'): # Very first conv, and it doesn't have its own qi
                             # This case needs careful handling. If input is float, qi for first layer is effectively None or a dummy.
                             # QConv2d freeze logic needs to be robust to this.
                             # For now, let's assume QConv2d with qi=False handles this by not needing an explicit qi for freezing its weights if no input scale is given
                             # Or, we establish a convention: first layer's weights are quantized based on its qw, and its output qo is established.
                             # The `freeze` in `QConv2d` needs the input scale `qi.alpha` to calculate `M`.
                             # Let's assume an initial `QParam` for the input if not provided.
                             # This part is tricky and depends on QConv2d's freeze implementation details.
                             # Simplified: The QConv2d itself should handle if qi is passed as None to freeze when it's the first layer.
                             # Or, we need an explicit input QParam for the whole network.
                             # For now, let's assume QConv2d's freeze can take `None` for `qi` for the first layer.
                             pass # Fallthrough, QConv2d should handle if qi is None

                        module_qo = module.freeze(qi=current_module_qi)
                        last_qo = module_qo # Output quantizer of this conv becomes input for next
                    # ReLU and other layers usually don't change the quantization parameters themselves in freeze,
                    # but rely on the preceding/succeeding Q-layers.

        # Freeze fully connected layers
        if isinstance(self.fc1, QLinear):
            fc1_qo = self.fc1.freeze(qi=last_qo)
            last_qo = fc1_qo
        # ReLU after fc1 doesn't change last_qo scale
        if isinstance(self.fc2, QLinear):
            fc2_qo = self.fc2.freeze(qi=last_qo)
            last_qo = fc2_qo
        # ReLU after fc2 doesn't change last_qo scale
        if isinstance(self.fc3, QLinear):
            self.fc3.freeze(qi=last_qo)
            # Output layer, its qo is the final network qo if needed elsewhere.

    def quantize_inference(self, x):
        # Assume x is already quantized if the model expects quantized input
        # Or, x is float and the first layer handles its quantization (integer mult by M0/2^n)

        # Stage 1
        for i, layer in enumerate(self.stage1):
            if isinstance(layer, QConv2d):
                x = layer.quantize_inference(x)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x) # Apply ReLU directly on potentially integer data (clamped)
            else:
                x = layer(x) # MaxPool, etc.
        
        # Stage 2
        for i, layer in enumerate(self.stage2):
            if isinstance(layer, QConv2d):
                x = layer.quantize_inference(x)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x)
            else:
                x = layer(x)

        # Stage 3
        for i, layer in enumerate(self.stage3):
            if isinstance(layer, QConv2d):
                x = layer.quantize_inference(x)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x)
            else:
                x = layer(x)

        # Stage 4
        for i, layer in enumerate(self.stage4):
            if isinstance(layer, QConv2d):
                x = layer.quantize_inference(x)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x)
            else:
                x = layer(x)

        x = torch.flatten(x, 1)
        
        # FC layers
        if isinstance(self.fc1, QLinear):
            x = self.fc1.quantize_inference(x)
        else:
            x = self.fc1(x)
        x = self.relu1(x) # Assuming relu1 is nn.ReLU
        x = self.dp1(x) 
        
        if isinstance(self.fc2, QLinear):
            x = self.fc2.quantize_inference(x)
        else:
            x = self.fc2(x)
        x = self.relu2(x) # Assuming relu2 is nn.ReLU
        x = self.dp2(x)

        if isinstance(self.fc3, QLinear):
            x = self.fc3.quantize_inference(x)
        else:
            x = self.fc3(x)
            
        return x

if __name__ == '__main__':
    model = TinyVGG(image_size=64, num_classes=200)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(y.shape)