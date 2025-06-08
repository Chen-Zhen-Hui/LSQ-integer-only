import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Rounds a tensor with a Straight-Through Estimator for the gradient.
    """
    return (torch.round(x) - x).detach() + x


def round_ste_for_inference(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x)


def quantize_tensor(x, scale, num_bits):
    if num_bits > 1:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0
        qmax = 1
    
    x_div_scale = x / scale
    x_clamped = torch.clamp(x_div_scale, qmin, qmax)
    
    # Round with STE
    q_x = round_ste(x_clamped)
    return q_x


def quantize_tensor_for_inference(x: torch.Tensor, scale: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Quantizes a tensor for inference: scales, clamps, and rounds.
    """
    if num_bits > 1:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0
        qmax = 1
    
    x_div_scale = x / scale
    x_clamped = torch.clamp(x_div_scale, qmin, qmax)
    
    q_x = round_ste_for_inference(x_clamped)
    return q_x


def dequantize_tensor(q_x, scale):
    return scale * q_x


def fake_quantize_tensor(x, scale, num_bits):
    q_x = quantize_tensor(x, scale, num_bits=num_bits)
    return dequantize_tensor(q_x, scale)


def search(M, max_bits=14):
    n = 1
    while True:
        Mo = int(round(2 ** n * M))
        approx = Mo / 2**n
        error = approx - M

        if Mo>2**max_bits:
            return Mo, n

        if n>7 and (math.fabs(error) < 1e-6 or n >= 22):
            return Mo, n
        n += 1


class QParam(nn.Module):

    def __init__(self, num_bits):
        super(QParam, self).__init__()
        self.num_bits = num_bits
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('init_state', torch.zeros(1))
        self.num_elements_for_grad_scale = None  # Store N for gradient scaling

        # Register gradient hook for alpha
        if self.alpha.requires_grad:
            self.alpha.register_hook(self._grad_scaling_hook)

    def _grad_scaling_hook(self, grad):
        """Scales the gradient of alpha as per LSQ paper."""
        if self.num_elements_for_grad_scale is not None and self.num_elements_for_grad_scale > 0:
            q_p = 2. ** (self.num_bits - 1) - 1
            if q_p > 0:
                scale_factor = 1.0 / (self.num_elements_for_grad_scale * q_p)**0.5
                return grad * scale_factor
            else:
                scale_factor = 1.0 / (self.num_elements_for_grad_scale)**0.5
                return grad * scale_factor
        return grad

    def initialize_alpha(self, tensor: torch.Tensor):
        """Initializes alpha based on the input tensor's statistics."""
        if self.init_state.item() == 0:
            q_p = 2. ** (self.num_bits - 1) - 1 if self.num_bits > 1 else 1
            if isinstance(tensor, nn.Parameter):
                tensor_data = tensor.data
                self.num_elements_for_grad_scale = tensor_data.numel()
            else:
                tensor_data = tensor
                self.num_elements_for_grad_scale = tensor_data[0].numel()
            
            mean_abs_val = torch.mean(torch.abs(tensor_data.float()))

            if mean_abs_val.item() == 0:
                 self.alpha.data.fill_(1.0)
            else:
                self.alpha.data = 2 * mean_abs_val / (q_p ** 0.5)
            self.init_state.fill_(1)

    def quantize_tensor(self, tensor):
        if self.training:
            return quantize_tensor(tensor, self.alpha, self.num_bits)
        else:
            return quantize_tensor_for_inference(tensor, self.alpha, self.num_bits)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.alpha)
    
    def fake_quantize(self, tensor):
        target_dtype = tensor.dtype
        result = fake_quantize_tensor(tensor, self.alpha, self.num_bits)
        return result.to(target_dtype)


class QModule(nn.Module):

    def __init__(self, num_bits, qi=False, qo=True):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits=num_bits)
        if qo:
            self.qo = QParam(num_bits=num_bits)

    def freeze(self):
        pass

    def quantize_inference(self):
        '''
        This function is used to quantize the inference of the model in integer-only mode.
        '''
        pass


class QConv2d(QModule):
    def __init__(self, conv_module, w_num_bits, a_num_bits, qi=False, qo=True):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=a_num_bits)
        self.conv_module = conv_module
        self.qw = QParam(num_bits=w_num_bits)
        self.register_buffer('M', torch.zeros(1))
        self.register_buffer('M0', torch.zeros(1))
        self.register_buffer('n', torch.zeros(1))
        self.register_buffer('Mo', torch.zeros(1))
        self.register_buffer('no', torch.zeros(1))

    def forward(self, x, qi=None):
        target_dtype = self._get_target_dtype()

        if self.qw.init_state.item() == 0:
            self.qw.initialize_alpha(self.conv_module.weight.float())
        
        q_weight = self.qw.fake_quantize(self.conv_module.weight)
        
        if self.conv_module.bias is not None:
            if qi:
                # bias_scale = self.qw.alpha.detach() * qi.alpha.detach()
                bias_scale = self.qw.alpha * qi.alpha
                q_bias = fake_quantize_tensor(self.conv_module.bias.float(), bias_scale, 32)
            else:
                q_bias = self.conv_module.bias.float()
            q_bias_prec = q_bias.to(target_dtype)
        else:
            q_bias_prec = None

        x_prec = x.to(target_dtype)
        q_weight_prec = q_weight.to(target_dtype)

        output_prec = F.conv2d(x_prec, q_weight_prec, q_bias_prec, 
                               self.conv_module.stride, self.conv_module.padding, 
                               self.conv_module.dilation, self.conv_module.groups)

        if hasattr(self, 'qo'):
            if self.qo.init_state.item() == 0:
                self.qo.initialize_alpha(output_prec.float())
            final_output = self.qo.fake_quantize(output_prec)
        else:
            final_output = output_prec.float()
        
        return final_output

    def freeze(self, qi:QParam=None, qo:QParam=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        
        if not hasattr(self, 'qi') or not hasattr(self, 'qo') or not hasattr(self, 'qw'):
            print("Warning: qi, qo, or qw not fully available for M calculation in freeze. Skipping M calculation.")
        else:
            M_val = (self.qw.alpha.data.item() * self.qi.alpha.data.item()) / self.qo.alpha.data.item()

            Mo, n_val = search(M_val.item() if isinstance(M_val, torch.Tensor) else M_val)
            self.M0.data = torch.tensor(Mo, dtype=torch.float) # M0 can be large, ensure float or int64
            self.n.data = torch.tensor(n_val, dtype=torch.float)
            self.Mo.data = torch.tensor(1, dtype=torch.float)
            self.no.data = torch.tensor(0, dtype=torch.float)


        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        if self.conv_module.bias is not None:
            if hasattr(self, 'qi') and hasattr(self, 'qw'): # Ensure scales are available
                bias_scale = self.qo.alpha.data
                if bias_scale.item() != 0:
                    self.conv_module.bias.data = quantize_tensor(self.conv_module.bias.data, scale=bias_scale, num_bits=64)
                else:
                    print("Warning: Bias scale is zero. Bias not quantized for conv layer.")
            else:
                print("Warning: qi.alpha or qw.alpha not available for conv bias quantization. Bias not quantized.")
        return self.qo

    def quantize_inference(self, x):
        x = F.conv2d(x, self.conv_module.weight, None,
                     self.conv_module.stride, self.conv_module.padding, 
                     self.conv_module.dilation, self.conv_module.groups)
        x = x*self.M0.data.item()/2**self.n.data.item()
        x = round_ste_for_inference(x)
        x = x+self.conv_module.bias.view(1, self.conv_module.out_channels, 1, 1)
        x = x*self.Mo.data.item()/2**self.no.data.item()
        x.clamp_(-2**(self.qo.num_bits-1), 2**(self.qo.num_bits-1)-1)
        return x

class QLinear(QModule):
    def __init__(self, linear_module, w_num_bits, a_num_bits, qi=False, qo=True):
        super(QLinear, self).__init__(qi=qi, qo=qo, num_bits=a_num_bits)
        self.linear_module = linear_module
        self.qw = QParam(num_bits=w_num_bits)
        self.register_buffer('M', torch.zeros(1))
        self.register_buffer('M0', torch.zeros(1))
        self.register_buffer('n', torch.zeros(1))

    def forward(self, x, qi=None):
        target_dtype = self._get_target_dtype()

        if self.qw.init_state.item() == 0:
            self.qw.initialize_alpha(self.linear_module.weight.float())
        
        q_weight = self.qw.fake_quantize(self.linear_module.weight)
        
        if self.linear_module.bias is not None:
            if qi:
                bias_scale = self.qw.alpha * qi.alpha
                q_bias = fake_quantize_tensor(self.linear_module.bias.float(), bias_scale, 32)
            else:
                q_bias = self.linear_module.bias.float()
            q_bias_prec = q_bias.to(target_dtype)
        else:
            q_bias_prec = None

        x_prec = x.to(target_dtype)
        q_weight_prec = q_weight.to(target_dtype)

        output_prec = F.linear(x_prec, q_weight_prec, q_bias_prec)

        if hasattr(self, 'qo'):
            if self.qo.init_state.item() == 0:
                self.qo.initialize_alpha(output_prec.float())
            final_output = self.qo.fake_quantize(output_prec)
        else:
            final_output = output_prec.float()
        
        return final_output

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function for QLinear.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed but required for freeze, or should be passed to QLinear constructor.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function for QLinear.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed but required for freeze, or should be passed to QLinear constructor.')

        if qi is not None: # This case implies qi was not an attribute but passed to freeze
            self.qi = qi
        if qo is not None: # Similar for qo
            self.qo = qo
        
        if not hasattr(self, 'qi') or not hasattr(self, 'qo') or not hasattr(self, 'qw'):
            print("Warning: qi, qo, or qw not fully available for M calculation in QLinear. Skipping M calculation.")
        else:
            M_val = (self.qw.alpha.data.item() * self.qi.alpha.data.item()) / self.qo.alpha.data.item()
            self.M.data = torch.tensor(M_val, dtype=torch.float)
            Mo, n_val = search(M_val.item() if isinstance(M_val, torch.Tensor) else M_val)
            self.M0.data = torch.tensor(Mo, dtype=torch.float)
            self.n.data = torch.tensor(n_val, dtype=torch.float)

        self.linear_module.weight.data = self.qw.quantize_tensor(self.linear_module.weight.data)
        if self.linear_module.bias is not None:
            if hasattr(self, 'qi') and hasattr(self, 'qw'): 
                bias_scale = self.qw.alpha.data.item() * self.qi.alpha.data.item()
                if bias_scale != 0:
                    self.linear_module.bias.data = quantize_tensor(self.linear_module.bias.data, scale=bias_scale, num_bits=32)
                else:
                    raise ValueError("Warning: Bias scale is zero. Bias not quantized for linear layer.")
            else:
                print("Warning: qi.alpha or qw.alpha not available for linear bias quantization. Bias not quantized.")
        return self.qo

    def quantize_inference(self, x):
        x = F.linear(x, self.linear_module.weight, self.linear_module.bias)
        x = x*self.M0.data.item()/2**self.n.data.item()
        x = round_ste_for_inference(x)
        x.clamp_(-2**(self.qo.num_bits-1), 2**(self.qo.num_bits-1)-1)
        return x
    
class QAdd(QModule):
    def __init__(self, num_bits, qi=False, q_shortcut=False, qo=True):
        super(QAdd, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.register_buffer('M', torch.zeros(1))
        self.register_buffer('n', torch.zeros(1))
        self.register_buffer('M_shortcut', torch.zeros(1))
        self.register_buffer('n_shortcut', torch.zeros(1))
        self.register_buffer('Mo', torch.zeros(1))
        self.register_buffer('no', torch.zeros(1))

    def forward(self, x, shortcut):
        x = x.to(self._get_target_dtype()) + shortcut.to(self._get_target_dtype())
        
        if hasattr(self, 'qo'):
            if self.qo.init_state.item() == 0:
                self.qo.initialize_alpha(x)
            x = self.qo.fake_quantize(x).to(self._get_target_dtype())
        
        return x
    
    def freeze(self, qi=None, q_shortcut=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function for QAdd.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed but required for freeze, or should be passed to QAdd constructor.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function for QAdd.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed but required for freeze, or should be passed to QAdd constructor.')
        
        if qi is not None:
            self.qi = qi
        if q_shortcut is not None:
            self.q_shortcut = q_shortcut
        if qo is not None:
            self.qo = qo
        
        if not hasattr(self, 'qi') or not hasattr(self, 'qo'):
            print("Warning: qi or qo not fully available for M calculation in QAdd. Skipping M calculation.")
        else:            
            M_val = self.qi.alpha.data.item() / self.qo.alpha.data.item() * 1024
            M_shortcut_val = self.q_shortcut.alpha.data.item() / self.qo.alpha.data.item() * 1024

            Mo, n_val = search(M_val.item() if isinstance(M_val, torch.Tensor) else M_val)
            Mo_shortcut, n_shortcut_val = search(M_shortcut_val.item() if isinstance(M_shortcut_val, torch.Tensor) else M_shortcut_val)
            self.M.data = torch.tensor(Mo, dtype=torch.float)
            self.n.data = torch.tensor(n_val, dtype=torch.float)
            self.M_shortcut.data = torch.tensor(Mo_shortcut, dtype=torch.float)
            self.n_shortcut.data = torch.tensor(n_shortcut_val, dtype=torch.float)
            self.Mo.data = torch.tensor(1, dtype=torch.float)
            self.no.data = torch.tensor(10, dtype=torch.float)
        
        return self.qo

    def quantize_inference(self, x, shortcut):
        x = x * self.M.data.item() / 2**self.n.data.item() + shortcut * self.M_shortcut.data.item() / 2**self.n_shortcut.data.item()
        x = round_ste_for_inference(x)
        x = x * self.Mo.data.item() / 2**self.no.data.item()
        if self.qo.num_bits > 1:
            x.clamp_(-2**(self.qo.num_bits-1), 2**(self.qo.num_bits-1)-1)
        else:
            x.clamp_(0, 1)
        return x
    