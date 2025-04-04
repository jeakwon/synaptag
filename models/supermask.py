import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold):
        return (scores >= threshold).float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.threshold = 0

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.threshold)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.threshold = 0

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.threshold)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)

# SupermaskNet class
class SupermaskNet(nn.Module):
    def __init__(self, base_model, sparsity, layer_wise=True):
        super().__init__()
        self.base_model = base_model.eval()
        self._sparsity = sparsity  # set as internal variable
        self.layer_wise = layer_wise

        self.net = replace_layers(self.base_model)
        self._update_thresholds()  # calculate threshold when reinitialize

    @property
    def sparsity(self):
        return self._sparsity

    @sparsity.setter
    def sparsity(self, value):
        self._sparsity = value
        self._update_thresholds()  # sparsity changes whenever threshold updates

    def _update_thresholds(self):
        # Layer-wise threshold calculate
        if self.layer_wise:
            for name, module in self.named_modules():
                if isinstance(module, (SupermaskConv, SupermaskLinear)):
                    abs_scores = module.scores.abs().flatten()
                    _, idx = abs_scores.sort()
                    j = int(self._sparsity * abs_scores.numel())
                    module.threshold = abs_scores[idx[j]]

        # Global threshold calculate
        else:
            all_scores = []
            for name, module in self.named_modules():
                if isinstance(module, (SupermaskConv, SupermaskLinear)):
                    abs_scores = module.scores.abs().flatten()
                    all_scores.append(abs_scores)
            all_scores = torch.cat(all_scores)
            _, idx = all_scores.sort()
            j = int(self._sparsity * all_scores.numel())
            threshold = all_scores[idx[j]]
            for name, module in self.named_modules():
                if isinstance(module, (SupermaskConv, SupermaskLinear)):
                    module.threshold = threshold

    def forward(self, x):
        return self.net(x)

    def get_masks(self):
        masks = {}
        for name, module in self.named_modules():
            if isinstance(module, (SupermaskConv, SupermaskLinear)):
                masks[name] = (module.scores.abs() >= module.threshold).float()
        return masks

    def get_masks(self):
        masks = {}
        for name, module in self.named_modules():
            if isinstance(module, (SupermaskConv, SupermaskLinear)):
                masks[name] = (module.scores.abs() >= module.threshold).float()
        return masks

def replace_layers(model):
    """ Recursively replace Conv2d and Linear layers with Supermask versions """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new_layer = SupermaskConv(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            )
            new_layer.weight.data = module.weight.data.clone()  # Copy original weights
            if module.bias is not None:
                new_layer.bias.data = module.bias.data.clone()
            setattr(model, name, new_layer)

        elif isinstance(module, nn.Linear):
            new_layer = SupermaskLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            new_layer.weight.data = module.weight.data.clone()  # Copy original weights
            if module.bias is not None:
                new_layer.bias.data = module.bias.data.clone()
            setattr(model, name, new_layer)

        else:
            replace_layers(module)  # Recursively apply to submodules

    """Freeze all parameters except scores in Supermask layers"""
    for name, param in model.named_parameters():
        param.requires_grad = False  # Freeze all parameters

    for name, module in model.named_modules():
        if isinstance(module, (SupermaskConv, SupermaskLinear)):
            module.scores.requires_grad = True  # Unfreeze only scores

    return model
