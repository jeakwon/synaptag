from models.ResNet import resnet18, resnet50
from models.supermask import SupermaskNet
from huggingface_hub import hf_hub_download
import torch

__all__ = [
    "SupermaskResNet18CIFAR10",
    "SupermaskResNet50CIFAR10",
    "SupermaskResNet18CIFAR100",
    "SupermaskResNet50CIFAR100",
]

def SupermaskResNet18CIFAR10(sparsity=0.1, layer_wise=True):
    base_model = resnet18(num_classes=10)
    file_path = hf_hub_download(repo_id="onlytojay/engram", filename="base_models/cifar10/resnet18/0model_SA_best.pth.tar")
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
    base_model.load_state_dict(checkpoint['state_dict'])
    model = SupermaskNet(base_model, sparsity=sparsity, layer_wise=layer_wise)
    return model


def SupermaskResNet50CIFAR10(sparsity=0.1, layer_wise=True):
    base_model = resnet50(num_classes=10)
    file_path = hf_hub_download(repo_id="onlytojay/engram", filename="base_models/cifar10/resnet50/0model_SA_best.pth.tar")
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
    base_model.load_state_dict(checkpoint['state_dict'])
    model = SupermaskNet(base_model, sparsity=sparsity, layer_wise=layer_wise)
    return model

def SupermaskResNet18CIFAR100(sparsity=0.01, layer_wise=True):
    base_model = resnet18(num_classes=100)
    file_path = hf_hub_download(repo_id="onlytojay/engram", filename="base_models/cifar100/resnet18/0model_SA_best.pth.tar")
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
    base_model.load_state_dict(checkpoint['state_dict'])
    model = SupermaskNet(base_model, sparsity=sparsity, layer_wise=layer_wise)
    return model


def SupermaskResNet50CIFAR100(sparsity=0.01, layer_wise=True):
    base_model = resnet50(num_classes=100)
    file_path = hf_hub_download(repo_id="onlytojay/engram", filename="base_models/cifar100/resnet50/0model_SA_best.pth.tar")
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
    base_model.load_state_dict(checkpoint['state_dict'])
    model = SupermaskNet(base_model, sparsity=sparsity, layer_wise=layer_wise)
    return model

