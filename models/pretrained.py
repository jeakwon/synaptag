from models.ResNet import resnet18, resnet50
from huggingface_hub import hf_hub_download
import torch

__all__ = [
    "ResNet18CIFAR10",
    "ResNet50CIFAR10",
    "ResNet18CIFAR100",
    "ResNet50CIFAR100",
]

def ResNet18CIFAR10():
    model = resnet18(num_classes=10)
    file_path = hf_hub_download(repo_id="onlytojay/engram", filename="base_models/cifar10/resnet18/0model_SA_best.pth.tar")
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def ResNet50CIFAR10():
    model = resnet50(num_classes=10)
    file_path = hf_hub_download(repo_id="onlytojay/engram", filename="base_models/cifar10/resnet50/0model_SA_best.pth.tar")
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def ResNet18CIFAR100():
    model = resnet18(num_classes=100)
    file_path = hf_hub_download(repo_id="onlytojay/engram", filename="base_models/cifar100/resnet18/0model_SA_best.pth.tar")
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def ResNet50CIFAR100():
    model = resnet50(num_classes=100)
    file_path = hf_hub_download(repo_id="onlytojay/engram", filename="base_models/cifar100/resnet50/0model_SA_best.pth.tar")
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return model

