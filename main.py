from models.pretrained import ResNet18CIFAR10, ResNet50CIFAR10, ResNet18CIFAR100, ResNet50CIFAR100
from models.supermask import SupermaskNet
from datasets import cifar10_dataloaders, cifar100_dataloaders
from evaluate import get_labels_and_preds, get_overall_accuracy, get_confusion_matrix, get_class_wise_accuracy, get_class_group_accuracy, get_unlearning_accuracy
from unlearning.random_labels import unlearn_one_epoch_with_random_labels
from unlearning.gradient_ascent import unlearn_one_epoch_with_gradient_ascent
from unlearning.chance_level import unlearn_one_epoch_with_chance_level
from unlearning import Unlearner

from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 기본 모델 및 데이터 설정
base_model = ResNet18CIFAR100()
sparsity_values = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
seeds = [0,1,2]  # 사용할 3개의 랜덤 시드
layer_wise = False
methods = [
    'random_pruning', 
    'random_labels', 
    'chance_level', 
    'gradient_ascent',
]

for method in methods:
    forget_classes = [0]
    retain_classes = list(range(1, 100))

    train_loader, val_loader, test_loader = cifar100_dataloaders()

    # sparsity와 seed에 따른 정확도 계산
    for sparsity in sparsity_values:
        for seed in seeds:
            set_seed(seed)  # 각 시드 설정
            model = SupermaskNet(deepcopy(base_model), sparsity=sparsity, layer_wise=layer_wise).cuda()
            optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
            unlearner = Unlearner(model, train_loader, val_loader, test_loader, optimizer, method=method, epochs=50, early_stop_patience=5, save_path=f'../results/synaptag/layer_wise={layer_wise}/method={method}/sparsity={sparsity}/seed={seed}/best_model.pt', device='cuda')
            if method!='random_pruning':
                unlearner.run(forget_classes)
            retain_acc, forget_acc, rfa = get_unlearning_accuracy(
                forget_classes=forget_classes,
                retain_classes=retain_classes,
                model=model,
                data_loader=test_loader
            )