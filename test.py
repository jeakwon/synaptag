
from models.pretrained import ResNet18CIFAR10, ResNet50CIFAR10, ResNet18CIFAR100, ResNet50CIFAR100
from models.supermask import SupermaskNet
from datasets import cifar10_dataloaders, cifar100_dataloaders
from evaluate import get_labels_and_preds, get_overall_accuracy, get_confusion_matrix, get_class_wise_accuracy, get_class_group_accuracy
from unlearning.random_labels import unlearn_one_epoch_with_random_labels
from unlearning.gradient_ascent import unlearn_one_epoch_with_gradient_ascent

from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim


torch.autograd.set_detect_anomaly(True)
base_model = ResNet18CIFAR10()
model = SupermaskNet(deepcopy(base_model), sparsity=0.01, layer_wise=True).cuda()
train_loader, val_loader, test_loader = cifar10_dataloaders()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.05)

forget_classes = [0]
retain_classes = [1,2,3,4,5,6,7,8,9]

labels, preds = get_labels_and_preds(model, test_loader)
class_wise_acc = get_class_wise_accuracy(labels, preds)
print(class_wise_acc)

unlearn_one_epoch_with_gradient_ascent(retain_classes, forget_classes, model, train_loader, criterion, optimizer, device='cuda')

labels, preds = get_labels_and_preds(model, test_loader)
class_wise_acc = get_class_wise_accuracy(labels, preds)
print(class_wise_acc)

# cm = get_confusion_matrix(labels, preds)
# acc = get_overall_accuracy(labels, preds)
# retain_acc = get_class_group_accuracy(labels, preds, retain_classes)
# forget_acc = get_class_group_accuracy(labels, preds, forget_classes)
