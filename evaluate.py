import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def get_labels_and_preds(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            # Move to GPU if available
            if torch.cuda.is_available():
                model.cuda()
                images = images.cuda()
                labels = labels.cuda()
            
            # Get model predictions
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)
    
def get_confusion_matrix(true_labels, pred_labels):
    return confusion_matrix(true_labels, pred_labels)

def get_overall_accuracy(true_labels, pred_labels):
    return np.mean(true_labels == pred_labels)

def get_class_wise_accuracy(true_labels, pred_labels, target_classes=None):
    unique_classes = np.unique(true_labels)
    
    if target_classes is None:
        target_classes = unique_classes
    
    target_classes = [int(cls) for cls in target_classes if cls in unique_classes]
    if not target_classes:
        raise ValueError("No valid target classes found in the data.")
    
    class_accuracies = {}
    for class_idx in target_classes:
        class_mask = (true_labels == class_idx)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(pred_labels[class_mask] == true_labels[class_mask])
            class_accuracies[class_idx] = float(class_acc)
        else:
            class_accuracies[class_idx] = 0.0
    return class_accuracies

def get_class_group_accuracy(true_labels, pred_labels, target_classes=None):
    unique_classes = np.unique(true_labels)
    
    if target_classes is None:
        target_classes = unique_classes
    
    target_classes = [cls for cls in target_classes if cls in unique_classes]
    if not target_classes:
        raise ValueError("No valid target classes found in the data.")
    
    group_mask = np.isin(true_labels, target_classes)
    if np.sum(group_mask) > 0:
        group_accuracy = np.mean(pred_labels[group_mask] == true_labels[group_mask])
    else:
        group_accuracy = 0.0
    
    return group_accuracy