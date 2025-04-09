import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def unlearn_one_epoch_with_random_labels(retain_classes, forget_classes, model, dataloader, optimizer, device='cuda'):
    model.train()
    total_loss = 0.0
    total_forget_loss = 0.0
    total_retain_loss = 0.0
    total_retain_samples = 0
    total_forget_samples = 0
    
    all_classes = np.arange(len(dataloader.dataset.classes))
    if not isinstance(all_classes, (list, np.ndarray)):
        raise ValueError("dataloader.dataset.classes must be a list or array of class labels.")
    
    candidate_classes = [cls for cls in all_classes if cls not in forget_classes]
    if not candidate_classes:
        raise ValueError("No available classes for random label assignment after excluding forget_classes.")
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        retain_mask = torch.isin(labels, torch.tensor(retain_classes, device=device))
        forget_mask = torch.isin(labels, torch.tensor(forget_classes, device=device))
        
        total_retain_samples += retain_mask.sum().item()
        total_forget_samples += forget_mask.sum().item()
        
        if forget_mask.sum() > 0:
            num_forget_samples = forget_mask.sum().item()
            forget_labels = labels[forget_mask].cpu().numpy()
            random_labels = []
            
            for orig_label in forget_labels:
                available_classes = [c for c in all_classes if c != orig_label]
                random_label = np.random.choice(available_classes)
                random_labels.append(random_label)
            
            random_labels = torch.tensor(random_labels, dtype=torch.long, device=device)
            labels[forget_mask] = random_labels
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if retain_mask.any():
            retain_loss = F.cross_entropy(outputs[retain_mask], labels[retain_mask])
        else:
            retain_loss = torch.tensor(0.0, device=device)
        
        if forget_mask.any():
            forget_loss = F.cross_entropy(outputs[forget_mask], labels[forget_mask])
        else:
            forget_loss = torch.tensor(0.0, device=device)
        
        loss = retain_loss + forget_loss
        
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_retain_loss += retain_loss.item() * retain_mask.sum().item()
        total_forget_loss += forget_loss.item() * forget_mask.sum().item()
    
    num_samples = len(dataloader.dataset)
    avg_total_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_retain_loss = total_retain_loss / total_retain_samples if total_retain_samples > 0 else 0.0
    avg_forget_loss = total_forget_loss / total_forget_samples if total_forget_samples > 0 else 0.0

    return avg_total_loss, avg_retain_loss, avg_forget_loss