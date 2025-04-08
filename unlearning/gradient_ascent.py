import torch
import numpy as np

def unlearn_one_epoch_with_gradient_ascent(retain_classes, forget_classes, model, dataloader, criterion, optimizer, device='cuda'):
    model.train()
    total_loss = 0.0
    total_forget_loss = 0.0
    total_retain_loss = 0.0
    total_retain_samples = 0
    total_forget_samples = 0
    
    all_classes = np.arange(len(dataloader.dataset.classes))
    if not isinstance(all_classes, (list, np.ndarray)):
        raise ValueError("dataloader.dataset.classes must be a list or array of class labels.")
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # retain과 forget 마스크 생성
        retain_mask = torch.isin(labels, torch.tensor(retain_classes, device=device))
        forget_mask = torch.isin(labels, torch.tensor(forget_classes, device=device))
        
        total_retain_samples += retain_mask.sum().item()
        total_forget_samples += forget_mask.sum().item()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Retain 손실 계산 (Gradient Descent)
        if retain_mask.any():
            retain_loss = criterion(outputs[retain_mask], labels[retain_mask])
        else:
            retain_loss = torch.tensor(0.0, device=device)
        
        # Forget 손실 계산 (Gradient Ascent)
        if forget_mask.any():
            forget_loss = criterion(outputs[forget_mask], labels[forget_mask])
        else:
            forget_loss = torch.tensor(0.0, device=device)
        
        # 총 손실: retain_loss는 최소화, forget_loss는 최대화
        loss = retain_loss - forget_loss  # Gradient Ascent를 위해 forget_loss에 음수 적용
        
        # Backward pass 및 최적화
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        
        # 손실 누적 (참고: total_loss는 방향을 반영하므로 음수일 수 있음)
        total_loss += loss.item() * images.size(0)
        total_retain_loss += retain_loss.item() * retain_mask.sum().item()
        total_forget_loss += forget_loss.item() * forget_mask.sum().item()  # 원래 손실 값으로 기록
    
    # 평균 손실 계산
    num_samples = len(dataloader.dataset)
    avg_total_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_retain_loss = total_retain_loss / total_retain_samples if total_retain_samples > 0 else 0.0
    avg_forget_loss = total_forget_loss / total_forget_samples if total_forget_samples > 0 else 0.0

    return avg_total_loss, avg_retain_loss, avg_forget_loss
