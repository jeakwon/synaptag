import torch

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
    return all_labels, all_preds
    
