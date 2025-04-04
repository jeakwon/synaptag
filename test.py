from models.zoo import SupermaskResNet18CIFAR10, SupermaskResNet50CIFAR10, SupermaskResNet18CIFAR100, SupermaskResNet50CIFAR100
model = SupermaskResNet18CIFAR10(sparsity=0.01, layer_wise=True)
# model = SupermaskResNet18CIFAR100(sparsity=0.01, layer_wise=True)
# model = SupermaskResNet50CIFAR10(sparsity=0.01, layer_wise=True)
# model = SupermaskResNet50CIFAR100(sparsity=0.01, layer_wise=True)

from datasets import cifar10_dataloaders, cifar100_dataloaders
train_loader, val_loader, test_loader = cifar10_dataloaders()
# train_loader, val_loader, test_loader = cifar100_dataloaders()

from evaluate import get_labels_and_preds

labels, preds = get_labels_and_preds(model, test_loader)

print(labels[:10], preds[:10])
print("successful")



# import torch
# import numpy as np
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# def get_confusion_matrix(model, test_loader, num_classes=10):
#     # Set model to evaluation mode
#     model.eval()
    
#     # Lists to store predictions and true labels
#     all_preds = []
#     all_labels = []
    
#     # Disable gradient computation for evaluation
#     with torch.no_grad():
#         for images, labels in test_loader:
#             # Move to GPU if available
#             if torch.cuda.is_available():
#                 images = images.cuda()
#                 labels = labels.cuda()
            
#             # Get model predictions
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)
            
#             # Store predictions and labels
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
    
#     # Create confusion matrix
#     cm = confusion_matrix(all_labels, all_preds)
    
#     return cm

# # Get the confusion matrix
# confusion_mat = get_confusion_matrix(model.cuda(), test_loader)

# # Optional: Visualize the confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix for SupermaskResNet18CIFAR10')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# # plt.show()
# plt.savefig('test.png')

# # Print the confusion matrix
# print("Confusion Matrix:")
# print(confusion_mat)