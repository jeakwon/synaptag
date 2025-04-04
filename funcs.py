import requests
from io import BytesIO
import torch
import torch.nn.functional as F
import timm

def get_model(url, num_classes):
    response = requests.get(url)
    if response.status_code == 200:
        file_content = BytesIO(response.content)
    else:
        raise Exception("Failed to download the file")
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(file_content, map_location=torch.device('cpu'), weights_only=True)  # Use 'cuda' for GPU
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def selected_class_accuracy(model, data_loader, selected_classes, device, verbose=False):
    """Evaluate the model accuracy for selected classes."""
    model.eval()
    total_acc = 0.0
    total_samples = 0

    num_classes = len(data_loader.dataset.classes)
    existing_classes = torch.arange(num_classes).to(device)
    selected_classes = torch.as_tensor(selected_classes).to(device)
    selected_class_indices = torch.where( torch.isin(existing_classes, selected_classes) )[0]


    assert torch.isin( selected_classes, existing_classes ).all().item(), f'Selected classes must be included in dataloaders partition'

    if len(data_loader)==0:
        return 0.0

    with torch.no_grad():
        for _, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = F.one_hot(labels, num_classes=num_classes).float().to(device)

            # Filter predictions and labels by selected classes
            mask = torch.isin(labels.argmax(dim=1), selected_class_indices)
            if not mask.any():
                continue  # Skip this batch if no selected class exists

            images = images[mask]
            labels = labels[mask]

            pred = model(images)[:, existing_classes] # Matchs with data_loader label order
            pred_labels = pred.argmax(dim=1)
            true_labels = labels.argmax(dim=1)

            # Compute accuracy for selected classes
            correct_predictions = (pred_labels == true_labels).sum().item()
            batch_size = len(true_labels)

            total_acc += correct_predictions
            total_samples += batch_size

    acc = total_acc / total_samples if total_samples > 0 else 0.0
    return acc


def unlearn_one_epoch(forget_classes, model, data_loader, criterion, optimizer, device='cuda'):
    model.train()
    avg_loss = 0.0

    num_classes = len(data_loader.dataset.classes)
    entire_classes = torch.arange(num_classes).to(device)
    forget_classes = torch.as_tensor(forget_classes).to(device)
    forget_indices = torch.where( torch.isin(entire_classes, forget_classes) )[0]

    assert torch.isin( forget_classes, entire_classes ).all().item(), f'Selected classes must be included in data_loader partition'
    assert len(data_loader)>0, 'Data loader is empty'

    for _, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = F.one_hot(labels, num_classes=num_classes).float().to(device)

        # Set selected class labels to chance level (1/num_classes)
        mask = torch.isin(labels.argmax(dim=1), forget_indices)
        labels[mask] = torch.ones_like(labels[mask]) / len(entire_classes)

        # Rearrange model outputs to match data loader class setting.
        outputs = model(images) # (batch_size x 100)
        outputs = outputs[:, entire_classes] # (batch_size x num_classes) & change order

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

    avg_loss /= len(data_loader)
    return avg_loss

def unlearning_accuacy(forget_classes, model, data_loader, device='cuda'):
    entire_classes = torch.arange( len(data_loader.dataset.classes)).to(device)
    forget_classes = torch.as_tensor( forget_classes).to(device)
    retain_classes = torch.as_tensor( [c for c in entire_classes if c not in forget_classes]).to(device)

    forget_acc = selected_class_accuracy(model, data_loader, forget_classes, device='cuda')
    retain_acc = selected_class_accuracy(model, data_loader, retain_classes, device='cuda')
    rfa = retain_acc - forget_acc
    return retain_acc, forget_acc, rfa

class Unlearning:
    def __init__(self,
                 model,
                 train_loader,
                 valid_loader,
                 test_loader,
                 criterion,
                 optimizer,
                 epochs=50,
                 early_stop_patience=5,
                 save_path='./best_model.pt',
                 device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.save_path = save_path
        self.device = device

    def run(self, forget_classes):
        baseline_retain_acc, baseline_forget_acc, baseline_rfa = unlearning_accuacy(forget_classes, self.model, self.valid_loader, device=self.device)

        print(f"**Training** Loss <- Trainset | Retain Acc. <- Validset | Forget Acc. <- Validset | RFA <- Validset")
        print(f"[ Baseline ] Loss: {'N/A':>10} | Retain Acc.: {baseline_retain_acc:10.2%} | Forget Acc.: {baseline_forget_acc:10.2%} | RFA: {baseline_rfa:10.2%}")

        best_rfa = baseline_rfa
        best_epoch = 0
        epochs_without_improvement = 0
        torch.save(self.model.state_dict(), self.save_path)
        for epoch in range(1, self.epochs+1):
            loss = unlearn_one_epoch(forget_classes, self.model, self.train_loader, self.criterion, self.optimizer, device=self.device)

            retain_acc, forget_acc, rfa = unlearning_accuacy(forget_classes, self.model, self.valid_loader, device=self.device
                                                             )
            if rfa > best_rfa:
                best_rfa = rfa
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.save_path)
                epochs_without_improvement = 0
                mark_best = f' <-- BEST (Saved as {self.save_path})'
            else:
                mark_best = ''
                epochs_without_improvement += 1

            print(f"[ Epoch {epoch:2d} ] Loss: {loss:10.4f} | Retain Acc.: {retain_acc:10.2%} | Forget Acc.: {forget_acc:10.2%} | RFA: {rfa:10.2%}{mark_best}")

            if epochs_without_improvement >= self.early_stop_patience:
                print(f"Early stopping triggered. No improvement in RFA for {self.early_stop_patience} consecutive epochs.")
                break


        print("\nTraining finished. Loading best model for testset evaluation...")

        self.model.load_state_dict(torch.load(self.save_path, weights_only=True))
        endpoint_retain_acc, endpoint_forget_acc, endpoint_rfa = unlearning_accuacy(forget_classes, self.model, self.test_loader, device=self.device)
        print(f"**Testing*** Loss <- Trainset  | Retain Acc. <- Testset  | Forget Acc. <- Testset  | RFA <- Testset ")
        print(f"[ Endpoint ] Loss: {'N/A':>10} | Retain Acc.: {endpoint_retain_acc:10.2%} | Forget Acc.: {endpoint_forget_acc:10.2%} | RFA: {endpoint_rfa:10.2%}")

        result = dict(
            best_epoch=best_epoch,
            best_rfa=best_rfa,
            baseline_retain_acc=baseline_retain_acc,
            baseline_forget_acc=baseline_forget_acc,
            baseline_rfa=baseline_rfa,
            endpoint_retain_acc=endpoint_retain_acc,
            endpoint_forget_acc=endpoint_forget_acc,
            endpoint_rfa=endpoint_rfa,
        )
        return result