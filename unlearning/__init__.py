import os
import torch

from unlearning.random_labels import unlearn_one_epoch_with_random_labels
from unlearning.gradient_ascent import unlearn_one_epoch_with_gradient_ascent
from unlearning.chance_level import unlearn_one_epoch_with_chance_level
from evaluate import get_unlearning_accuracy



class Unlearner:
    def __init__(self, model, train_loader, valid_loader, test_loader, optimizer, method='chance_level', epochs=50, early_stop_patience=5, save_path='./best_model.pt', device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.method = method
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.save_path = save_path
        self.device = device
        
        self.unlearn_methods = {
            'random_labels': unlearn_one_epoch_with_random_labels,
            'gradient_ascent': unlearn_one_epoch_with_gradient_ascent,
            'chance_level': unlearn_one_epoch_with_chance_level
        }
        if self.method not in self.unlearn_methods:
            raise ValueError(f"Method must be one of {list(self.unlearn_methods.keys())}")
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def run(self, forget_classes):
        retain_classes = [i for i in range(len(self.train_loader.dataset.classes)) if i not in forget_classes]
        
        baseline_retain_acc, baseline_forget_acc, baseline_rfa = get_unlearning_accuracy(retain_classes, forget_classes, self.model, self.valid_loader)
        print(f"**{self.method} Baseline**")
        print(f"Retain Acc.: {baseline_retain_acc:10.2%} | Forget Acc.: {baseline_forget_acc:10.2%} | RFA: {baseline_rfa:10.2%}")

        best_rfa = baseline_rfa
        best_epoch = 0
        epochs_without_improvement = 0
        torch.save(self.model.state_dict(), self.save_path)

        unlearn_fn = self.unlearn_methods[self.method]
        for epoch in range(1, self.epochs + 1):
            total_loss, retain_loss, forget_loss = unlearn_fn(retain_classes, forget_classes, self.model, self.train_loader, self.optimizer, device=self.device)
            retain_acc, forget_acc, rfa = get_unlearning_accuracy(retain_classes, forget_classes, self.model, self.valid_loader)
            
            print(f"[Epoch {epoch:2d}] Loss: {total_loss:10.4f} | Retain Acc.: {retain_acc:10.2%} | Forget Acc.: {forget_acc:10.2%} | RFA: {rfa:10.2%}", end='')
            
            if rfa > best_rfa:
                best_rfa = rfa
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.save_path)
                epochs_without_improvement = 0
                print(f" <-- BEST (Saved as {self.save_path})")
            else:
                epochs_without_improvement += 1
                print("")

            if epochs_without_improvement >= self.early_stop_patience:
                print(f"Early stopping triggered. No improvement in RFA for {self.early_stop_patience} consecutive epochs.")
                break

        print(f"\nTraining finished. Loading best model for {self.method} testset evaluation...")
        self.model.load_state_dict(torch.load(self.save_path, weights_only=True))
        endpoint_retain_acc, endpoint_forget_acc, endpoint_rfa = get_unlearning_accuracy(retain_classes, forget_classes, self.model, self.test_loader)
        print(f"**{self.method} Endpoint**")
        print(f"Retain Acc.: {endpoint_retain_acc:10.2%} | Forget Acc.: {endpoint_forget_acc:10.2%} | RFA: {endpoint_rfa:10.2%}")

        result = {
            'method': self.method,
            'best_epoch': best_epoch,
            'best_rfa': best_rfa,
            'baseline_retain_acc': baseline_retain_acc,
            'baseline_forget_acc': baseline_forget_acc,
            'baseline_rfa': baseline_rfa,
            'endpoint_retain_acc': endpoint_retain_acc,
            'endpoint_forget_acc': endpoint_forget_acc,
            'endpoint_rfa': endpoint_rfa
        }
        return result