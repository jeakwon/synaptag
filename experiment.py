from models.pretrained import ResNet18CIFAR10, ResNet18CIFAR100, ResNet50CIFAR10, ResNet50CIFAR100
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
import argparse
import yaml
import os
import logging
import shutil
import json
from datetime import datetime

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_experiment_directory(save_dir, layer_wise, method, sparsity, seed):
    """Create a unique experiment directory and set up logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{timestamp}_lw={layer_wise}_m={method}_sp={sparsity}_s={seed}"
    exp_dir = os.path.join(save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(exp_dir, "log.txt")),
            logging.StreamHandler()
        ]
    )
    return exp_dir

def run(
    dataset='cifar100',
    model_type='resnet18',
    sparsity=0.1,
    seed=0,
    layer_wise=False,
    method='random_pruning',
    forget_classes=[0],
    epochs=50,
    early_stop_patience=5,
    save_dir='../results/synaptag',
    optimizer='adamw',
    optimizer_kwargs={'lr': 0.01},
    config_path=None
):
    # Set up experiment directory and logging
    exp_dir = setup_experiment_directory(save_dir, layer_wise, method, sparsity, seed)
    logger = logging.getLogger()
    logger.info(f"Starting experiment in {exp_dir}")

    # Save config
    if config_path:
        shutil.copy(config_path, os.path.join(exp_dir, "config.yaml"))
        logger.info(f"Copied config from {config_path} to {exp_dir}")

    try:
        # Dataset selection
        if dataset == 'cifar100':
            train_loader, val_loader, test_loader = cifar100_dataloaders()
            base_model = ResNet18CIFAR100() if model_type == 'resnet18' else ResNet50CIFAR100()
            all_classes = list(range(100))
        else:  # cifar10
            train_loader, val_loader, test_loader = cifar10_dataloaders()
            base_model = ResNet18CIFAR10() if model_type == 'resnet18' else ResNet50CIFAR10()
            all_classes = list(range(10))

        # Compute retain_classes
        retain_classes = [c for c in all_classes if c not in forget_classes]
        logger.info(f"Forget classes: {forget_classes}, Retain classes: {retain_classes}")

        # Set seed
        set_seed(seed)
        logger.info(f"Set random seed to {seed}")

        # Initialize model
        model = SupermaskNet(deepcopy(base_model), sparsity=sparsity, layer_wise=layer_wise).cuda()
        logger.info(f"Initialized model with sparsity={sparsity}, layer_wise={layer_wise}")

        # Select optimizer
        optimizer_map = {
            'sgd': optim.SGD,
            'adam': optim.Adam,
            'adamw': optim.AdamW
        }
        if optimizer.lower() not in optimizer_map:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Choose from {list(optimizer_map.keys())}")
        
        optimizer_class = optimizer_map[optimizer.lower()]
        optimizer = optimizer_class(
            [p for p in model.parameters() if p.requires_grad],
            **optimizer_kwargs
        )
        logger.info(f"Using optimizer {optimizer} with kwargs: {optimizer_kwargs}")

        # Set save path
        save_path = os.path.join(exp_dir, "best_model.pt")
        
        # Initialize unlearner
        unlearner = Unlearner(
            model,
            train_loader,
            val_loader,
            test_loader,
            optimizer,
            method=method,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            save_path=save_path,
            device='cuda'
        )
        logger.info(f"Initialized unlearner with method={method}, epochs={epochs}")

        # Run experiment
        if method != 'random_pruning':
            logger.info("Starting unlearning process")
            unlearner.run(forget_classes)
        else:
            logger.info("Skipping unlearning for random_pruning method")

        # Compute and save metrics
        logger.info("Computing unlearning accuracy metrics")
        retain_acc, forget_acc, rfa = get_unlearning_accuracy(
            forget_classes=forget_classes,
            retain_classes=retain_classes,
            model=model,
            data_loader=test_loader
        )
        metrics = {
            "retain_acc": float(retain_acc),  # Convert to float for JSON serialization
            "forget_acc": float(forget_acc),
            "rfa": float(rfa)
        }
        metrics_path = os.path.join(exp_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_path}: {metrics}")

        logger.info("Experiment completed successfully")

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unlearning experiments with YAML config')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file')
    
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = [
        'dataset', 'model_type', 'sparsity', 'seed', 'layer_wise',
        'method', 'forget_classes', 'epochs', 'early_stop_patience', 'save_dir',
        'optimizer', 'optimizer_kwargs'
    ]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in YAML config")

    # Run experiment
    run(
        dataset=config['dataset'],
        model_type=config['model_type'],
        sparsity=config['sparsity'],
        seed=config['seed'],
        layer_wise=config['layer_wise'],
        method=config['method'],
        forget_classes=config['forget_classes'],
        epochs=config['epochs'],
        early_stop_patience=config['early_stop_patience'],
        save_dir=config['save_dir'],
        optimizer=config['optimizer'],
        optimizer_kwargs=config['optimizer_kwargs'],
        config_path=args.config
    )