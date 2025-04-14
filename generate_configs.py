import os
import yaml
import itertools
from datetime import datetime

def generate_configs(config_dir="configs", base_config=None):
    """Generate YAML config files for experiments."""
    os.makedirs(config_dir, exist_ok=True)
    
    # Default base config
    if base_config is None:
        base_config = {
            "dataset": "cifar100",
            "model_type": "resnet18",
            "seed": 0,
            "layer_wise": False,
            "method": "random_pruning",
            "epochs": 50,
            "early_stop_patience": 5,
            "save_dir": "../results"
        }
    
    # Experiment variations
    sparsity_values = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    optimizers = [
        {"name": "adamw", "kwargs": {"lr": 0.01, "weight_decay": 0.01}},
    ]
    forget_classes_list = [[i] for i in range(10)]  # [[0], [1], ..., [9]]
    
    # Generate all combinations
    for sparsity, opt, forget_classes in itertools.product(sparsity_values, optimizers, forget_classes_list):
        config = base_config.copy()
        config["sparsity"] = sparsity
        config["optimizer"] = opt["name"]
        config["optimizer_kwargs"] = opt["kwargs"]
        config["forget_classes"] = forget_classes
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        forget_str = f"fc={forget_classes[0]}"  # e.g., fc=0 for [0]
        config_name = f"config_sp={sparsity}_opt={opt['name']}_{forget_str}_{timestamp}.yaml"
        config_path = os.path.join(config_dir, config_name)
        
        # Save config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Generated config: {config_path}")

if __name__ == "__main__":
    generate_configs(config_dir="configs")