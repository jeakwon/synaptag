from models.pretrained import ResNet18CIFAR10, ResNet50CIFAR10, ResNet18CIFAR100, ResNet50CIFAR100
from models.supermask import SupermaskNet
from datasets import cifar10_dataloaders, cifar100_dataloaders
from evaluate import get_unlearning_accuracy
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd

# 랜덤 시드 설정 함수
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 기본 모델 및 데이터 설정
base_model = ResNet18CIFAR100()
sparsity_values = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
seeds = [42, 123, 777]  # 사용할 3개의 랜덤 시드
methods = ['random_pruning', 'random_labels', 'gradient_ascent', 'chance_level']
layer_wise_options = [False, True]  # global과 local

# forget 및 retain 클래스 정의
forget_classes = [0]
retain_classes = list(range(1, 100))

# 데이터로더 생성
_, _, test_loader = cifar100_dataloaders()

# 결과 저장 및 로드 함수
def get_eval_result(method, sparsity, seed, layer_wise):
    eval_file = f'../results/synaptag/layer_wise={layer_wise}/method={method}/sparsity={sparsity}/seed={seed}/eval_results.json'
    model_file = f'../results/synaptag/layer_wise={layer_wise}/method={method}/sparsity={sparsity}/seed={seed}/best_model.pt'
    
    # 중간 결과 파일이 있으면 로드
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded eval result from {eval_file}")
        return data['retain_acc'], data['forget_acc'], data['rfa']
    
    # 없으면 모델 로드 후 계산 및 저장
    if os.path.exists(model_file):
        set_seed(seed)
        model = SupermaskNet(deepcopy(base_model), sparsity=sparsity, layer_wise=layer_wise).cuda()
        model.load_state_dict(torch.load(model_file, weights_only=True))
        retain_acc, forget_acc, rfa = get_unlearning_accuracy(
            forget_classes=forget_classes,
            retain_classes=retain_classes,
            model=model,
            data_loader=test_loader
        )
        # 결과 저장
        result = {'retain_acc': float(retain_acc), 'forget_acc': float(forget_acc), 'rfa': float(rfa)}
        os.makedirs(os.path.dirname(eval_file), exist_ok=True)
        with open(eval_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"Computed and saved eval result to {eval_file}")
        return retain_acc, forget_acc, rfa
    else:
        print(f"Model not found: {model_file}")
        return None, None, None

# 데이터 계산 및 저장
results = {str(layer_wise): {method: {sparsity: {'retain_accs': [], 'forget_accs': [], 'rfas': []} for sparsity in sparsity_values} for method in methods} for layer_wise in layer_wise_options}

for layer_wise in layer_wise_options:
    for method in methods:
        for sparsity in sparsity_values:
            for seed in seeds:
                retain_acc, forget_acc, rfa = get_eval_result(method, sparsity, seed, layer_wise)
                if retain_acc is not None:
                    results[str(layer_wise)][method][sparsity]['retain_accs'].append(retain_acc)
                    results[str(layer_wise)][method][sparsity]['forget_accs'].append(forget_acc)
                    results[str(layer_wise)][method][sparsity]['rfas'].append(rfa)

# 평균과 표준편차 계산 및 CSV 저장
avg_results = {str(layer_wise): {method: {'avg_retain_accs': [], 'avg_forget_accs': [], 'avg_rfas': [], 'std_retain_accs': []} for method in methods} for layer_wise in layer_wise_options}
csv_data = []

for layer_wise in layer_wise_options:
    for method in methods:
        for sparsity in sparsity_values:
            retain_accs = results[str(layer_wise)][method][sparsity]['retain_accs']
            forget_accs = results[str(layer_wise)][method][sparsity]['forget_accs']
            rfas = results[str(layer_wise)][method][sparsity]['rfas']
            if retain_accs:  # 데이터가 있는 경우만 계산
                avg_retain = np.mean(retain_accs)
                avg_forget = np.mean(forget_accs)
                avg_rfa = np.mean(rfas)
                std_retain = np.std(retain_accs)
                avg_results[str(layer_wise)][method]['avg_retain_accs'].append(avg_retain)
                avg_results[str(layer_wise)][method]['avg_forget_accs'].append(avg_forget)
                avg_results[str(layer_wise)][method]['avg_rfas'].append(avg_rfa)
                avg_results[str(layer_wise)][method]['std_retain_accs'].append(std_retain)
                print(f"Layer Wise: {layer_wise} | Method: {method} | Sparsity: {sparsity:.4f} | Avg Retain Acc.: {avg_retain:10.2%} | Avg Forget Acc.: {avg_forget:10.2%} | Avg RFA: {avg_rfa:10.2%} | Std Retain: {std_retain:.4f}")
                csv_data.append({
                    'layer_wise': layer_wise,
                    'method': method,
                    'sparsity': sparsity,
                    'avg_retain_acc': avg_retain,
                    'avg_forget_acc': avg_forget,
                    'avg_rfa': avg_rfa,
                    'std_retain_acc': std_retain
                })

# CSV 파일로 저장
csv_file = '../results/synaptag/cifar100_unlearning_avg_results.csv'
pd.DataFrame(csv_data).to_csv(csv_file, index=False)
print(f"Average results saved to {csv_file}")

# CSV에서 데이터 로드
df = pd.read_csv(csv_file)

# Matplotlib으로 그래프 그리기 (두 가지 플롯)
colors = {
    'random_pruning': 'gray',
    'random_labels': 'blue',
    'gradient_ascent': 'green',
    'chance_level': 'red'
}
styles = {
    False: '-',  # global: 실선
    True: ':'   # local: 점선
}

# 1. Retain vs Forget 그래프
plt.figure(figsize=(12, 12))
for layer_wise in layer_wise_options:
    label_prefix = 'global' if not layer_wise else 'local'
    layer_df = df[df['layer_wise'] == layer_wise]
    for method in methods:
        method_df = layer_df[layer_df['method'] == method]
        avg_retain_accs = method_df['avg_retain_acc'].values
        avg_forget_accs = method_df['avg_forget_acc'].values
        std_retain_accs = method_df['std_retain_acc'].values
        
        min_size = 5
        max_size = 100
        min_std = min(std_retain_accs) if len(std_retain_accs) > 0 else 0
        max_std = max(std_retain_accs) if len(std_retain_accs) > 0 else 0
        if max_std > min_std:
            sizes = [min_size + (max_size - min_size) * (std - min_std) / (max_std - min_std) for std in std_retain_accs]
        else:
            sizes = [min_size] * len(std_retain_accs)

        plt.plot(avg_retain_accs, avg_forget_accs, color=colors[method], linestyle=styles[layer_wise], linewidth=2, label=f"{label_prefix}_{method}")
        plt.scatter(avg_retain_accs, avg_forget_accs, s=sizes, color=colors[method], alpha=0.5)
        for i, sparsity in enumerate(method_df['sparsity'].values):
            plt.annotate(f"{sparsity:.4f}", (avg_retain_accs[i], avg_forget_accs[i]), textcoords="offset points", xytext=(0, 10), ha='center')

plt.xlabel('Retain Accuracy (%)')
plt.ylabel('Forget Accuracy (%)')
plt.title('Avg Retain vs Avg Forget Accuracy for Different Methods (CIFAR-100)')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.xticks(np.arange(-0.1, 1.2, 0.2), [f"{x:.0%}" for x in np.arange(-0.1, 1.2, 0.2)])
plt.yticks(np.arange(-0.1, 1.2, 0.2), [f"{x:.0%}" for x in np.arange(-0.1, 1.2, 0.2)])
plt.grid(True, which="both", ls="--")
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('cifar100_retain_vs_forget_all_methods_local_global.png')
plt.show()

# 2. Sparsity vs RFA 그래프
plt.figure(figsize=(12, 6))
for layer_wise in layer_wise_options:
    label_prefix = 'global' if not layer_wise else 'local'
    layer_df = df[df['layer_wise'] == layer_wise]
    for method in methods:
        method_df = layer_df[layer_df['method'] == method]
        sparsity_vals = method_df['sparsity'].values
        avg_rfas = method_df['avg_rfa'].values
        std_retain_accs = method_df['std_retain_acc'].values
        
        min_size = 5
        max_size = 100
        min_std = min(std_retain_accs) if len(std_retain_accs) > 0 else 0
        max_std = max(std_retain_accs) if len(std_retain_accs) > 0 else 0
        if max_std > min_std:
            sizes = [min_size + (max_size - min_size) * (std - min_std) / (max_std - min_std) for std in std_retain_accs]
        else:
            sizes = [min_size] * len(std_retain_accs)

        plt.plot(sparsity_vals, avg_rfas, color=colors[method], linestyle=styles[layer_wise], linewidth=2, label=f"{label_prefix}_{method}")
        plt.scatter(sparsity_vals, avg_rfas, s=sizes, color=colors[method], alpha=0.5)

plt.xlabel('Sparsity')
plt.ylabel('RFA (%)')
plt.title('Sparsity vs Avg RFA for Different Methods (CIFAR-100)')
plt.xscale('log')  # sparsity는 로그 스케일로
plt.ylim(-1, 1)
plt.yticks(np.arange(-0.1, 1.2, 0.2), [f"{x:.0%}" for x in np.arange(-0.1, 1.2, 0.2)])
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('cifar100_sparsity_vs_rfa_all_methods_local_global.png')
plt.show()