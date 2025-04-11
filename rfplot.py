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

# forget 및 retain 클래스 정의
forget_classes = [0]
retain_classes = list(range(1, 100))

# 데이터로더 생성
_, _, test_loader = cifar100_dataloaders()

# 메서드별, sparsity별, 시드별 결과 저장
results = {method: {sparsity: {'retain_accs': [], 'forget_accs': []} for sparsity in sparsity_values} for method in methods}
layer_wise = True
# 저장된 모델 불러오기 및 정확도 계산
for method in methods:
    for sparsity in sparsity_values:
        for seed in seeds:
            save_path = f'../results/synaptag/layer_wise={layer_wise}/method={method}/sparsity={sparsity}/seed={seed}/best_model.pt'
            if os.path.exists(save_path):
                set_seed(seed)  # 모델 초기화 시 동일한 시드 사용
                model = SupermaskNet(deepcopy(base_model), sparsity=sparsity, layer_wise=True).cuda()
                model.load_state_dict(torch.load(save_path, weights_only=True))
                retain_acc, forget_acc, rfa = get_unlearning_accuracy(
                    forget_classes=forget_classes,
                    retain_classes=retain_classes,
                    model=model,
                    data_loader=test_loader
                )
                results[method][sparsity]['retain_accs'].append(retain_acc)
                results[method][sparsity]['forget_accs'].append(forget_acc)
                print(f"Method: {method} | Sparsity: {sparsity:.4f} | Seed: {seed} | Retain Acc.: {retain_acc:10.2%} | Forget Acc.: {forget_acc:10.2%} | RFA: {rfa:10.2%}")
            else:
                print(f"Model not found: {save_path}")

# 평균과 표준편차 계산
avg_results = {}
for method in methods:
    avg_results[method] = {
        'avg_retain_accs': [],
        'avg_forget_accs': [],
        'std_retain_accs': []
    }
    for sparsity in sparsity_values:
        retain_accs = results[method][sparsity]['retain_accs']
        forget_accs = results[method][sparsity]['forget_accs']
        if retain_accs:  # 데이터가 있는 경우만 계산
            avg_retain = np.mean(retain_accs)
            avg_forget = np.mean(forget_accs)
            std_retain = np.std(retain_accs)
            avg_results[method]['avg_retain_accs'].append(avg_retain)
            avg_results[method]['avg_forget_accs'].append(avg_forget)
            avg_results[method]['std_retain_accs'].append(std_retain)
            print(f"Method: {method} | Sparsity: {sparsity:.4f} | Avg Retain Acc.: {avg_retain:10.2%} | Avg Forget Acc.: {avg_forget:10.2%} | Std Retain: {std_retain:.4f}")

# Matplotlib으로 그래프 그리기
plt.figure(figsize=(10, 10))  # 정사각형 비율 설정

# 색상 정의
colors = {
    'random_pruning': 'gray',
    'random_labels': 'blue',
    'gradient_ascent': 'green',
    'chance_level': 'red',
}

# 메서드별 플랏
for method in methods:
    avg_retain_accs = avg_results[method]['avg_retain_accs']
    avg_forget_accs = avg_results[method]['avg_forget_accs']
    std_retain_accs = avg_results[method]['std_retain_accs']
    
    # 점 크기 스케일링
    min_size = 5
    max_size = 100
    min_std = min(std_retain_accs) if std_retain_accs else 0
    max_std = max(std_retain_accs) if std_retain_accs else 0
    if max_std > min_std:
        sizes = [min_size + (max_size - min_size) * (std - min_std) / (max_std - min_std) for std in std_retain_accs]
    else:
        sizes = [min_size] * len(std_retain_accs)

    # 선으로 연결 및 점 표시
    plt.plot(avg_retain_accs, avg_forget_accs, color=colors[method], linestyle='--', linewidth=2, label=method)
    plt.scatter(avg_retain_accs, avg_forget_accs, s=sizes, color=colors[method], alpha=0.5)

    # sparsity 값 라벨 추가
    for i, sparsity in enumerate(sparsity_values):
        plt.annotate(f"{sparsity:.4f}", (avg_retain_accs[i], avg_forget_accs[i]), textcoords="offset points", xytext=(0, 10), ha='center')

# 축 설정 (선형 스케일, 0~100%)
plt.xlabel('Retain Accuracy (%)')
plt.ylabel('Forget Accuracy (%)')
plt.title('Avg Retain vs Avg Forget Accuracy for Different Methods (CIFAR-100)')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.arange(0, 1.1, 0.1), [f"{x:.0%}" for x in np.arange(0, 1.1, 0.1)])
plt.yticks(np.arange(0, 1.1, 0.1), [f"{x:.0%}" for x in np.arange(0, 1.1, 0.1)])
plt.grid(True, which="both", ls="--")
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')  # 정사각형 대칭 보장
plt.savefig('cifar100_retain_vs_forget_all_methods.png')  # 그래프를 파일로 저장
plt.show()