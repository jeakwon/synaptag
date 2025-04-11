from models.pretrained import ResNet18CIFAR10, ResNet50CIFAR10, ResNet18CIFAR100, ResNet50CIFAR100
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
layer_wise = False
methods = [
    # 'random_pruning', 
    # 'random_labels', 
    # 'chance_level', 
    'gradient_ascent',
]

for method in methods:

    # forget 및 retain 클래스 정의
    forget_classes = [0]
    retain_classes = list(range(1, 100))

    # 데이터로더 생성
    train_loader, val_loader, test_loader = cifar100_dataloaders()

    # sparsity별로 시드에 따른 결과 저장
    retain_accs_per_sparsity = {sparsity: [] for sparsity in sparsity_values}
    forget_accs_per_sparsity = {sparsity: [] for sparsity in sparsity_values}

    # sparsity와 seed에 따른 정확도 계산
    for sparsity in sparsity_values:
        for seed in seeds:
            set_seed(seed)  # 각 시드 설정
            model = SupermaskNet(deepcopy(base_model), sparsity=sparsity, layer_wise=layer_wise).cuda()
            optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
            unlearner = Unlearner(model, train_loader, val_loader, test_loader, optimizer, method=method, epochs=50, early_stop_patience=5, save_path=f'../results/synaptag/layer_wise={layer_wise}/method={method}/sparsity={sparsity}/seed={seed}/best_model.pt', device='cuda')
            if method!='random_pruning':
                unlearner.run(forget_classes)
            retain_acc, forget_acc, rfa = get_unlearning_accuracy(
                forget_classes=forget_classes,
                retain_classes=retain_classes,
                model=model,
                data_loader=test_loader
            )
            retain_accs_per_sparsity[sparsity].append(retain_acc)
            forget_accs_per_sparsity[sparsity].append(forget_acc)
            print(f"Sparsity: {sparsity:.4f} | Seed: {seed} | Retain Acc.: {retain_acc:10.2%} | Forget Acc.: {forget_acc:10.2%} | RFA: {rfa:10.2%}")

    # 평균과 표준편차 계산
    avg_retain_accs = []
    avg_forget_accs = []
    std_retain_accs = []

    for sparsity in sparsity_values:
        avg_retain = np.mean(retain_accs_per_sparsity[sparsity])
        avg_forget = np.mean(forget_accs_per_sparsity[sparsity])
        std_retain = np.std(retain_accs_per_sparsity[sparsity])
        
        avg_retain_accs.append(avg_retain)
        avg_forget_accs.append(avg_forget)
        std_retain_accs.append(std_retain)
        print(f"Sparsity: {sparsity:.4f} | Avg Retain Acc.: {avg_retain:10.2%} | Avg Forget Acc.: {avg_forget:10.2%} | Std Retain: {std_retain:.4f}")

    # 점 크기 스케일링 (가장 작은 std를 5~10으로 클램핑)
    min_size = 5  # 최소 크기
    max_size = 100  # 최대 크기
    min_std = min(std_retain_accs)
    max_std = max(std_retain_accs)

    if max_std > min_std:  # std가 모두 동일하지 않은 경우
        sizes = [min_size + (max_size - min_size) * (std - min_std) / (max_std - min_std) for std in std_retain_accs]
    else:  # std가 모두 동일한 경우
        sizes = [min_size] * len(std_retain_accs)

    # Matplotlib으로 평균 retain vs forget 그래프 그리기 (로그 스케일)
    plt.figure(figsize=(8, 8))  # 정사각형 비율 설정

    # 선으로 연결 (회색 점선)
    plt.plot(avg_retain_accs, avg_forget_accs, color='gray', linestyle='--', linewidth=2)

    # 점 크기로 오차 범위 표시 (회색)
    plt.scatter(avg_retain_accs, avg_forget_accs, s=sizes, color='gray', alpha=0.5)

    # 각 점에 sparsity 값 라벨 추가
    for i, sparsity in enumerate(sparsity_values):
        plt.annotate(f"{sparsity:.4f}", (avg_retain_accs[i], avg_forget_accs[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # 축 설정 (로그 스케일, 0.1~1.0 범위, 퍼센트로 표시)
    plt.xlabel('Retain Accuracy (%)')
    plt.ylabel('Forget Accuracy (%)')
    plt.title('Avg Retain vs Avg Forget Accuracy with Sparsity (3 Seeds, Log Scale)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.01, 1)  # 10% ~ 100%
    plt.ylim(0.01, 1)  # 10% ~ 100%
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [f"{x:.0%}" for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [f"{x:.0%}" for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    plt.grid(True, which="both", ls="--")
    plt.gca().set_aspect('equal', adjustable='box')  # 정사각형 대칭 보장
    plt.savefig('avg_retain_vs_forget_sparsity_log_scale.png')  # 그래프를 파일로 저장
    plt.show()  # 그래프 표시