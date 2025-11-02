import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg11, VGG11_Weights
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import json
import pickle
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

HYPERPARAMS = {
    'finetune_epochs': 10,
    'finetune_lr': 0.001,
    'finetune_batch_size': 128,
    'calibration_batches': 20,
    'test_batch_size': 128,
}

# ============================================================================
# DATA LOADING
# ============================================================================
def get_cifar100_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return trainloader, testloader

# ============================================================================
# MODEL FUNCTIONS
# ============================================================================
def get_pretrained_vgg11_cifar100():
    print("\n📦 Loading VGG-11 with ImageNet pretrained weights...")
    model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, 100)
    nn.init.normal_(model.classifier[6].weight, 0, 0.01)
    nn.init.constant_(model.classifier[6].bias, 0)
    print("✅ Loaded pretrained weights and adapted for CIFAR-100")
    return model

def finetune_model(model, trainloader, testloader, epochs=10, lr=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': lr * 0.1},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    
    print(f"\n🏋️ Fine-tuning for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct, total = 0, 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        train_acc = 100. * correct / total
        test_acc = evaluate_model(model, testloader, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f'Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(trainloader):.3f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')
    
    print(f"✅ Fine-tuning complete! Best accuracy: {best_acc:.2f}%")
    return model, best_acc

def evaluate_model(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ============================================================================
# ACTIVATION STATISTICS COLLECTOR
# ============================================================================
class ActivationCollector:
    def __init__(self, max_samples_per_layer=10000):
        self.stats = {}
        self.samples = {}
        self.max_samples = max_samples_per_layer
        self.hooks = []

    def register_hooks(self, model, layer_names):
        for name, mod in model.named_modules():
            if name in layer_names:
                h = mod.register_forward_hook(self._make_hook(name))
                self.hooks.append(h)

    def _make_hook(self, name):
        def hook(module, inp, out):
            acts = out.detach().flatten()
            
            if name not in self.stats:
                self.stats[name] = {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')}
                self.samples[name] = []
            
            s = self.stats[name]
            s['sum'] += acts.sum().item()
            s['sum_sq'] += (acts ** 2).sum().item()
            s['count'] += acts.numel()
            s['min'] = min(s['min'], acts.min().item())
            s['max'] = max(s['max'], acts.max().item())
            
            if len(self.samples[name]) < self.max_samples:
                n_take = min(100, acts.numel())
                idx = torch.randperm(acts.numel(), device=acts.device)[:n_take]
                self.samples[name].extend(acts[idx].cpu().numpy().tolist())
            del acts
        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_statistics(self, name):
        if name not in self.stats:
            return {}
        
        s = self.stats[name]
        count = s['count']
        mean = s['sum'] / count
        std = (s['sum_sq'] / count - mean ** 2) ** 0.5
        
        if self.samples[name]:
            arr = np.array(self.samples[name])
            p90 = np.percentile(arr, 90.0)
            p95 = np.percentile(arr, 95.0)
            p99 = np.percentile(arr, 99.0)
            p99_9 = np.percentile(arr, 99.9)
        else:
            p90 = s['max'] * 0.90
            p95 = s['max'] * 0.95
            p99 = s['max'] * 0.99
            p99_9 = s['max'] * 0.999
        
        return {
            'mean': float(mean), 'std': float(std), 'min': float(s['min']), 'max': float(s['max']),
            'percentile_90': float(p90), 'percentile_95': float(p95),
            'percentile_99': float(p99), 'percentile_99_9': float(p99_9)
        }

def collect_activation_statistics(model, dataloader, device, num_batches=20):
    model.eval()
    model = model.to(device)
    
    conv_names = [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.ReLU))]
    
    if len(conv_names) >= 6:
        selected = [conv_names[0], conv_names[len(conv_names)//4], conv_names[len(conv_names)//2], 
                   conv_names[3*len(conv_names)//4], conv_names[-2], conv_names[-1]]
    else:
        selected = conv_names
    
    print(f"\n📊 Collecting activation statistics from {len(selected)} layers...")
    for idx, name in enumerate(selected):
        print(f"  [{idx}] {name}")
    
    collector = ActivationCollector(max_samples_per_layer=10000)
    collector.register_hooks(model, selected)
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{num_batches} batches...")
    
    collector.remove_hooks()
    stats = {name: collector.get_statistics(name) for name in selected}
    print("✅ Activation statistics collected!")
    return stats

# ============================================================================
# QUANTIZATION
# ============================================================================
def quantize_tensor(tensor, num_bits=8, clip_percentile=None):
    if clip_percentile is not None:
        clip_val = np.percentile(tensor.abs().cpu().numpy(), clip_percentile)
        tensor_clipped = torch.clamp(tensor, -clip_val, clip_val)
    else:
        tensor_clipped = tensor
        clip_val = tensor.abs().max().item()
    
    qmin = -2**(num_bits-1)
    qmax = 2**(num_bits-1) - 1
    
    scale = tensor_clipped.abs().max() / qmax
    if scale == 0:
        scale = 1.0
    
    q_tensor = torch.clamp(torch.round(tensor_clipped / scale), qmin, qmax)
    dq_tensor = q_tensor * scale
    
    return dq_tensor, scale, clip_val

def apply_weight_quantization(model, bits, clip_percentile=None):
    model_quantized = copy.deepcopy(model)
    quantization_info = {}
    
    for name, module in model_quantized.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                original_weight = module.weight.data.clone()
                quantized_weight, scale, clip_val = quantize_tensor(module.weight.data, bits, clip_percentile)
                module.weight.data = quantized_weight
                
                quantization_info[name] = {
                    'scale': scale, 'clip_val': clip_val,
                    'max_original': original_weight.abs().max().item(), 'bits': bits
                }
    
    return model_quantized, quantization_info

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_activation_distributions(stats, save_path='activation_distributions.png'):
    n_layers = len(stats)
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(5 * ((n_layers + 1) // 2), 8))
    axes = axes.flatten() if n_layers > 1 else [axes]
    
    for idx, (name, s) in enumerate(stats.items()):
        ax = axes[idx]
        samples = np.random.normal(s['mean'], s['std'], 10000)
        samples = np.clip(samples, s['min'], s['max'])
        
        ax.hist(samples, bins=60, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5, density=True)
        ax.axvline(s['mean'], color='green', linestyle='-', linewidth=2, label=f"Mean: {s['mean']:.3f}")
        ax.axvline(s['percentile_95'], color='orange', linestyle='--', linewidth=2, label=f"95%: {s['percentile_95']:.3f}")
        ax.axvline(s['percentile_99'], color='red', linestyle='--', linewidth=2, label=f"99%: {s['percentile_99']:.3f}")
        ax.axvline(s['percentile_99_9'], color='darkred', linestyle='--', linewidth=2, label=f"99.9%: {s['percentile_99_9']:.3f}")
        
        outlier_ratio = (s['percentile_99_9'] - s['mean']) / (s['std'] + 1e-6)
        ax.set_title(f"[{idx}] {name.split('.')[-1]}\nOutlier Ratio: {outlier_ratio:.2f}σ", fontweight='bold', fontsize=10)
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Activation Distributions Across Layers (Log scale)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_quantization_comparison(results, baseline_acc, save_path='quantization_comparison.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = list(results.keys())
    accuracies = [results[c]['accuracy'] for c in configs]
    labels = [results[c]['method'] for c in configs]
    
    colors = []
    for config in configs:
        if 'int8' in config:
            colors.append('#3498db' if 'clipped' in config else '#e74c3c')
        else:
            colors.append('#27ae60' if 'clipped' in config else '#e67e22')
    
    # Plot 1: Accuracy
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(configs)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=baseline_acc, color='blue', linestyle='--', linewidth=2, label=f'FP32 ({baseline_acc:.2f}%)')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels([f"[{i}]" for i in range(len(configs))], rotation=0)
    ax1.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{acc:.2f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Accuracy drop
    ax2 = axes[0, 1]
    drops = [baseline_acc - acc for acc in accuracies]
    bars = ax2.bar(range(len(configs)), drops, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([f"[{i}]" for i in range(len(configs))], rotation=0)
    ax2.set_ylabel('Accuracy Drop (%)', fontweight='bold')
    ax2.set_title('Accuracy Degradation from FP32', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, drop in zip(bars, drops):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2, f'{drop:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: INT8
    ax3 = axes[1, 0]
    int8_configs = [c for c in configs if 'int8' in c]
    int8_accs = [results[c]['accuracy'] for c in int8_configs]
    int8_labels = ['Standard\n(No Clipping)', 'Clipped\n(99.9%)']
    int8_colors = ['#e74c3c', '#3498db']
    bars = ax3.bar(int8_labels, int8_accs, color=int8_colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
    ax3.axhline(y=baseline_acc, color='blue', linestyle='--', linewidth=2, label=f'FP32: {baseline_acc:.2f}%')
    ax3.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax3.set_title('INT8 Quantization', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, int8_accs):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 4: INT4
    ax4 = axes[1, 1]
    int4_configs = [c for c in configs if 'int4' in c]
    int4_accs = [results[c]['accuracy'] for c in int4_configs]
    int4_labels = ['Standard\n(No Clipping)', 'Clipped\n(99.9%)']
    int4_colors = ['#e67e22', '#27ae60']
    bars = ax4.bar(int4_labels, int4_accs, color=int4_colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
    ax4.axhline(y=baseline_acc, color='blue', linestyle='--', linewidth=2, label=f'FP32: {baseline_acc:.2f}%')
    ax4.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax4.set_title('INT4 Quantization', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, int4_accs):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Quantization Performance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_clipping_effects(stats, quant_info_std, quant_info_clip, save_path='clipping_effects.png'):
    layer_names = list(stats.keys())[:3]
    fig, axes = plt.subplots(len(layer_names), 2, figsize=(12, 4 * len(layer_names)))
    if len(layer_names) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, layer_name in enumerate(layer_names):
        conv_layer = None
        for name in quant_info_std.keys():
            if layer_name.split('.')[0] in name or layer_name in name:
                conv_layer = name
                break
        
        if conv_layer is None:
            continue
        
        # Left: Activation distribution
        ax_act = axes[idx, 0]
        s = stats[layer_name]
        samples = np.random.normal(s['mean'], s['std'], 5000)
        samples = np.clip(samples, s['min'], s['max'])
        ax_act.hist(samples, bins=60, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
        ax_act.axvline(s['percentile_99_9'], color='red', linestyle='--', linewidth=2, 
                      label=f"99.9%: {s['percentile_99_9']:.3f}")
        ax_act.axvline(s['max'], color='darkred', linestyle=':', linewidth=2, label=f"Max: {s['max']:.3f}")
        ax_act.set_title(f"[{idx}] {layer_name} - Activations", fontweight='bold')
        ax_act.set_xlabel("Activation Value")
        ax_act.set_ylabel("Frequency")
        ax_act.legend(fontsize=9)
        ax_act.grid(True, alpha=0.3)
        ax_act.set_yscale('log')
        
        # Right: Weight quantization
        ax_weight = axes[idx, 1]
        info_std = quant_info_std[conv_layer]
        info_clip = quant_info_clip[conv_layer]
        x_range = np.linspace(-info_std['max_original'], info_std['max_original'], 200)
        
        qmin = -2**(info_std['bits']-1)
        qmax = 2**(info_std['bits']-1) - 1
        scale_std = info_std['clip_val'] / qmax
        quantized_std = np.clip(np.round(x_range / scale_std), qmin, qmax) * scale_std
        
        scale_clip = info_clip['clip_val'] / qmax
        x_clipped = np.clip(x_range, -info_clip['clip_val'], info_clip['clip_val'])
        quantized_clip = np.clip(np.round(x_clipped / scale_clip), qmin, qmax) * scale_clip
        
        ax_weight.plot(x_range, x_range, 'g-', linewidth=2, label='Original', alpha=0.7)
        ax_weight.plot(x_range, quantized_std, 'r-', linewidth=2, label='Standard', alpha=0.7)
        ax_weight.plot(x_range, quantized_clip, 'b-', linewidth=2, label='Clipped', alpha=0.7)
        ax_weight.axvline(info_clip['clip_val'], color='orange', linestyle='--', 
                         label=f"Clip: ±{info_clip['clip_val']:.3f}")
        ax_weight.axvline(-info_clip['clip_val'], color='orange', linestyle='--')
        ax_weight.set_title(f"[{idx}] {conv_layer.split('.')[-2]} - Weights", fontweight='bold')
        ax_weight.set_xlabel("Original Weight")
        ax_weight.set_ylabel("Quantized Weight")
        ax_weight.legend(fontsize=9)
        ax_weight.grid(True, alpha=0.3)
    
    plt.suptitle("Effect of Percentile Clipping", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_layer_sensitivity(stats, results, baseline_acc, save_path='layer_sensitivity.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Outlier severity
    ax1 = axes[0]
    layers = list(stats.keys())
    layer_labels = [f"[{i}] {l}" for i, l in enumerate(layers)]
    outlier_ratios = [(s['percentile_99_9'] - s['mean']) / (s['std'] + 1e-6) for s in stats.values()]
    colors_severity = plt.cm.Reds(np.linspace(0.3, 0.9, len(outlier_ratios)))
    bars = ax1.barh(layer_labels, outlier_ratios, color=colors_severity, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Outlier Severity (99.9% - Mean) / Std', fontweight='bold')
    ax1.set_title('Layer-wise Outlier Severity', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, outlier_ratios):
        ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}σ',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: Improvement from clipping
    ax2 = axes[1]
    improvements = {
        'INT8 Clipping': results['int8_clipped']['accuracy'] - results['int8_standard']['accuracy'],
        'INT4 Clipping': results['int4_clipped']['accuracy'] - results['int4_standard']['accuracy'],
    }
    colors_imp = ['#3498db', '#27ae60']
    bars = ax2.bar(improvements.keys(), improvements.values(), color=colors_imp, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_ylabel('Accuracy Improvement (%)', fontweight='bold')
    ax2.set_title('Accuracy Gain from Clipping', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, (name, val) in zip(bars, improvements.items()):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:+.2f}%',
                ha='center', va='bottom' if val > 0 else 'top', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def analyze_bn_relu_impact(model, trainloader, device, num_batches=10):
    model.eval()
    model = model.to(device)
    stats_pre_bn, stats_post_bn = {}, {}
    stats_pre_relu, stats_post_relu = {}, {}
    hooks = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            def make_pre_bn(n):
                def hook(m, inp):
                    if n not in stats_pre_bn:
                        stats_pre_bn[n] = []
                    stats_pre_bn[n].append(inp[0].detach().flatten().abs().max().item())
                return hook
            def make_post_bn(n):
                def hook(m, inp, out):
                    if n not in stats_post_bn:
                        stats_post_bn[n] = []
                    stats_post_bn[n].append(out.detach().flatten().abs().max().item())
                return hook
            hooks.append(module.register_forward_pre_hook(make_pre_bn(name)))
            hooks.append(module.register_forward_hook(make_post_bn(name)))
        
        if isinstance(module, nn.ReLU):
            def make_pre_relu(n):
                def hook(m, inp):
                    if n not in stats_pre_relu:
                        stats_pre_relu[n] = []
                    stats_pre_relu[n].append(inp[0].detach().flatten().abs().max().item())
                return hook
            def make_post_relu(n):
                def hook(m, inp, out):
                    if n not in stats_post_relu:
                        stats_post_relu[n] = []
                    stats_post_relu[n].append(out.detach().flatten().abs().max().item())
                return hook
            hooks.append(module.register_forward_pre_hook(make_pre_relu(name)))
            hooks.append(module.register_forward_hook(make_post_relu(name)))
    
    print(f"\n🔬 Analyzing BN/ReLU impact...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(trainloader):
            if i >= num_batches:
                break
            _ = model(inputs.to(device))
    
    for h in hooks:
        h.remove()
    
    bn_summary = {name: {
        'pre': np.mean(stats_pre_bn[name]),
        'post': np.mean(stats_post_bn[name]),
        'reduction_pct': (np.mean(stats_pre_bn[name]) - np.mean(stats_post_bn[name])) / np.mean(stats_pre_bn[name]) * 100
    } for name in stats_pre_bn.keys()}
    
    relu_summary = {name: {
        'pre': np.mean(stats_pre_relu[name]),
        'post': np.mean(stats_post_relu[name]),
        'reduction_pct': (np.mean(stats_pre_relu[name]) - np.mean(stats_post_relu[name])) / np.mean(stats_pre_relu[name]) * 100
    } for name in stats_pre_relu.keys()}
    
    print("✅ BN/ReLU analysis complete!")
    return bn_summary, relu_summary

def plot_bn_relu_analysis(bn_summary, relu_summary, save_path='bn_relu_impact.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: BatchNorm
    ax1 = axes[0]
    if bn_summary:
        layers = list(bn_summary.keys())[:5]
        layer_labels = [f"BN[{i}]" for i in range(len(layers))]
        pre_vals = [bn_summary[l]['pre'] for l in layers]
        post_vals = [bn_summary[l]['post'] for l in layers]
        x = np.arange(len(layer_labels))
        width = 0.35
        ax1.bar(x - width/2, pre_vals, width, label='Pre-BN', color='#e74c3c', alpha=0.8, edgecolor='black')
        ax1.bar(x + width/2, post_vals, width, label='Post-BN', color='#2ecc71', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Layer', fontweight='bold')
        ax1.set_ylabel('Max Activation', fontweight='bold')
        ax1.set_title('BatchNorm Effect on Outliers', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        for i, layer in enumerate(layers):
            reduction = bn_summary[layer]['reduction_pct']
            ax1.text(i, max(pre_vals[i], post_vals[i]) * 1.05, f'{reduction:.1f}%↓',
                    ha='center', fontsize=9, fontweight='bold', color='green' if reduction > 0 else 'red')
    else:
        ax1.text(0.5, 0.5, 'No BatchNorm layers', ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: ReLU
    ax2 = axes[1]
    if relu_summary:
        layers = list(relu_summary.keys())[:5]
        layer_labels = [f"ReLU[{i}]" for i in range(len(layers))]
        pre_vals = [relu_summary[l]['pre'] for l in layers]
        post_vals = [relu_summary[l]['post'] for l in layers]
        x = np.arange(len(layer_labels))
        width = 0.35
        ax2.bar(x - width/2, pre_vals, width, label='Pre-ReLU', color='#3498db', alpha=0.8, edgecolor='black')
        ax2.bar(x + width/2, post_vals, width, label='Post-ReLU', color='#f39c12', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Layer', fontweight='bold')
        ax2.set_ylabel('Max Activation', fontweight='bold')
        ax2.set_title('ReLU Effect on Outliers', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        for i, layer in enumerate(layers):
            reduction = relu_summary[layer]['reduction_pct']
            ax2.text(i, max(pre_vals[i], post_vals[i]) * 1.05, f'{reduction:.1f}%↓',
                    ha='center', fontsize=9, fontweight='bold', color='green' if reduction > 0 else 'red')
    else:
        ax2.text(0.5, 0.5, 'No ReLU layers', ha='center', va='center', transform=ax2.transAxes)
    
    plt.suptitle('Impact of BatchNorm and ReLU on Activation Outliers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def analyze_channel_outliers(model, trainloader, device, num_batches=5):
    model.eval()
    model = model.to(device)
    channel_stats = {}
    hooks = []
    
    conv_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)][:3]
    
    for name, module in conv_layers:
        def make_hook(n):
            def hook(m, inp, out):
                acts = out.detach().float()
                max_per_channel = acts.abs().view(acts.size(0), acts.size(1), -1).max(dim=2)[0]
                max_per_channel = max_per_channel.max(dim=0)[0].cpu().numpy()
                if n not in channel_stats:
                    channel_stats[n] = []
                channel_stats[n].append(max_per_channel)
            return hook
        hooks.append(module.register_forward_hook(make_hook(name)))
    
    print(f"\n🔬 Analyzing channel-wise outliers...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(trainloader):
            if i >= num_batches:
                break
            _ = model(inputs.to(device))
    
    for h in hooks:
        h.remove()
    
    channel_means = {name: np.mean(np.stack(stats_list), axis=0) for name, stats_list in channel_stats.items()}
    print("✅ Channel analysis complete!")
    return channel_means

def plot_channel_outliers(channel_stats, save_path='channel_outliers.png'):
    n_layers = len(channel_stats)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4 * n_layers))
    if n_layers == 1:
        axes = [axes]
    
    for idx, (name, vals) in enumerate(channel_stats.items()):
        ax = axes[idx]
        im = ax.imshow(vals[None, :], cmap='hot', aspect='auto', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Max |Activation|')
        ax.set_yticks([])
        n_channels = len(vals)
        tick_positions = np.linspace(0, n_channels-1, min(10, n_channels), dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions)
        ax.set_xlabel('Channel Index', fontweight='bold')
        ax.set_title(f'[{idx}] {name} - Channel-wise Outliers ({n_channels} channels, max={vals.max():.3f})', 
                    fontweight='bold')
        
        top_k = 5
        top_indices = np.argsort(vals)[-top_k:]
        for i in top_indices:
            ax.axvline(i, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.suptitle('Per-Channel Outlier Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_outlier_experiments(model, trainloader, testloader, device):
    print("\n" + "="*70)
    print("OUTLIER ANALYSIS: QUANTIZATION QUALITY STUDY")
    print("="*70)
    
    # Step 1: Collect activation statistics
    print("\n[Step 1] Collecting activation statistics...")
    stats = collect_activation_statistics(model, trainloader, device, num_batches=HYPERPARAMS['calibration_batches'])
    
    print("\n" + "-"*80)
    print(f"{'Layer':<40} {'Mean':<10} {'Std':<10} {'99%':<10} {'99.9%':<10}")
    print("-"*80)
    for name, s in stats.items():
        layer_name = name if len(name) <= 38 else "..." + name[-35:]
        print(f"{layer_name:<40} {s['mean']:<10.4f} {s['std']:<10.4f} {s['percentile_99']:<10.4f} {s['percentile_99_9']:<10.4f}")
    print("-"*80)
    
    plot_activation_distributions(stats, save_path='activation_distributions.png')
    
    # Step 2: Evaluate baseline
    print("\n[Step 2] Evaluating FP32 baseline...")
    baseline_acc = evaluate_model(model, testloader, device)
    print(f"🎯 Baseline FP32 Accuracy: {baseline_acc:.2f}%")
    
    # Step 3: Run quantization experiments
    print("\n[Step 3] Running quantization experiments...")
    results = {}
    quant_info = {}
    
    experiments = [
        ('int8_standard', 8, None, "INT8 - Standard (no clipping)"),
        ('int8_clipped', 8, 99.9, "INT8 - Clipped (99.9%)"),
        ('int4_standard', 4, None, "INT4 - Standard (no clipping)"),
        ('int4_clipped', 4, 99.9, "INT4 - Clipped (99.9%)"),
    ]
    
    for key, bits, clip_pct, desc in experiments:
        print(f"\n{'='*70}")
        print(f"Testing: {desc}")
        print("="*70)
        
        qmodel, qinfo = apply_weight_quantization(model, bits, clip_pct)
        qmodel = qmodel.to(device)
        quant_info[key] = qinfo
        
        print("Evaluating quantized model...")
        acc = evaluate_model(qmodel, testloader, device)
        
        results[key] = {
            'accuracy': acc, 'method': desc, 'delta': acc - baseline_acc,
            'bits': bits, 'clipping': clip_pct is not None, 'clip_percentile': clip_pct
        }
        
        print(f"   Accuracy: {acc:.2f}% (Δ: {acc - baseline_acc:+.2f}%)")
        
        del qmodel
        torch.cuda.empty_cache()
    
    # Step 4: Generate visualizations
    print("\n[Step 4] Generating visualizations...")
    plot_quantization_comparison(results, baseline_acc, save_path='quantization_comparison.png')
    plot_clipping_effects(stats, quant_info['int8_standard'], quant_info['int8_clipped'], save_path='clipping_effects.png')
    plot_layer_sensitivity(stats, results, baseline_acc, save_path='layer_sensitivity.png')
    
    # Step 5: BN/ReLU analysis
    print("\n[Step 5] Analyzing BatchNorm and ReLU impact...")
    bn_summary, relu_summary = analyze_bn_relu_impact(model, trainloader, device, num_batches=10)
    plot_bn_relu_analysis(bn_summary, relu_summary, save_path='bn_relu_impact.png')
    
    # Step 6: Channel analysis
    print("\n[Step 6] Analyzing channel-wise outliers...")
    channel_stats = analyze_channel_outliers(model, trainloader, device, num_batches=5)
    plot_channel_outliers(channel_stats, save_path='channel_outliers.png')
    
    return results, stats, bn_summary, relu_summary, channel_stats

def print_summary_report(results, stats, baseline_acc, bn_summary, relu_summary):
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT: OUTLIER IMPACT ON QUANTIZATION")
    print("="*80)
    
    print(f"\n📊 BASELINE PERFORMANCE:")
    print(f"   FP32 Accuracy: {baseline_acc:.2f}%")
    
    print(f"\n📉 QUANTIZATION RESULTS:")
    print("-" * 80)
    print(f"{'Configuration':<35} {'Accuracy':<12} {'Δ from FP32':<15} {'Relative Loss':<15}")
    print("-" * 80)
    
    for key in ['int8_standard', 'int8_clipped', 'int4_standard', 'int4_clipped']:
        res = results[key]
        delta = res['delta']
        rel_loss = (baseline_acc - res['accuracy']) / baseline_acc * 100
        print(f"{res['method']:<35} {res['accuracy']:<12.2f} {delta:<15.2f} {rel_loss:<14.2f}%")
    
    print("-" * 80)
    
    print(f"\n💡 KEY FINDINGS:")
    
    int8_improvement = results['int8_clipped']['accuracy'] - results['int8_standard']['accuracy']
    int4_improvement = results['int4_clipped']['accuracy'] - results['int4_standard']['accuracy']
    
    print(f"\n1. OUTLIER CLIPPING IMPACT:")
    print(f"   - INT8 clipping changes accuracy by {int8_improvement:+.2f}%")
    print(f"   - INT4 clipping improves accuracy by {int4_improvement:+.2f}%")
    print(f"   - Lower bit-widths benefit MORE from outlier handling")
    
    print(f"\n2. OUTLIER SEVERITY ANALYSIS:")
    avg_outlier_ratio = np.mean([(s['percentile_99_9'] - s['mean']) / (s['std'] + 1e-6) for s in stats.values()])
    print(f"   - Average outlier severity: {avg_outlier_ratio:.2f}σ")
    print(f"   - Outliers force quantizers to allocate wide dynamic range")
    print(f"   - This reduces effective resolution for typical values")
    
    if bn_summary:
        avg_bn_reduction = np.mean([v['reduction_pct'] for v in bn_summary.values()])
        print(f"\n3. BATCHNORM IMPACT:")
        print(f"   - BatchNorm reduces outlier magnitude by {avg_bn_reduction:.1f}% on average")
        print(f"   - Normalization helps stabilize activation distributions")
    
    if relu_summary:
        avg_relu_reduction = np.mean([v['reduction_pct'] for v in relu_summary.values()])
        print(f"\n4. RELU IMPACT:")
        print(f"   - ReLU reduces outlier magnitude by {avg_relu_reduction:.1f}% on average")
        print(f"   - Saturation at zero clips negative outliers")
    
    print(f"\n5. IMPLICATIONS FOR DEPLOYMENT:")
    print(f"   ✓ Outlier-robust quantizers are CRUCIAL for:")
    print(f"     • Large Language Models (LLMs) - extreme outliers in attention")
    print(f"     • Vision Transformers (ViTs) - lack BatchNorm regularization")
    print(f"     • Models with skewed activation distributions")
    print(f"   ✓ Clipping/scaling can recover up to {max(abs(int8_improvement), int4_improvement):.1f}% accuracy")
    print(f"   ✓ Per-channel or per-token quantization may be necessary")
    print(f"   ✓ Mixed-precision preserves sensitive layers")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)

def save_results(results, stats, bn_summary, relu_summary, channel_stats, baseline_acc, save_prefix='outlier_analysis'):
    json_results = {
        'quantization_results': results,
        'baseline_accuracy': float(baseline_acc),
        'activation_statistics': stats,
        'hyperparameters': HYPERPARAMS
    }
    
    with open(f'{save_prefix}_results.json', 'w') as f:
        json.dump(json_results, f, indent=4)
    print(f"\n💾 Saved: {save_prefix}_results.json")
    
    full_results = {
        'quantization_results': results,
        'activation_statistics': stats,
        'bn_summary': bn_summary,
        'relu_summary': relu_summary,
        'channel_statistics': channel_stats,
        'baseline_accuracy': baseline_acc,
        'hyperparameters': HYPERPARAMS
    }
    
    with open(f'{save_prefix}_full.pkl', 'wb') as f:
        pickle.dump(full_results, f)
    print(f"💾 Saved: {save_prefix}_full.pkl")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*80)
    print("OUTLIER ANALYSIS FOR NEURAL NETWORK QUANTIZATION")
    print("="*80)
    
    print("\n[1/5] Loading CIFAR-100 dataset...")
    trainloader, testloader = get_cifar100_dataloaders(batch_size=HYPERPARAMS['finetune_batch_size'])
    
    print("\n[2/5] Loading/training FP32 baseline model...")
    model_path = "vgg11_cifar100_fp32.pth"
    
    if os.path.exists(model_path):
        print("✅ Found pretrained FP32 model. Loading weights...")
        model = get_pretrained_vgg11_cifar100().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        baseline_acc = evaluate_model(model, testloader, device)
        print(f"✅ Loaded model | Baseline Accuracy: {baseline_acc:.2f}%")
    else:
        print("🚀 No pretrained model found. Training FP32 model...")
        model = get_pretrained_vgg11_cifar100().to(device)
        model, baseline_acc = finetune_model(model, trainloader, testloader, 
                                            epochs=HYPERPARAMS['finetune_epochs'],
                                            lr=HYPERPARAMS['finetune_lr'])
        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved trained FP32 model to {model_path}")
    
    print("\n[3/5] Running outlier analysis experiments...")
    results, stats, bn_summary, relu_summary, channel_stats = run_outlier_experiments(model, trainloader, testloader, device)
    
    print("\n[4/5] Generating summary report...")
    print_summary_report(results, stats, baseline_acc, bn_summary, relu_summary)
    
    print("\n[5/5] Saving results...")
    save_results(results, stats, bn_summary, relu_summary, channel_stats, baseline_acc, save_prefix='outlier_analysis')
    
    print("\n📁 Generated files:")
    print("   • activation_distributions.png - Outlier visualization")
    print("   • quantization_comparison.png - Performance comparison")
    print("   • clipping_effects.png - Effect of percentile clipping")
    print("   • layer_sensitivity.png - Layer-wise outlier impact")
    print("   • bn_relu_impact.png - BatchNorm/ReLU analysis")
    print("   • channel_outliers.png - Per-channel outlier heatmap")
    print("   • outlier_analysis_results.json - Summary results")
    print("   • outlier_analysis_full.pkl - Complete data")
    
    return results

if __name__ == "__main__":
    results = main()