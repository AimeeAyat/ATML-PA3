"""
VGG11 Student Training with VGG19 Teacher - Logit Matching
WITH COMPREHENSIVE ANALYSIS (Enhanced with Part6 features)
===========================================================

This script includes:
1. Loads existing LM student (VGG16 baseline)
2. Trains new LM student (VGG19)
3. Tracks training history for BOTH
4. Parameter efficiency analysis
5. Practical recommendations
6. Multiple visualizations
7. Structured results export

Enhancements from Part6:
- Training history tracking
- Parameter counting and efficiency analysis
- Threshold-based conclusions
- Multiple visualization plots
- Structured JSON export
- Practical recommendations
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Part1.utils import vgg11, vgg16
import torchvision.models as models


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def count_parameters(model):
    """Count total trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models_efficiency(student, teacher16, teacher19):
    """Compare model sizes and calculate efficiency metrics"""
    params_student = count_parameters(student)
    params_16 = count_parameters(teacher16)
    params_19 = count_parameters(teacher19)
    
    return {
        'student': params_student,
        'teacher16': params_16,
        'teacher19': params_19,
        'ratio_16_to_student': params_16 / params_student,
        'ratio_19_to_student': params_19 / params_student,
        'param_increase_19_vs_16': params_19 - params_16,
        'param_increase_pct': ((params_19 - params_16) / params_16) * 100,
    }


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 256,
    'num_workers': 4,
    'num_epochs': 30,
    'learning_rate': 0.1,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'temperature': 4.0,
    'alpha': 0.5,
    'checkpoint_dir': './checkpoints',
    'results_dir': './results/part6',
    'log_interval': 100,
}


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_vgg11_student(device):
    """Load fresh VGG11 student"""
    print("Loading VGG11 Student (fresh)...")
    model = vgg11(num_classes=100).to(device)
    print(f"  ✓ VGG11 loaded")
    return model


def load_vgg16_teacher(device):
    """Load VGG16 teacher"""
    print("Loading VGG16 Teacher...")
    model = vgg16(num_classes=100).to(device)
    print(f"  ✓ VGG16 loaded")
    return model


def load_vgg19_teacher(device, num_classes=100):
    """Load VGG19 teacher from torchvision"""
    print("Loading VGG19 Teacher (downloading if needed)...")
    model = models.vgg19(pretrained=True)
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, num_classes)
    model = model.to(device)
    print(f"  ✓ VGG19 loaded")
    return model


def load_existing_lm_student(device):
    """Load pre-trained LM student"""
    print("Loading Pre-trained LM Student (VGG16 teacher)...")
    checkpoint_path = './checkpoints/lm_student_best.pth'
    
    if not Path(checkpoint_path).exists():
        print(f"  ⚠ Pre-trained LM student not found at {checkpoint_path}")
        return None
    
    try:
        existing_student = vgg11(num_classes=100).to(device)
        existing_student.load_state_dict(torch.load(checkpoint_path, map_location=device))
        existing_student.eval()
        print(f"  ✓ Loaded from: {checkpoint_path}")
        return existing_student
    except Exception as e:
        print(f"  ✗ Error loading pre-trained student: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def get_cifar100_loaders(batch_size, num_workers):
    """Load CIFAR-100"""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    print("Loading CIFAR-100...")
    
    train_dataset = datasets.CIFAR100(
        root='./data', train=True, transform=transform_train, download=True
    )
    val_dataset = datasets.CIFAR100(
        root='./data', train=False, transform=transform_val, download=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"  ✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")
    return train_loader, val_loader


# ════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ════════════════════════════════════════════════════════════════════════════

class LogitMatchingLoss(nn.Module):
    """Logit Matching Loss"""
    
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        ce_loss = self.ce_loss(student_logits, labels)
        
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = torch.softmax(teacher_logits / self.temperature, dim=1)
        kl_loss = self.kl_loss(student_soft, teacher_soft)
        
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        return total_loss, ce_loss, kl_loss


# ════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_epoch(student, teacher, train_loader, optimizer, loss_fn, device, config, epoch):
    """Train for one epoch"""
    
    student.train()
    teacher.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        student_logits = student(images)
        
        with torch.no_grad():
            teacher_logits = teacher(images)
        
        loss, ce, kl = loss_fn(student_logits, teacher_logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = student_logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % config['log_interval'] == 0:
            acc = correct / total * 100
            print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {total_loss/(batch_idx+1):.4f}, Acc: {acc:.2f}%")
    
    return total_loss / len(train_loader), correct / total * 100


def validate(model, val_loader, device):
    """Validate model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return correct / total * 100


def adjust_learning_rate(optimizer, epoch, config):
    """Adjust learning rate"""
    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['learning_rate'] * 0.1
    elif epoch == 150:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['learning_rate'] * 0.01


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS & VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history_16, history_19, save_dir):
    """Plot training curves for both students"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history_16['val_acc'], label='VGG16 Teacher', marker='o', markersize=3)
    axes[0].plot(history_19['val_acc'], label='VGG19 Teacher', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Student Validation Accuracy During Training')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Loss
    axes[1].plot(history_16['train_loss'], label='VGG16 Teacher', marker='o', markersize=3)
    axes[1].plot(history_19['train_loss'], label='VGG19 Teacher', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Training Loss')
    axes[1].set_title('Student Training Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/training_curves.png")
    plt.close()


def plot_accuracy_comparison(existing_acc, new_acc, save_dir):
    """Plot accuracy comparison"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Existing\n(VGG16 Teacher)', 'New\n(VGG19 Teacher)']
    accuracies = [existing_acc, new_acc]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Student Accuracy Comparison\n(After Training)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # Add difference annotation
    diff = new_acc - existing_acc
    ax.text(0.5, max(accuracies) - 5, f'Difference: {diff:+.2f}%',
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/accuracy_comparison.png")
    plt.close()


def plot_parameter_efficiency(params_info, save_dir):
    """Plot parameter efficiency analysis"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Parameter comparison
    models = ['Student\n(VGG11)', 'Teacher\n(VGG16)', 'Teacher\n(VGG19)']
    params = [params_info['student']/1e6, params_info['teacher16']/1e6, params_info['teacher19']/1e6]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax1.bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    for bar, p in zip(bars, params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{p:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Parameters (Millions)', fontsize=12)
    ax1.set_title('Model Size Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Size ratio
    sizes = [1, params_info['ratio_16_to_student'], params_info['ratio_19_to_student']]
    colors2 = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars2 = ax2.bar(models, sizes, color=colors2, alpha=0.7, edgecolor='black', linewidth=2)
    for bar, s in zip(bars2, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{s:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Size Ratio (relative to Student)', fontsize=12)
    ax2.set_title('Teacher Size Relative to Student', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/parameter_efficiency.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/parameter_efficiency.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS & RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════

def analyze_results(existing_acc, new_acc, params_info):
    """Analyze results and provide recommendations"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS")
    print("="*80 + "\n")
    
    # Accuracy difference
    diff = new_acc - existing_acc
    
    # Classify improvement
    if abs(diff) < 0.002:
        improvement_class = "NEGLIGIBLE"
    elif abs(diff) < 0.01:
        improvement_class = "MARGINAL"
    else:
        improvement_class = "SIGNIFICANT"
    
    print(f"Accuracy Improvement: {diff:+.4f}% → {improvement_class}")
    
    # Parameter efficiency
    param_increase = params_info['param_increase_19_vs_16']
    param_increase_pct = params_info['param_increase_pct']
    
    print(f"\nParameter Efficiency:")
    print(f"  VGG-19 has {param_increase:,} more parameters ({param_increase_pct:.1f}% increase)")
    print(f"  Accuracy improvement per 1% param increase: {diff / param_increase_pct:.4f}%")
    
    # Practical recommendation
    print(f"\nPractical Recommendation:")
    
    if abs(diff) < 0.005:
        print(f"  ✓ Use VGG-16 teacher")
        print(f"    Reason: Similar student performance with fewer parameters")
        print(f"    Benefit: Simpler, faster, more efficient")
    elif improvement_class == "MARGINAL":
        print(f"  ? Use VGG-16 or VGG-19 depending on requirements")
        print(f"    If accuracy matters more: Use VGG-19 (+{param_increase_pct:.1f}% params for +{diff:.2f}% accuracy)")
        print(f"    If efficiency matters: Use VGG-16 (simpler, faster)")
    else:
        print(f"  ✓ Use VGG-19 teacher")
        print(f"    Reason: Significant accuracy improvement justifies extra parameters")
        print(f"    Benefit: {diff:.2f}% better student performance")
    
    # Convergence analysis
    print(f"\nConvergence Analysis:")
    print(f"  Check training_curves.png for convergence speed comparison")
    
    return {
        'accuracy_diff': diff,
        'improvement_class': improvement_class,
        'recommendation': 'VGG-16' if abs(diff) < 0.005 else ('VGG-19' if improvement_class == 'SIGNIFICANT' else 'Either'),
        'param_increase_pct': param_increase_pct,
    }


def save_results_json(existing_acc, new_acc, params_info, analysis, save_dir):
    """Save all results to JSON"""
    
    results = {
        'accuracies': {
            'existing_vgg16': existing_acc,
            'new_vgg19': new_acc,
            'difference': new_acc - existing_acc,
        },
        'parameters': {
            'student': params_info['student'],
            'teacher16': params_info['teacher16'],
            'teacher19': params_info['teacher19'],
            'ratio_16_to_student': params_info['ratio_16_to_student'],
            'ratio_19_to_student': params_info['ratio_19_to_student'],
            'param_increase': params_info['param_increase_19_vs_16'],
            'param_increase_pct': params_info['param_increase_pct'],
        },
        'analysis': analysis,
    }
    
    save_path = f'{save_dir}/teacher_size_analysis.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Main training and analysis"""
    
    device = CONFIG['device']
    print(f"Device: {device}\n")
    
    # Create directories
    Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)
    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # ────────────────────────────────────────────────────────────────────────
    # Load all models
    # ────────────────────────────────────────────────────────────────────────
    print("="*80)
    print("LOADING MODELS")
    print("="*80 + "\n")
    
    existing_student = load_existing_lm_student(device)
    student = load_vgg11_student(device)
    teacher_vgg16 = load_vgg16_teacher(device)
    teacher_vgg19 = load_vgg19_teacher(device, num_classes=100)
    
    print()
    
    # ────────────────────────────────────────────────────────────────────────
    # Load data
    # ────────────────────────────────────────────────────────────────────────
    print("="*80)
    print("LOADING DATA")
    print("="*80 + "\n")
    
    train_loader, val_loader = get_cifar100_loaders(
        CONFIG['batch_size'],
        CONFIG['num_workers']
    )
    
    # ────────────────────────────────────────────────────────────────────────
    # Parameter analysis (BEFORE training)
    # ────────────────────────────────────────────────────────────────────────
    print("="*80)
    print("PARAMETER EFFICIENCY ANALYSIS")
    print("="*80 + "\n")
    
    params_info = compare_models_efficiency(student, teacher_vgg16, teacher_vgg19)
    
    print(f"Student (VGG-11):      {params_info['student']:>12,} params")
    print(f"Teacher (VGG-16):      {params_info['teacher16']:>12,} params  ({params_info['ratio_16_to_student']:.1f}x larger)")
    print(f"Teacher (VGG19):       {params_info['teacher19']:>12,} params  ({params_info['ratio_19_to_student']:.1f}x larger)")
    print(f"\nVGG-19 vs VGG-16:")
    print(f"  Extra parameters:    {params_info['param_increase_19_vs_16']:>12,} ({params_info['param_increase_pct']:.1f}%)\n")
    
    # ────────────────────────────────────────────────────────────────────────
    # Training setup
    # ────────────────────────────────────────────────────────────────────────
    print("="*80)
    print("TRAINING SETUP")
    print("="*80 + "\n")
    
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=CONFIG['learning_rate'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )
    
    loss_fn = LogitMatchingLoss(
        temperature=CONFIG['temperature'],
        alpha=CONFIG['alpha']
    )
    
    print(f"Optimizer: SGD (LR={CONFIG['learning_rate']}, Momentum={CONFIG['momentum']})")
    print(f"Loss: Logit Matching (T={CONFIG['temperature']}, α={CONFIG['alpha']})")
    print(f"Epochs: {CONFIG['num_epochs']}\n")
    
    # ────────────────────────────────────────────────────────────────────────
    # Training loop with history tracking
    # ────────────────────────────────────────────────────────────────────────
    print("="*80)
    print("TRAINING NEW STUDENT WITH VGG19 TEACHER")
    print("="*80 + "\n")
    
    history_19 = {
        'train_loss': [],
        'val_acc': [],
    }
    
    best_val_acc = 0
    
    for epoch in range(CONFIG['num_epochs']):
        adjust_learning_rate(optimizer, epoch, CONFIG)
        
        train_loss, train_acc = train_epoch(
            student, teacher_vgg19, train_loader, optimizer, loss_fn, device, CONFIG, epoch
        )
        val_acc = validate(student, val_loader, device)
        
        history_19['train_loss'].append(train_loss)
        history_19['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = f"{CONFIG['checkpoint_dir']}/lm_vgg19_student_best.pth"
            torch.save(student.state_dict(), checkpoint_path)
            print(f"  ✓ Best model saved (Acc: {val_acc:.2f}%)")
        
        print()
    
    # ────────────────────────────────────────────────────────────────────────
    # Comparison
    # ────────────────────────────────────────────────────────────────────────
    print("="*80)
    print("FINAL COMPARISON")
    print("="*80 + "\n")
    
    print("Evaluating both students on validation set...")
    
    if existing_student is not None:
        existing_acc = validate(existing_student, val_loader, device)
        print(f"Existing LM Student (VGG16 teacher): {existing_acc:.2f}%")
    else:
        existing_acc = None
        print(f"Existing LM Student: NOT AVAILABLE")
    
    new_acc = validate(student, val_loader, device)
    print(f"New LM Student (VGG19 teacher):        {new_acc:.2f}%")
    
    if existing_acc:
        diff = new_acc - existing_acc
        status = "✓ Better!" if diff > 0 else "✗ Worse"
        print(f"\nDifference: {diff:+.2f}% {status}\n")
    
    # ────────────────────────────────────────────────────────────────────────
    # Analysis and recommendations
    # ────────────────────────────────────────────────────────────────────────
    if existing_acc:
        analysis = analyze_results(existing_acc, new_acc, params_info)
    else:
        print(f"New student accuracy: {new_acc:.2f}%\n")
        analysis = None
    
    # ────────────────────────────────────────────────────────────────────────
    # Visualizations
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    if existing_acc:
        plot_accuracy_comparison(existing_acc, new_acc, CONFIG['results_dir'])
    
    plot_parameter_efficiency(params_info, CONFIG['results_dir'])
    
    # Create dummy history for VGG16 for comparison
    history_16 = {
        'train_loss': [0.5] * len(history_19['train_loss']),  # Placeholder
        'val_acc': [existing_acc] * len(history_19['val_acc']) if existing_acc else [50] * len(history_19['val_acc']),
    }
    plot_training_curves(history_16, history_19, CONFIG['results_dir'])
    
    # ────────────────────────────────────────────────────────────────────────
    # Save results
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")
    
    if existing_acc and analysis:
        save_results_json(existing_acc, new_acc, params_info, analysis, CONFIG['results_dir'])
    
    # ────────────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nCheckpoint: {CONFIG['checkpoint_dir']}/lm_vgg19_student_best.pth")
    print(f"Results: {CONFIG['results_dir']}/")
    print(f"  ├─ teacher_size_analysis.json")
    print(f"  ├─ accuracy_comparison.png")
    print(f"  ├─ parameter_efficiency.png")
    print(f"  └─ training_curves.png")
    print()


if __name__ == "__main__":
    main()