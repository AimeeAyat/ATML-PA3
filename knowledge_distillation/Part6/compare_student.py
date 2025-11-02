"""
Simple Accuracy Comparison - Load Both Students and Compare
===========================================================

This script:
1. Loads existing trained LM student (VGG16 teacher)
2. Loads new trained LM student (VGG19 teacher)
3. Evaluates both on validation set
4. Compares accuracies
5. Creates comparison visualization

NO TRAINING - Just comparison!
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

from Part1.utils import vgg11


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 256,
    'num_workers': 4,
    'checkpoint_dir': './checkpoints',
    'results_dir': './results/part6',
}

# Checkpoint paths
CHECKPOINTS = {
    'existing': './checkpoints/lm_student_best.pth',
    'new': './checkpoints/lm_vgg19_student_best.pth',
}


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def get_cifar100_val_loader(batch_size, num_workers):
    """Load CIFAR-100 validation set only"""
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    print("Loading CIFAR-100 validation set...")
    
    val_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        transform=transform_val,
        download=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ Validation samples: {len(val_dataset)}\n")
    return val_loader


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_student(checkpoint_path, device, name):
    """Load a student model from checkpoint"""
    
    print(f"Loading {name}...")
    
    if not Path(checkpoint_path).exists():
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        model = vgg11(num_classes=100).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume the whole dict is the state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"  ✓ Loaded from: {checkpoint_path}")
        return model
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, val_loader, device, model_name):
    """Evaluate model on validation set"""
    
    print(f"\nEvaluating {model_name}...")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Progress
            if (batch_idx + 1) % 20 == 0:
                current_acc = correct / total * 100
                print(f"  [{batch_idx+1}/{len(val_loader)}] Current Acc: {current_acc:.2f}%")
    
    accuracy = correct / total * 100
    print(f"  ✓ Final Accuracy: {accuracy:.2f}%\n")
    
    return accuracy


# ════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

def plot_accuracy_comparison(existing_acc, new_acc, save_dir):
    """Create accuracy comparison plot"""
    
    print(f"Creating accuracy comparison plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Existing\n(VGG16 Teacher)', 'New\n(VGG19 Teacher)']
    accuracies = [existing_acc, new_acc]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Student Accuracy Comparison\n(Trained with Different Teachers)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add difference annotation
    diff = new_acc - existing_acc
    diff_text = f'Difference: {diff:+.2f}%'
    diff_color = 'green' if diff > 0 else 'red'
    
    ax.text(0.5, max(accuracies) - 8, diff_text,
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=diff_color, alpha=0.3))
    
    # Add interpretation
    if diff > 0.01:
        interpretation = "✓ VGG19 teacher is BETTER"
    elif diff < -0.01:
        interpretation = "✗ VGG16 teacher is BETTER"
    else:
        interpretation = "≈ Similar performance"
    
    ax.text(0.5, 10, interpretation,
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))
    
    plt.tight_layout()
    save_path = f'{save_dir}/accuracy_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS & RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════

def analyze_and_recommend(existing_acc, new_acc):
    """Analyze results and provide recommendations"""
    
    print("\n" + "="*80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*80 + "\n")
    
    diff = new_acc - existing_acc
    abs_diff = abs(diff)
    
    print(f"Existing LM Student (VGG16 teacher): {existing_acc:.2f}%")
    print(f"New LM Student (VGG19 teacher):      {new_acc:.2f}%")
    print(f"Difference:                          {diff:+.2f}%\n")
    
    # Classification
    if abs_diff < 0.002:
        classification = "NEGLIGIBLE"
        threshold_info = "< 0.2%"
    elif abs_diff < 0.01:
        classification = "MARGINAL"
        threshold_info = "0.2% - 1.0%"
    else:
        classification = "SIGNIFICANT"
        threshold_info = "> 1.0%"
    
    print(f"Improvement Classification: {classification} ({threshold_info})")
    
    # Recommendation
    print(f"\nRECOMMENDATION:")
    
    if diff > 0:
        print(f"  ✓ VGG19 Teacher is BETTER")
        if classification == "SIGNIFICANT":
            print(f"    → Use VGG19 teacher for better student performance")
            print(f"    → Accuracy improvement: {diff:.2f}%")
        elif classification == "MARGINAL":
            print(f"    → VGG19 is slightly better, but consider:")
            print(f"      - VGG16: Simpler, faster training")
            print(f"      - VGG19: Better accuracy (+{diff:.2f}%)")
        else:
            print(f"    → Negligible difference, either teacher works")
            print(f"    → Prefer VGG16 for efficiency")
    elif diff < 0:
        print(f"  ✗ VGG16 Teacher is BETTER")
        if abs_diff > 0.01:
            print(f"    → VGG19 is worse by {abs_diff:.2f}%")
            print(f"    → Use VGG16 teacher instead")
        else:
            print(f"    → Negligible difference, either teacher works")
    else:
        print(f"  ≈ EQUAL Performance")
        print(f"    → Both teachers produce identical results")
        print(f"    → Use VGG16 for simplicity")
    
    return {
        'existing_accuracy': existing_acc,
        'new_accuracy': new_acc,
        'difference': diff,
        'classification': classification,
        'recommendation': 'VGG19' if diff > 0.01 else 'VGG16' if diff < -0.01 else 'Either',
    }


# ════════════════════════════════════════════════════════════════════════════
# EXPORT RESULTS
# ════════════════════════════════════════════════════════════════════════════

def save_comparison_json(analysis, save_dir):
    """Save comparison results to JSON"""
    
    save_path = f'{save_dir}/accuracy_comparison.json'
    
    with open(save_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✓ Saved comparison: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Main comparison function"""
    
    device = CONFIG['device']
    print(f"Device: {device}\n")
    
    # Create results directory
    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # ────────────────────────────────────────────────────────────────────────
    # Load validation data
    # ────────────────────────────────────────────────────────────────────────
    print("="*80)
    print("LOADING DATA")
    print("="*80 + "\n")
    
    val_loader = get_cifar100_val_loader(
        CONFIG['batch_size'],
        CONFIG['num_workers']
    )
    
    # ────────────────────────────────────────────────────────────────────────
    # Load models
    # ────────────────────────────────────────────────────────────────────────
    print("="*80)
    print("LOADING MODELS")
    print("="*80 + "\n")
    
    existing_model = load_student(
        CHECKPOINTS['existing'],
        device,
        "Existing LM Student (VGG16 teacher)"
    )
    
    new_model = load_student(
        CHECKPOINTS['new'],
        device,
        "New LM Student (VGG19 teacher)"
    )
    
    # Check if both models loaded
    if existing_model is None or new_model is None:
        print("\n✗ ERROR: Could not load one or both models")
        print(f"  Existing checkpoint: {CHECKPOINTS['existing']}")
        print(f"  New checkpoint: {CHECKPOINTS['new']}")
        return
    
    print()
    
    # ────────────────────────────────────────────────────────────────────────
    # Evaluate models
    # ────────────────────────────────────────────────────────────────────────
    print("="*80)
    print("EVALUATING MODELS ON VALIDATION SET")
    print("="*80)
    
    existing_acc = evaluate_model(
        existing_model,
        val_loader,
        device,
        "Existing LM Student (VGG16)"
    )
    
    new_acc = evaluate_model(
        new_model,
        val_loader,
        device,
        "New LM Student (VGG19)"
    )
    
    # ────────────────────────────────────────────────────────────────────────
    # Analysis
    # ────────────────────────────────────────────────────────────────────────
    analysis = analyze_and_recommend(existing_acc, new_acc)
    
    # ────────────────────────────────────────────────────────────────────────
    # Visualization
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    plot_accuracy_comparison(existing_acc, new_acc, CONFIG['results_dir'])
    
    # ────────────────────────────────────────────────────────────────────────
    # Save results
    # ────────────────────────────────────────────────────────────────────────
    print("="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    save_comparison_json(analysis, CONFIG['results_dir'])
    
    # ────────────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    print(f"\nResults saved to: {CONFIG['results_dir']}/")
    print(f"  ├─ accuracy_comparison.png")
    print(f"  └─ accuracy_comparison.json")
    print()


if __name__ == "__main__":
    main()