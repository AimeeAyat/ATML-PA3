"""
Example Usage: Part 3 - Probability Distribution Alignment Analysis

This script demonstrates how to:
1. Load your trained models from Part 1 & Part 2
2. Extract probability distributions from all models
3. Compute alignment metrics using multiple divergence measures
4. Generate comparison visualizations

SETUP INSTRUCTIONS:
1. Update checkpoint paths below to match your saved model locations
2. Ensure you have a validation/test dataset loader ready
3. Run this script to generate distribution analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part3_distribution_alignment import DistributionAnalyzer
from Part1.utils import vgg11, vgg16


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model architecture: VGG-16 (Teacher), VGG-11 (Students)
# Teacher will be loaded from utils (not from checkpoint)
# Students will be loaded from their respective checkpoints

# ============================================================================
# CHECKPOINT PATHS - These match your actual checkpoint files
# ============================================================================

CHECKPOINT_PATHS = {
    'teacher': {
        'path': None,  # Teacher loaded directly from utils, not from checkpoint
        'model_fn': lambda: vgg16(num_classes=100),
        'description': 'VGG-16 (Teacher)'
    },
    'SI': {
        'path': './checkpoints/si_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Independent Student (No Distillation)'
    },
    'LM': {
        'path': './checkpoints/lm_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Logit Matching Student'
    },
    'LS': {
        'path': './checkpoints/ls_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Label Smoothing Student'
    },
    'DKD': {
        'path': './checkpoints/dkd_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Decoupled KD Student'
    },
    'Hints': {
        'path': './checkpoints/hints_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Hints-based Distillation Student'
    },
    'CRD': {
        'path': './checkpoints/crd_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Contrastive Representation Distillation Student'
    }
}


# ============================================================================
# DATASET SETUP
# ============================================================================

def get_cifar100_loader(split='val', batch_size=256, num_workers=4):
    """
    Load CIFAR-100 dataset using the same configuration as Part 1 and Part 2.
    """
    
    # Use the same normalization as in utils.py
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    if split == 'val' or split == 'test':
        dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            transform=transform,
            download=True
        )
    else:  # train
        dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            transform=transform,
            download=True
        )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_distribution_analysis(checkpoint_paths, subset_size=None):
    """
    Main function to run Part 3 analysis.
    
    Args:
        checkpoint_paths: Dictionary of checkpoint information
        subset_size: If specified, only analyze first N samples (for testing)
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Initialize analyzer
    analyzer = DistributionAnalyzer(device=device)
    
    # Get validation loader
    print("Loading CIFAR-100 validation set...")
    val_loader = get_cifar100_loader('val', batch_size=256)
    
    # If subset is specified, create a smaller loader
    if subset_size is not None:
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_loader.dataset, range(subset_size)),
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        print(f"Using subset of {subset_size} samples for analysis")
    
    # ========================================================================
    # 1. LOAD ALL MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    models_loaded = {}
    model_configs = {}
    
    for model_key, config in checkpoint_paths.items():
        try:
            # Create model instance
            model = config['model_fn']().to(device)
            
            # Load checkpoint if path is provided (not for teacher)
            if config['path'] is not None:
                checkpoint_path = config['path']
                
                # Check if checkpoint exists
                if not Path(checkpoint_path).exists():
                    print(f"⚠ Warning: {checkpoint_path} not found, skipping {model_key}")
                    continue
                
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Extract model_state_dict if checkpoint contains metadata
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                print(f"  ✓ Loaded {config['description']} from checkpoint")
            else:
                # Teacher model - no checkpoint needed
                print(f"  ✓ Initialized {config['description']} (randomly initialized)")
            
            model.eval()
            models_loaded[model_key] = model
            model_configs[model_key] = config
            
        except Exception as e:
            print(f"✗ Error loading {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if 'teacher' not in models_loaded:
        print("\n✗ ERROR: Teacher model not loaded. Cannot proceed with analysis.")
        return
    
    # ========================================================================
    # 2. EXTRACT DISTRIBUTIONS FROM ALL MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("EXTRACTING PROBABILITY DISTRIBUTIONS")
    print("="*80 + "\n")
    
    teacher = models_loaded['teacher']
    print("Teacher Model:")
    teacher_outputs = analyzer.get_output_distributions(teacher, val_loader, temperature=1.0)
    print(f"  Samples: {len(teacher_outputs['distributions'])}")
    print(f"  Shape: {teacher_outputs['distributions'].shape}")
    print(f"  Accuracy: {np.mean(teacher_outputs['predictions'] == teacher_outputs['targets']) * 100:.2f}%\n")
    
    # Extract from all students
    student_names = [k for k in models_loaded.keys() if k != 'teacher']
    student_outputs_dict = {}
    
    print("Student Models:")
    for student_name in student_names:
        print(f"  {student_name}...")
        student = models_loaded[student_name]
        student_outputs = analyzer.get_output_distributions(student, val_loader, temperature=1.0)
        student_outputs_dict[student_name] = student_outputs
        acc = np.mean(student_outputs['predictions'] == student_outputs['targets']) * 100
        print(f"    Accuracy: {acc:.2f}%")
    
    # ========================================================================
    # 3. ANALYZE ALIGNMENT
    # ========================================================================
    print("\n" + "="*80)
    results = analyzer.analyze_alignment(
        teacher_outputs,
        student_outputs_dict,
        student_names,
        temperatures=[1.0, 2.0, 4.0, 8.0]
    )
    print("="*80)
    
    # ========================================================================
    # 4. CREATE COMPARISON TABLE
    # ========================================================================
    # Create results directory first
    os.makedirs('./results/part3', exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING COMPARISON TABLE")
    print("="*80)
    comparison = analyzer.create_comparison_table(results, save_path='./results/part3/distribution_alignment_results.json')
    print("\nComparison Summary:")
    print(f"{'Model':<20} {'Accuracy':<12} {'KL Div':<12} {'JS Div':<12} {'Cosine Sim':<12}")
    print("-" * 68)
    
    for student_name in student_names:
        if student_name in comparison['students_metrics']:
            metrics = comparison['students_metrics'][student_name]
            acc_pct = metrics['accuracy'] * 100
            print(f"{student_name:<20} {acc_pct:<12.2f} "
                  f"{metrics['kl_divergence_mean']:<12.6f} {metrics['js_divergence_mean']:<12.6f} "
                  f"{metrics['cosine_similarity_mean']:<12.4f}")
    
    # ========================================================================
    # 5. GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    analyzer.plot_divergence_comparison(results, student_names, save_dir='./results/part3')
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print("  - ./results/part3/distribution_alignment_results.json (metrics)")
    print("  - ./results/part3/ (visualizations)")
    
    return results, comparison


# ============================================================================
# QUICK TEST MODE
# ============================================================================

def run_quick_test():
    """
    Quick test to verify setup without full analysis.
    """
    print("Quick Test Mode: Loading first 500 samples\n")
    
    results, comparison = run_distribution_analysis(
        CHECKPOINT_PATHS,
        subset_size=500
    )
    
    return results, comparison


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Part 3: Distribution Alignment Analysis')
    parser.add_argument('--quick', action='store_true', help='Run quick test with 500 samples')
    parser.add_argument('--subset', type=int, default=None, help='Analyze first N samples')
    
    args = parser.parse_args()
    
    if args.quick:
        results, comparison = run_quick_test()
    else:
        results, comparison = run_distribution_analysis(CHECKPOINT_PATHS, subset_size=args.subset)