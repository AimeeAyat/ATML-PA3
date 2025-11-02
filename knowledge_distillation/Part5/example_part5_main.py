"""
Part 5: Color Invariance Analysis - Example Usage

Complete pipeline:
1. Load pre-trained teacher
2. Fine-tune teacher with color jitter augmentations
3. Evaluate color-invariant teacher
4. Distill to student using CRD
5. Evaluate student on color-jittered data
6. Compare with other methods
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Part1.utils import vgg11, vgg16

from part5_color_invariance_ import (
    ColorInvarianceAnalyzer,
    get_cifar100_loaders
)




CHECKPOINT_PATHS = {
    'original_teacher': {
        'path': None,  # Teacher loaded directly from utils, not from checkpoint
        'model_fn': lambda: vgg16(num_classes=100),
        'description': 'VGG-16 (Teacher)'
    },
    'si_student': {
        'path': './checkpoints/si_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Independent Student (No Distillation)'
    },
    'lm_student': {
        'path': './checkpoints/ls_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Logit Matching Student'
    },
    'crd_student': {
        'path': './checkpoints/crd_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'CRD Student'
    },
}


# ============================================================================
# ANALYSIS PIPELINE
# ============================================================================

def run_part5_analysis(quick_mode=False, skip_finetuning=False):
    """
    Complete Part 5 analysis pipeline.
    
    Args:
        quick_mode: Quick test (fewer epochs)
        skip_finetuning: Skip teacher fine-tuning (use original)
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    analyzer = ColorInvarianceAnalyzer(device=device)
    
    # ========================================================================
    # STAGE 1: LOAD AND EVALUATE ORIGINAL TEACHER
    # ========================================================================
    print("="*80)
    print("STAGE 1: ORIGINAL TEACHER EVALUATION")
    print("="*80)
    
    teacher_path = CHECKPOINT_PATHS['original_teacher']['path']
    
    # Create teacher model
    original_teacher = CHECKPOINT_PATHS['original_teacher']['model_fn']()
    
    # Load from checkpoint if path is provided
    if teacher_path is not None:
        if not Path(teacher_path).exists():
            print(f"✗ Error: Teacher checkpoint not found at {teacher_path}")
            return
        
        print(f"\nLoading teacher from: {teacher_path}")
        checkpoint = torch.load(teacher_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            teacher_state = checkpoint['model_state_dict']
        else:
            teacher_state = checkpoint
        
        original_teacher.load_state_dict(teacher_state, strict=False)
    else:
        print("\nCreating fresh teacher model (VGG-16) with random initialization")
    
    # Move model to device
    original_teacher = original_teacher.to(device)
    
    # Evaluate original teacher
    _, val_loader, jittered_loader = get_cifar100_loaders('normal', batch_size=256)
    
    print("\nEvaluating original teacher...")
    results_orig_teacher = analyzer.evaluate_model(
        original_teacher, val_loader, jittered_loader, 'Original Teacher'
    )
    
    print(f"  Normal Accuracy: {results_orig_teacher['accuracy_normal']:.4f}")
    print(f"  Jittered Accuracy: {results_orig_teacher['accuracy_jittered']:.4f}")
    print(f"  Accuracy Drop: {results_orig_teacher['accuracy_drop']:.4f}")
    print(f"  Invariance Ratio: {results_orig_teacher['invariance_ratio']:.4f}")
    
    # ========================================================================
    # STAGE 2: FINE-TUNE TEACHER WITH COLOR JITTER
    # ========================================================================
    print("\n" + "="*80)
    print("STAGE 2: FINE-TUNE TEACHER WITH COLOR JITTER")
    print("="*80)
    
    if skip_finetuning:
        print("\nSkipping fine-tuning (using original teacher)")
        color_invariant_teacher = original_teacher
    else:
        train_loader, val_loader, jittered_loader = get_cifar100_loaders(
            'color_jitter', batch_size=128
        )
        
        num_epochs = 3 if quick_mode else 10
        
        color_invariant_teacher, history = analyzer.fine_tune_teacher(
            original_teacher,
            train_loader,
            val_loader,
            jittered_loader,
            num_epochs=num_epochs,
            learning_rate=0.01
        )
        
        # Evaluate fine-tuned teacher
        _, val_loader, jittered_loader = get_cifar100_loaders('normal', batch_size=256)
        
        print("\nEvaluating color-invariant teacher...")
        results_inv_teacher = analyzer.evaluate_model(
            color_invariant_teacher, val_loader, jittered_loader, 
            'Color-Invariant Teacher'
        )
        
        print(f"  Normal Accuracy: {results_inv_teacher['accuracy_normal']:.4f}")
        print(f"  Jittered Accuracy: {results_inv_teacher['accuracy_jittered']:.4f}")
        print(f"  Accuracy Drop: {results_inv_teacher['accuracy_drop']:.4f}")
        print(f"  Invariance Ratio: {results_inv_teacher['invariance_ratio']:.4f}")
    
    # ========================================================================
    # STAGE 3: CRD DISTILLATION
    # ========================================================================
    print("\n" + "="*80)
    print("STAGE 3: DISTILL COLOR-INVARIANT TEACHER TO STUDENT WITH CRD")
    print("="*80)
    
    student_crd = CHECKPOINT_PATHS['crd_student']['model_fn']()
    
    train_loader, val_loader, jittered_loader = get_cifar100_loaders(
        'normal', batch_size=128  # Normal augmentations for student training
    )
    
    num_epochs = 5 if quick_mode else 100
    
    student_crd, history = analyzer.knowledge_distillation_crd(
        color_invariant_teacher,
        student_crd,
        train_loader,
        val_loader,
        jittered_loader,
        num_epochs=num_epochs,
        temperature=4.0,
        alpha=0.5,
        learning_rate=0.1
    )
    
    # Evaluate CRD student
    print("\nEvaluating CRD student on color-jittered data...")
    results_crd_student = analyzer.evaluate_model(
        student_crd, val_loader, jittered_loader, 'CRD Student (From Color-Invariant Teacher)'
    )
    
    print(f"  Normal Accuracy: {results_crd_student['accuracy_normal']:.4f}")
    print(f"  Jittered Accuracy: {results_crd_student['accuracy_jittered']:.4f}")
    print(f"  Accuracy Drop: {results_crd_student['accuracy_drop']:.4f}")
    print(f"  Invariance Ratio: {results_crd_student['invariance_ratio']:.4f}")
    
    # ========================================================================
    # STAGE 4: LOAD OTHER STUDENTS FOR COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("STAGE 4: COMPARE WITH OTHER KD METHODS")
    print("="*80)
    
    # SI (Independent)
    if Path(CHECKPOINT_PATHS['si_student']['path']).exists():
        print("\nLoading and evaluating SI (independent student)...")
        si_checkpoint = torch.load(CHECKPOINT_PATHS['si_student']['path'], map_location=device)
        if isinstance(si_checkpoint, dict) and 'model_state_dict' in si_checkpoint:
            si_state = si_checkpoint['model_state_dict']
        else:
            si_state = si_checkpoint
        
        si_student = CHECKPOINT_PATHS['si_student']['model_fn']()
        si_student.load_state_dict(si_state, strict=False)
        si_student = si_student.to(device)
        
        results_si = analyzer.evaluate_model(
            si_student, val_loader, jittered_loader, 'Independent Student (SI)'
        )
        print(f"  Normal Accuracy: {results_si['accuracy_normal']:.4f}")
        print(f"  Jittered Accuracy: {results_si['accuracy_jittered']:.4f}")
        print(f"  Invariance Ratio: {results_si['invariance_ratio']:.4f}")
    
    # LM (Logit Matching)
    if Path(CHECKPOINT_PATHS['lm_student']['path']).exists():
        print("\nLoading and evaluating LM (Logit Matching)...")
        lm_checkpoint = torch.load(CHECKPOINT_PATHS['lm_student']['path'], map_location=device)
        if isinstance(lm_checkpoint, dict) and 'model_state_dict' in lm_checkpoint:
            lm_state = lm_checkpoint['model_state_dict']
        else:
            lm_state = lm_checkpoint
        
        lm_student = CHECKPOINT_PATHS['lm_student']['model_fn']()
        lm_student.load_state_dict(lm_state, strict=False)
        lm_student = lm_student.to(device)
        
        results_lm = analyzer.evaluate_model(
            lm_student, val_loader, jittered_loader, 'Logit Matching (LM)'
        )
        print(f"  Normal Accuracy: {results_lm['accuracy_normal']:.4f}")
        print(f"  Jittered Accuracy: {results_lm['accuracy_jittered']:.4f}")
        print(f"  Invariance Ratio: {results_lm['invariance_ratio']:.4f}")
    
    # ========================================================================
    # STAGE 5: SUMMARY AND EXPORT
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80 + "\n")
    
    print(f"{'Model':<45} {'Normal':<12} {'Jittered':<12} {'Drop':<12} {'Invariance':<10}")
    print("-"*100)
    
    for model_name, result in analyzer.results.items():
        print(f"{model_name:<45} {result['accuracy_normal']:<12.4f} "
              f"{result['accuracy_jittered']:<12.4f} {result['accuracy_drop']:<12.4f} "
              f"{result['invariance_ratio']:<10.4f}")
    
    # Export results
    analyzer.compare_methods('color_invariance_results.json')
    
    # Create visualizations
    analyzer.plot_comparison('part5_plots')
    
    print("\n✓ Analysis complete!")
    print("Output files:")
    print("  - color_invariance_results.json")
    print("  - part5_plots/accuracy_comparison.png")
    print("  - part5_plots/invariance_ratio.png")
    
    return analyzer


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Part 5: Color Invariance with CRD')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer epochs)')
    parser.add_argument('--skip-finetuning', action='store_true', 
                       help='Skip teacher fine-tuning')
    
    args = parser.parse_args()
    
    analyzer = run_part5_analysis(
        quick_mode=args.quick,
        skip_finetuning=args.skip_finetuning
    )