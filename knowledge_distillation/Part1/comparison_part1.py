"""
Comparison script for Logit Matching, Label Smoothing, and Decoupled KD
Trains all three methods with the same data and hyperparameters, then compares results
"""

import torch
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils import get_device, get_cifar100_loaders, validate, vgg11
from logit_matching import LogitMatchingKD
from label_smoothing import LabelSmoothingKD
from decoupled_kd import DecoupledKD


class KDComparison:
    """Compare three KD methods"""
    
    def __init__(self):
        self.device = get_device()
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs('./results', exist_ok=True)
        os.makedirs('./checkpoints', exist_ok=True)
    
    def run_all_experiments(self, train_loader, val_loader, test_loader, num_epochs=30):
        """Run all three KD methods"""
        
        print("\n" + "="*80)
        print("KNOWLEDGE DISTILLATION COMPARISON - PART 1: LOGIT MATCHING")
        print("="*80)
        
        # 1. Basic Logit Matching
        print("\n[1/3] Running Basic Logit Matching...")
        lm_kd = LogitMatchingKD(
            temperature=4.0,
            alpha=0.5,
            learning_rate=0.1,
            weight_decay=5e-4
        )
        lm_best_val = lm_kd.train(train_loader, val_loader, num_epochs=num_epochs)
        lm_test = lm_kd.evaluate(test_loader)
        self.results['Logit Matching'] = {
            'val_acc': lm_best_val,
            'test_acc': lm_test,
            'train_accs': lm_kd.train_accs,
            'val_accs': lm_kd.val_accs,
            'train_losses': lm_kd.train_losses,
            'model': lm_kd.get_student_model(),
            'teacher': lm_kd.get_teacher_model()
        }
        
        print("\n" + "="*80)
        print("KNOWLEDGE DISTILLATION COMPARISON - PART 2: LABEL SMOOTHING")
        print("="*80)
        
        # 2. Label Smoothing
        print("\n[2/3] Running Label Smoothing...")
        ls_kd = LabelSmoothingKD(
            smoothing=0.1,
            learning_rate=0.1,
            weight_decay=5e-4
        )
        ls_best_val = ls_kd.train(train_loader, val_loader, num_epochs=num_epochs)
        ls_test = ls_kd.evaluate(test_loader)
        self.results['Label Smoothing'] = {
            'val_acc': ls_best_val,
            'test_acc': ls_test,
            'train_accs': ls_kd.train_accs,
            'val_accs': ls_kd.val_accs,
            'train_losses': ls_kd.train_losses,
            'model': ls_kd.get_student_model(),
            'teacher': ls_kd.get_teacher_model()
        }
        
        print("\n" + "="*80)
        print("KNOWLEDGE DISTILLATION COMPARISON - PART 3: DECOUPLED KD")
        print("="*80)
        
        # 3. Decoupled KD
        print("\n[3/3] Running Decoupled Knowledge Distillation...")
        dkd_kd = DecoupledKD(
            temperature=4.0,
            alpha=1.0,
            beta=1.0,
            ce_weight=1.0,
            learning_rate=0.1,
            weight_decay=5e-4
        )
        dkd_best_val = dkd_kd.train(train_loader, val_loader, num_epochs=num_epochs)
        dkd_test = dkd_kd.evaluate(test_loader)
        self.results['Decoupled KD'] = {
            'val_acc': dkd_best_val,
            'test_acc': dkd_test,
            'train_accs': dkd_kd.train_accs,
            'val_accs': dkd_kd.val_accs,
            'train_losses': dkd_kd.train_losses,
            'model': dkd_kd.get_student_model(),
            'teacher': dkd_kd.get_teacher_model()
        }
    
    def print_comparison_table(self):
        """Print comparison table"""
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY - PART 1: LOGIT MATCHING COMPARISON")
        print("="*80)
        print()
        print(f"{'Method':<25} {'Best Val Acc':<15} {'Test Acc':<15} {'Improvement*':<15}")
        print("-"*70)
        
        # Get baseline (assuming all use same random initialization for comparison)
        for method, data in self.results.items():
            val_acc = data['val_acc']
            test_acc = data['test_acc']
            # Improvement relative to best result
            improvement = val_acc - min([d['val_acc'] for d in self.results.values()])
            print(f"{method:<25} {val_acc:>6.2f}%        {test_acc:>6.2f}%        {improvement:>6.2f}%")
        
        print("-"*70)
        print("* Improvement relative to worst performing method")
        print()
    
    def plot_training_curves(self):
        """Plot training curves for all methods"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training Accuracy
        ax = axes[0, 0]
        for method, data in self.results.items():
            ax.plot(data['train_accs'], label=method, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Validation Accuracy
        ax = axes[0, 1]
        for method, data in self.results.items():
            ax.plot(data['val_accs'], label=method, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Training Loss
        ax = axes[1, 0]
        for method, data in self.results.items():
            ax.plot(data['train_losses'], label=method, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Final Accuracies (Bar Chart)
        ax = axes[1, 1]
        methods = list(self.results.keys())
        test_accs = [self.results[m]['test_acc'] for m in methods]
        val_accs = [self.results[m]['val_acc'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, val_accs, width, label='Validation', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = f'./results/kd_comparison_{self.timestamp}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to {fig_path}")
        
        plt.close()
    
    def save_results_json(self):
        """Save results to JSON file"""
        
        results_data = {}
        for method, data in self.results.items():
            results_data[method] = {
                'val_accuracy': float(data['val_acc']),
                'test_accuracy': float(data['test_acc']),
                'train_accs': [float(x) for x in data['train_accs']],
                'val_accs': [float(x) for x in data['val_accs']],
                'train_losses': [float(x) for x in data['train_losses']],
            }
        
        json_path = f'./results/kd_comparison_{self.timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {json_path}")
    
    def generate_analysis_report(self):
        """Generate detailed analysis report"""
        
        report_path = f'./results/kd_comparison_analysis_{self.timestamp}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("KNOWLEDGE DISTILLATION COMPARISON ANALYSIS\n")
            f.write("Part 1: Logit Matching Methods\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n")
            
            # Summary table
            f.write("-"*80 + "\n")
            f.write("RESULTS SUMMARY\n")
            f.write("-"*80 + "\n\n")
            
            f.write(f"{'Method':<25} {'Best Val Acc':<15} {'Test Acc':<15}\n")
            f.write("-"*55 + "\n")
            
            for method, data in self.results.items():
                f.write(f"{method:<25} {data['val_acc']:>6.2f}%        {data['test_acc']:>6.2f}%\n")
            
            f.write("\n\n")
            
            # Detailed analysis for each method
            f.write("-"*80 + "\n")
            f.write("DETAILED ANALYSIS\n")
            f.write("-"*80 + "\n\n")
            
            f.write("1. BASIC LOGIT MATCHING\n")
            f.write("   How it works:\n")
            f.write("   - Student learns to match teacher's softened logits (temperature-scaled)\n")
            f.write("   - Uses KL divergence loss between softened distributions\n")
            f.write("   - Combines distillation loss with standard cross-entropy loss\n")
            f.write("   - Alpha parameter balances KD loss and CE loss\n\n")
            
            f.write("2. LABEL SMOOTHING\n")
            f.write("   How it works:\n")
            f.write("   - Regularization technique that softens target labels\n")
            f.write("   - Instead of one-hot [0,1,0,...], uses [0.001, 0.901, 0.001,...]\n")
            f.write("   - Prevents model overconfidence on training data\n")
            f.write("   - Does NOT directly use teacher (teacher role is implicit)\n\n")
            
            f.write("3. DECOUPLED KNOWLEDGE DISTILLATION (DKD)\n")
            f.write("   How it works:\n")
            f.write("   - Separates distillation into two components:\n")
            f.write("     * TCKD: Target Class KD (emphasizes correct class)\n")
            f.write("     * NCKD: Non-Target Class KD (differentiates wrong classes)\n")
            f.write("   - Gives finer control over what knowledge is transferred\n")
            f.write("   - More sophisticated than basic logit matching\n\n")
            
            # Performance analysis
            f.write("-"*80 + "\n")
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("-"*80 + "\n\n")
            
            val_accs = {m: d['val_acc'] for m, d in self.results.items()}
            best_method = max(val_accs, key=val_accs.get)
            best_acc = val_accs[best_method]
            worst_acc = min(val_accs.values())
            
            f.write(f"Best performing method: {best_method} ({best_acc:.2f}%)\n")
            f.write(f"Accuracy range: {worst_acc:.2f}% - {best_acc:.2f}%\n")
            f.write(f"Range: {best_acc - worst_acc:.2f}%\n\n")
            
            for method, acc in sorted(val_accs.items(), key=lambda x: x[1], reverse=True):
                improvement = acc - worst_acc
                f.write(f"{method:<25}: {acc:.2f}% (+{improvement:.2f}%)\n")
            
            f.write("\n\n")
            
            # Key insights
            f.write("-"*80 + "\n")
            f.write("KEY INSIGHTS & OBSERVATIONS\n")
            f.write("-"*80 + "\n\n")
            
            f.write("1. Comparison of Methods:\n")
            f.write("   - LM is direct knowledge transfer via soft targets\n")
            f.write("   - LS is indirect, regularizing the student's confidence\n")
            f.write("   - DKD is more nuanced, decomposing the knowledge transfer\n\n")
            
            f.write("2. Expected Trade-offs:\n")
            f.write("   - LM may overfit to matching teacher if alpha too high\n")
            f.write("   - LS may not fully leverage teacher knowledge\n")
            f.write("   - DKD has more parameters to tune (alpha, beta)\n\n")
            
            f.write("3. Training Stability:\n")
            f.write("   - Check training loss curves for oscillations\n")
            f.write("   - Monitor if val accuracy plateaus early\n")
            f.write("   - Look for convergence speed differences\n\n")
            
            f.write("-"*80 + "\n")
            f.write("END OF ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Analysis report saved to {report_path}")


def main():
    """Main comparison script"""
    
    print("\n" + "="*80)
    print("PART 1: LOGIT MATCHING KNOWLEDGE DISTILLATION COMPARISON")
    print("Comparing: Basic Logit Matching, Label Smoothing, Decoupled KD")
    print("="*80 + "\n")
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    train_loader, val_loader, test_loader, num_classes = get_cifar100_loaders(
        batch_size=128,
        num_workers=4
    )
    
    # # Run comparison
    comparison = KDComparison()
    
    print("\nStarting experiments (this will take a while)...")
    print("Number of epochs: 30\n")
    
    comparison.run_all_experiments(
        train_loader, val_loader, test_loader,
        num_epochs=30  # Change to lower value for testing
    )
    
    # Print and save results
    comparison.print_comparison_table()
    comparison.plot_training_curves()
    comparison.save_results_json()
    comparison.generate_analysis_report()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("Results saved to ./results/ directory")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()