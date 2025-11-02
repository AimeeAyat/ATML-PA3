"""
Comparison script for Logit Matching, Label Smoothing, and Decoupled KD
Loads pre-trained checkpoints and performs comprehensive comparison
"""

import torch
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils import get_device, get_cifar100_loaders, validate, vgg11, vgg16
from logit_matching import LogitMatchingKD
from label_smoothing import LabelSmoothingKD
from decoupled_kd import DecoupledKD


class KDComparison:
    """Compare three KD methods using pre-trained checkpoints"""
    
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.device = get_device()
        self.checkpoint_dir = checkpoint_dir
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs('./results', exist_ok=True)
        
        # Define checkpoint paths
        self.checkpoints = {
            'Logit Matching': os.path.join(checkpoint_dir, 'lm_student_best.pth'),
            'Label Smoothing': os.path.join(checkpoint_dir, 'ls_student_best.pth'),
            'Decoupled KD': os.path.join(checkpoint_dir, 'dkd_student_best.pth'),
        }
    
    def load_and_evaluate(self, train_loader, val_loader, test_loader):
        """Load checkpoints and evaluate all models"""
        
        print("\n" + "="*80)
        print("LOADING PRE-TRAINED CHECKPOINTS AND EVALUATING")
        print("="*80 + "\n")
        
        # 1. Logit Matching
        print("[1/3] Loading Logit Matching checkpoint...")
        if os.path.exists(self.checkpoints['Logit Matching']):
            lm_student = vgg11(num_classes=100).to(self.device)
            lm_teacher = vgg16(num_classes=100).to(self.device)
            
            checkpoint = torch.load(self.checkpoints['Logit Matching'], map_location=self.device)
            # Extract model_state_dict if checkpoint contains metadata
            if 'model_state_dict' in checkpoint:
                lm_student.load_state_dict(checkpoint['model_state_dict'])
            else:
                lm_student.load_state_dict(checkpoint)
            
            # Load teacher (if available in checkpoint or use trained one)
            teacher_path = os.path.join(self.checkpoint_dir, 'teacher_best.pth')
            if os.path.exists(teacher_path):
                teacher_ckpt = torch.load(teacher_path, map_location=self.device)
                if 'model_state_dict' in teacher_ckpt:
                    lm_teacher.load_state_dict(teacher_ckpt['model_state_dict'])
                else:
                    lm_teacher.load_state_dict(teacher_ckpt)
            
            # Evaluate
            lm_val = self._evaluate_model(lm_student, val_loader)
            lm_test = self._evaluate_model(lm_student, test_loader)
            lm_train = self._evaluate_model(lm_student, train_loader)
            
            self.results['Logit Matching'] = {
                'val_acc': lm_val,
                'test_acc': lm_test,
                'train_acc': lm_train,
                'model': lm_student,
                'teacher': lm_teacher,
                'loaded': True
            }
            print(f"✓ Loaded successfully!")
            print(f"  Train Acc: {lm_train:.2f}%")
            print(f"  Val Acc:   {lm_val:.2f}%")
            print(f"  Test Acc:  {lm_test:.2f}%\n")
        else:
            print(f"✗ Checkpoint not found: {self.checkpoints['Logit Matching']}\n")
            self.results['Logit Matching'] = None
        
        # 2. Label Smoothing
        print("[2/3] Loading Label Smoothing checkpoint...")
        if os.path.exists(self.checkpoints['Label Smoothing']):
            ls_student = vgg11(num_classes=100).to(self.device)
            ls_teacher = vgg16(num_classes=100).to(self.device)
            
            checkpoint = torch.load(self.checkpoints['Label Smoothing'], map_location=self.device)
            # Extract model_state_dict if checkpoint contains metadata
            if 'model_state_dict' in checkpoint:
                ls_student.load_state_dict(checkpoint['model_state_dict'])
            else:
                ls_student.load_state_dict(checkpoint)
            
            # Load teacher
            teacher_path = os.path.join(self.checkpoint_dir, 'teacher_best.pth')
            if os.path.exists(teacher_path):
                teacher_ckpt = torch.load(teacher_path, map_location=self.device)
                if 'model_state_dict' in teacher_ckpt:
                    ls_teacher.load_state_dict(teacher_ckpt['model_state_dict'])
                else:
                    ls_teacher.load_state_dict(teacher_ckpt)
            
            # Evaluate
            ls_val = self._evaluate_model(ls_student, val_loader)
            ls_test = self._evaluate_model(ls_student, test_loader)
            ls_train = self._evaluate_model(ls_student, train_loader)
            
            self.results['Label Smoothing'] = {
                'val_acc': ls_val,
                'test_acc': ls_test,
                'train_acc': ls_train,
                'model': ls_student,
                'teacher': ls_teacher,
                'loaded': True
            }
            print(f"✓ Loaded successfully!")
            print(f"  Train Acc: {ls_train:.2f}%")
            print(f"  Val Acc:   {ls_val:.2f}%")
            print(f"  Test Acc:  {ls_test:.2f}%\n")
        else:
            print(f"✗ Checkpoint not found: {self.checkpoints['Label Smoothing']}\n")
            self.results['Label Smoothing'] = None
        
        # 3. Decoupled KD
        print("[3/3] Loading Decoupled KD checkpoint...")
        if os.path.exists(self.checkpoints['Decoupled KD']):
            dkd_student = vgg11(num_classes=100).to(self.device)
            dkd_teacher = vgg16(num_classes=100).to(self.device)
            
            checkpoint = torch.load(self.checkpoints['Decoupled KD'], map_location=self.device)
            # Extract model_state_dict if checkpoint contains metadata
            if 'model_state_dict' in checkpoint:
                dkd_student.load_state_dict(checkpoint['model_state_dict'])
            else:
                dkd_student.load_state_dict(checkpoint)
            
            # Load teacher
            teacher_path = os.path.join(self.checkpoint_dir, 'teacher_best.pth')
            if os.path.exists(teacher_path):
                teacher_ckpt = torch.load(teacher_path, map_location=self.device)
                if 'model_state_dict' in teacher_ckpt:
                    dkd_teacher.load_state_dict(teacher_ckpt['model_state_dict'])
                else:
                    dkd_teacher.load_state_dict(teacher_ckpt)
            
            # Evaluate
            dkd_val = self._evaluate_model(dkd_student, val_loader)
            dkd_test = self._evaluate_model(dkd_student, test_loader)
            dkd_train = self._evaluate_model(dkd_student, train_loader)
            
            self.results['Decoupled KD'] = {
                'val_acc': dkd_val,
                'test_acc': dkd_test,
                'train_acc': dkd_train,
                'model': dkd_student,
                'teacher': dkd_teacher,
                'loaded': True
            }
            print(f"✓ Loaded successfully!")
            print(f"  Train Acc: {dkd_train:.2f}%")
            print(f"  Val Acc:   {dkd_val:.2f}%")
            print(f"  Test Acc:  {dkd_test:.2f}%\n")
        else:
            print(f"✗ Checkpoint not found: {self.checkpoints['Decoupled KD']}\n")
            self.results['Decoupled KD'] = None
        
        # Filter out None results
        self.results = {k: v for k, v in self.results.items() if v is not None}
    
    def _evaluate_model(self, model, data_loader):
        """Evaluate model on data loader"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = (correct / total) * 100
        return accuracy
    
    def print_comparison_table(self):
        """Print comprehensive comparison table"""
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS - PART 1: LOGIT MATCHING")
        print("="*80 + "\n")
        
        if not self.results:
            print("No results to display. Check checkpoint paths.\n")
            return
        
        print(f"{'Method':<25} {'Train Acc':<15} {'Val Acc':<15} {'Test Acc':<15}")
        print("-"*70)
        
        methods_data = []
        for method, data in self.results.items():
            train_acc = data['train_acc']
            val_acc = data['val_acc']
            test_acc = data['test_acc']
            methods_data.append((method, train_acc, val_acc, test_acc))
            print(f"{method:<25} {train_acc:>6.2f}%        {val_acc:>6.2f}%        {test_acc:>6.2f}%")
        
        print("-"*70)
        
        # Statistics
        if methods_data:
            val_accs = [d[2] for d in methods_data]
            test_accs = [d[3] for d in methods_data]
            
            print(f"\nBest Val Accuracy:  {max(val_accs):.2f}% ({methods_data[np.argmax(val_accs)][0]})")
            print(f"Best Test Accuracy: {max(test_accs):.2f}% ({methods_data[np.argmax(test_accs)][0]})")
            print(f"Val Accuracy Range: {max(val_accs) - min(val_accs):.2f}%")
            print(f"Test Accuracy Range: {max(test_accs) - min(test_accs):.2f}%")
        
        print()
    
    def plot_comparison(self):
        """Create comparison visualizations"""
        
        if not self.results:
            print("No results to plot.\n")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = list(self.results.keys())
        train_accs = [self.results[m]['train_acc'] for m in methods]
        val_accs = [self.results[m]['val_acc'] for m in methods]
        test_accs = [self.results[m]['test_acc'] for m in methods]
        
        # Plot 1: All accuracies comparison
        ax = axes[0, 0]
        x = np.arange(len(methods))
        width = 0.25
        
        bars1 = ax.bar(x - width, train_accs, width, label='Train', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, val_accs, width, label='Validation', alpha=0.8, color='orange')
        bars3 = ax.bar(x + width, test_accs, width, label='Test', alpha=0.8, color='green')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy Comparison Across All Splits', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Test accuracy focus
        ax = axes[0, 1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.barh(methods, test_accs, color=colors, alpha=0.8)
        ax.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Test Accuracy Ranking', fontsize=13, fontweight='bold')
        ax.set_xlim([0, 100])
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (method, acc) in enumerate(zip(methods, test_accs)):
            ax.text(acc + 1, i, f'{acc:.2f}%', va='center', fontsize=11, fontweight='bold')
        
        # Plot 3: Train vs Val (overfitting indicator)
        ax = axes[1, 0]
        gap = [train_accs[i] - val_accs[i] for i in range(len(methods))]
        colors_gap = ['red' if g > 5 else 'orange' if g > 2 else 'green' for g in gap]
        bars = ax.bar(methods, gap, color=colors_gap, alpha=0.8)
        ax.set_ylabel('Accuracy Gap (%)', fontsize=12, fontweight='bold')
        ax.set_title('Train-Val Gap (Overfitting Indicator)', fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate Overfitting')
        ax.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High Overfitting')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # Plot 4: Summary table as text
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create table data
        table_data = []
        table_data.append(['Method', 'Train', 'Val', 'Test', 'Train-Val Gap'])
        
        for method in methods:
            train = self.results[method]['train_acc']
            val = self.results[method]['val_acc']
            test = self.results[method]['test_acc']
            gap = train - val
            table_data.append([
                method,
                f'{train:.2f}%',
                f'{val:.2f}%',
                f'{test:.2f}%',
                f'{gap:.2f}%'
            ])
        
        # Add summary statistics
        table_data.append(['', '', '', '', ''])
        val_accs_list = [self.results[m]['val_acc'] for m in methods]
        test_accs_list = [self.results[m]['test_acc'] for m in methods]
        best_method_val = methods[np.argmax(val_accs_list)]
        best_method_test = methods[np.argmax(test_accs_list)]
        
        table_data.append(['Best Val', best_method_val, f'{max(val_accs_list):.2f}%', '', ''])
        table_data.append(['Best Test', best_method_test, f'{max(test_accs_list):.2f}%', '', ''])
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                        colWidths=[0.2, 0.15, 0.15, 0.15, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style summary rows
        for i in range(len(table_data) - 3, len(table_data)):
            for j in range(5):
                table[(i, j)].set_facecolor('#e6e6e6')
                table[(i, j)].set_text_props(weight='bold')
        
        ax.set_title('Detailed Results Summary', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = f'./results/kd_comparison_{self.timestamp}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison figure saved to: {fig_path}")
        
        plt.close()
    
    def save_results_json(self):
        """Save results to JSON file"""
        
        if not self.results:
            print("No results to save.\n")
            return
        
        results_data = {}
        for method, data in self.results.items():
            results_data[method] = {
                'train_accuracy': float(data['train_acc']),
                'val_accuracy': float(data['val_acc']),
                'test_accuracy': float(data['test_acc']),
            }
        
        json_path = f'./results/kd_comparison_{self.timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✓ Results saved to: {json_path}")
    
    def generate_analysis_report(self):
        """Generate detailed analysis report"""
        
        if not self.results:
            print("No results to analyze.\n")
            return
        
        report_path = f'./results/kd_comparison_analysis_{self.timestamp}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("KNOWLEDGE DISTILLATION COMPARISON - LOADED CHECKPOINT ANALYSIS\n")
            f.write("Part 1: Logit Matching Methods\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"Checkpoint Directory: {os.path.abspath(self.checkpoint_dir)}\n\n")
            
            # Summary table
            f.write("-"*80 + "\n")
            f.write("RESULTS SUMMARY\n")
            f.write("-"*80 + "\n\n")
            
            f.write(f"{'Method':<25} {'Train Acc':<15} {'Val Acc':<15} {'Test Acc':<15}\n")
            f.write("-"*70 + "\n")
            
            for method, data in self.results.items():
                f.write(f"{method:<25} {data['train_acc']:>6.2f}%        {data['val_acc']:>6.2f}%        {data['test_acc']:>6.2f}%\n")
            
            f.write("\n\n")
            
            # Method descriptions
            f.write("-"*80 + "\n")
            f.write("METHOD DESCRIPTIONS\n")
            f.write("-"*80 + "\n\n")
            
            f.write("1. BASIC LOGIT MATCHING\n")
            f.write("   How it works:\n")
            f.write("   - Student learns to match teacher's softened logits (temperature-scaled)\n")
            f.write("   - Uses KL divergence loss between softened distributions\n")
            f.write("   - Combines distillation loss with standard cross-entropy loss\n")
            f.write("   - Alpha parameter balances KD loss (default: 0.5)\n")
            f.write("   - Temperature parameter controls softness of distributions (default: 4.0)\n")
            f.write("   - Most straightforward KD approach\n\n")
            
            f.write("2. LABEL SMOOTHING\n")
            f.write("   How it works:\n")
            f.write("   - Regularization technique that softens target labels\n")
            f.write("   - Instead of one-hot [0,1,0,...], uses [0.001, 0.901, 0.001,...]\n")
            f.write("   - Prevents model overconfidence on training data\n")
            f.write("   - Does NOT directly use teacher (teacher role is implicit regularization)\n")
            f.write("   - Smoother parameter controls label smoothing strength\n")
            f.write("   - Simpler alternative to explicit knowledge distillation\n\n")
            
            f.write("3. DECOUPLED KNOWLEDGE DISTILLATION (DKD)\n")
            f.write("   How it works:\n")
            f.write("   - Separates distillation into two components:\n")
            f.write("     * TCKD: Target Class KD (emphasizes correct class matching)\n")
            f.write("     * NCKD: Non-Target Class KD (differentiates wrong classes)\n")
            f.write("   - Gives finer control over what knowledge is transferred\n")
            f.write("   - More sophisticated than basic logit matching\n")
            f.write("   - Typically outperforms standard KD on similar compute budgets\n")
            f.write("   - Multiple hyperparameters: alpha, beta, temperature\n\n")
            
            # Performance analysis
            f.write("-"*80 + "\n")
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("-"*80 + "\n\n")
            
            val_accs = {m: d['val_acc'] for m, d in self.results.items()}
            test_accs = {m: d['test_acc'] for m, d in self.results.items()}
            train_accs = {m: d['train_acc'] for m, d in self.results.items()}
            
            best_val = max(val_accs, key=val_accs.get)
            best_test = max(test_accs, key=test_accs.get)
            worst_val = min(val_accs.values())
            worst_test = min(test_accs.values())
            
            f.write(f"Validation Accuracy:\n")
            f.write(f"  Best:  {best_val} ({val_accs[best_val]:.2f}%)\n")
            f.write(f"  Worst: {min(val_accs, key=val_accs.get)} ({worst_val:.2f}%)\n")
            f.write(f"  Range: {val_accs[best_val] - worst_val:.2f}%\n\n")
            
            f.write(f"Test Accuracy:\n")
            f.write(f"  Best:  {best_test} ({test_accs[best_test]:.2f}%)\n")
            f.write(f"  Worst: {min(test_accs, key=test_accs.get)} ({worst_test:.2f}%)\n")
            f.write(f"  Range: {test_accs[best_test] - worst_test:.2f}%\n\n")
            
            # Overfitting analysis
            f.write(f"Overfitting Analysis (Train-Val Gap):\n")
            for method in self.results.keys():
                gap = train_accs[method] - val_accs[method]
                status = "High" if gap > 5 else "Moderate" if gap > 2 else "Low"
                f.write(f"  {method:<25}: {gap:>6.2f}% ({status})\n")
            
            f.write("\n\n")
            
            # Key insights
            f.write("-"*80 + "\n")
            f.write("KEY INSIGHTS & OBSERVATIONS\n")
            f.write("-"*80 + "\n\n")
            
            f.write("1. Comparison of Methods:\n")
            f.write("   - LM is direct knowledge transfer via soft targets\n")
            f.write("   - LS is indirect, regularizing the student's confidence\n")
            f.write("   - DKD is more nuanced, decomposing the knowledge transfer\n\n")
            
            f.write("2. Performance Trade-offs:\n")
            for method in self.results.keys():
                val_acc = val_accs[method]
                test_acc = test_accs[method]
                train_acc = train_accs[method]
                gap = train_acc - val_acc
                
                f.write(f"\n   {method}:\n")
                f.write(f"     - Validation:  {val_acc:.2f}%\n")
                f.write(f"     - Test:        {test_acc:.2f}%\n")
                f.write(f"     - Train:       {train_acc:.2f}%\n")
                f.write(f"     - Overfitting: {gap:.2f}%\n")
            
            f.write("\n\n3. Training Stability & Generalization:\n")
            f.write("   - Check if val accuracy aligns with test accuracy\n")
            f.write("   - Lower train-val gap indicates better regularization\n")
            f.write("   - Test accuracy closest to validation is most reliable\n\n")
            
            f.write("4. Recommendations:\n")
            f.write(f"   - Best for inference: {best_test} ({test_accs[best_test]:.2f}%)\n")
            f.write(f"   - Most stable: ")
            gaps = {m: train_accs[m] - val_accs[m] for m in self.results.keys()}
            most_stable = min(gaps, key=gaps.get)
            f.write(f"{most_stable} (gap: {gaps[most_stable]:.2f}%)\n")
            
            f.write("\n\n")
            f.write("-"*80 + "\n")
            f.write("END OF ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Analysis report saved to: {report_path}")


def main():
    """Main comparison script"""
    
    print("\n" + "="*80)
    print("LOADING PRE-TRAINED MODELS FOR COMPARISON")
    print("Part 1: Logit Matching Knowledge Distillation")
    print("Comparing: Logit Matching, Label Smoothing, Decoupled KD")
    print("="*80 + "\n")
    
    # Get checkpoint directory from command line or use default
    import sys
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else './checkpoints'
    
    if not os.path.exists(checkpoint_dir):
        print(f"✗ Error: Checkpoint directory not found: {checkpoint_dir}")
        print(f"  Please provide the correct path as argument:")
        print(f"  python comparison_load_checkpoints.py /path/to/checkpoints")
        sys.exit(1)
    
    print(f"Checkpoint directory: {os.path.abspath(checkpoint_dir)}\n")
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    train_loader, val_loader, test_loader, num_classes = get_cifar100_loaders(
        batch_size=256,
        num_workers=4
    )
    print(f"✓ Dataset loaded: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches\n")
    
    # Run comparison
    comparison = KDComparison(checkpoint_dir=checkpoint_dir)
    
    print("Loading and evaluating models...")
    comparison.load_and_evaluate(train_loader, val_loader, test_loader)
    
    if not comparison.results:
        print("\n✗ No models were loaded successfully.")
        print("  Please check that checkpoint files exist in the specified directory.")
        sys.exit(1)
    
    # Print and save results
    comparison.print_comparison_table()
    comparison.plot_comparison()
    comparison.save_results_json()
    comparison.generate_analysis_report()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("Results saved to ./results/ directory")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()