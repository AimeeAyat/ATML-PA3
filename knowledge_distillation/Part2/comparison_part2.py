"""
Part 2: Comparing Performance of State-of-the-Art Approaches
Compares: Logit Matching, Hints, CRD, and Independent Student
Generates comprehensive comparison table and visualizations
"""

import torch
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from Part1
from Part1.utils import get_device, get_cifar100_loaders, validate, vgg11
from Part1.logit_matching import LogitMatchingKD
from Part1.label_smoothing import LabelSmoothingKD
from Part1.decoupled_kd import DecoupledKD

# Import Part 2 methods
from Part2.independent_student import IndependentStudent
from Part2.hints_distillation import HintsDistillation
from Part2.contrastive_kd import ContrastiveRepDistillation


class Part2Comparison:
    """Compare all KD methods from Part 1 and Part 2"""
    
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.device = get_device()
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = checkpoint_dir
        
        # Create output directory
        os.makedirs('./checkpoints', exist_ok=True)
        os.makedirs('./results/part2', exist_ok=True)
    
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
    
    def load_model_from_checkpoint(self, checkpoint_path, model):
        """Load model weights from checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model_state_dict if checkpoint contains metadata
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"✓ Loaded model from: {checkpoint_path}")
        return True
    
    def load_and_evaluate_models(self, train_loader, val_loader, test_loader):
        """Load all models from checkpoints and evaluate them"""
        
        print("\n" + "="*80)
        print("LOADING PRE-TRAINED MODELS FROM CHECKPOINTS")
        print("="*80 + "\n")
        
        # 1. Load Independent Student
        print("[1/4] Loading Independent Student (SI)...")
        si_checkpoint = os.path.join(self.checkpoint_dir, 'si_student_best.pth')
        if os.path.exists(si_checkpoint):
            si_model = vgg11(num_classes=100).to(self.device)
            if self.load_model_from_checkpoint(si_checkpoint, si_model):
                si_val = self._evaluate_model(si_model, val_loader)
                si_test = self._evaluate_model(si_model, test_loader)
                si_train = self._evaluate_model(si_model, train_loader)
                
                self.results['Independent Student (SI)'] = {
                    'val_acc': si_val,
                    'test_acc': si_test,
                    'train_acc': si_train,
                    'train_accs': [],
                    'val_accs': [],
                    'train_losses': [],
                    'model': si_model,
                    'teacher': None
                }
                print(f"  Train: {si_train:.2f}%, Val: {si_val:.2f}%, Test: {si_test:.2f}%\n")
        else:
            print(f"✗ Checkpoint not found, skipping.\n")
        
        # 2. Load Logit Matching
        print("[2/4] Loading Logit Matching (LM)...")
        lm_checkpoint = os.path.join(self.checkpoint_dir, 'lm_student_best.pth')
        if os.path.exists(lm_checkpoint):
            lm_model = vgg11(num_classes=100).to(self.device)
            if self.load_model_from_checkpoint(lm_checkpoint, lm_model):
                lm_val = self._evaluate_model(lm_model, val_loader)
                lm_test = self._evaluate_model(lm_model, test_loader)
                lm_train = self._evaluate_model(lm_model, train_loader)
                
                self.results['Logit Matching (LM)'] = {
                    'val_acc': lm_val,
                    'test_acc': lm_test,
                    'train_acc': lm_train,
                    'train_accs': [],
                    'val_accs': [],
                    'train_losses': [],
                    'model': lm_model,
                    'teacher': None
                }
                print(f"  Train: {lm_train:.2f}%, Val: {lm_val:.2f}%, Test: {lm_test:.2f}%\n")
        else:
            print(f"✗ Checkpoint not found, skipping.\n")
        
        # 3. Load Hints-based Distillation
        print("[3/4] Loading Hints-based Distillation...")
        hints_checkpoint = os.path.join(self.checkpoint_dir, 'hints_student_best.pth')
        if os.path.exists(hints_checkpoint):
            hints_model = vgg11(num_classes=100).to(self.device)
            if self.load_model_from_checkpoint(hints_checkpoint, hints_model):
                hints_val = self._evaluate_model(hints_model, val_loader)
                hints_test = self._evaluate_model(hints_model, test_loader)
                hints_train = self._evaluate_model(hints_model, train_loader)
                
                self.results['Hints (FitNet)'] = {
                    'val_acc': hints_val,
                    'test_acc': hints_test,
                    'train_acc': hints_train,
                    'train_accs': [],
                    'val_accs': [],
                    'train_losses': [],
                    'model': hints_model,
                    'teacher': None
                }
                print(f"  Train: {hints_train:.2f}%, Val: {hints_val:.2f}%, Test: {hints_test:.2f}%\n")
        else:
            print(f"✗ Checkpoint not found, skipping.\n")
        
        # 4. Load CRD
        print("[4/4] Loading Contrastive Representation Distillation (CRD)...")
        crd_checkpoint = os.path.join(self.checkpoint_dir, 'crd_student_best.pth')
        if os.path.exists(crd_checkpoint):
            crd_model = vgg11(num_classes=100).to(self.device)
            if self.load_model_from_checkpoint(crd_checkpoint, crd_model):
                crd_val = self._evaluate_model(crd_model, val_loader)
                crd_test = self._evaluate_model(crd_model, test_loader)
                crd_train = self._evaluate_model(crd_model, train_loader)
                
                self.results['Contrastive (CRD)'] = {
                    'val_acc': crd_val,
                    'test_acc': crd_test,
                    'train_acc': crd_train,
                    'train_accs': [],
                    'val_accs': [],
                    'train_losses': [],
                    'model': crd_model,
                    'teacher': None
                }
                print(f"  Train: {crd_train:.2f}%, Val: {crd_val:.2f}%, Test: {crd_test:.2f}%\n")
        else:
            print(f"✗ Checkpoint not found, skipping.\n")
        
        print(f"✓ Loaded {len(self.results)}/4 models successfully\n")
    
    def train_hints_only(self, train_loader, val_loader, test_loader, num_epochs=30):
        """Train only Hints-based Distillation"""
        
        print("\n" + "="*80)
        print("TRAINING: HINTS-BASED DISTILLATION (FitNet)")
        print("="*80)
        print("\n[Training Hints model...]")
        
        hints = HintsDistillation(
            hint_layer=8, learning_rate=0.1, weight_decay=5e-4, alpha=0.5
        )
        hints_best_val = hints.train(train_loader, val_loader, num_epochs=num_epochs)
        hints_test = hints.evaluate(test_loader)
        hints_train = hints.get_train_acc()  # Get final training accuracy
        
        self.results['Hints (FitNet)'] = {
            'val_acc': hints_best_val,
            'test_acc': hints_test,
            'train_acc': hints_train if hints_train else hints_best_val,
            'train_accs': hints.train_accs,
            'val_accs': hints.val_accs,
            'train_losses': hints.train_losses,
            'model': hints.get_student_model(),
            'teacher': hints.get_teacher_model()
        }
        
        print(f"\n✓ Hints training complete!")
        print(f"  Best Val: {hints_best_val:.2f}%, Test: {hints_test:.2f}%\n")
    
    def run_all_experiments(self, train_loader, val_loader, test_loader, num_epochs=30):
        """Run all four methods: SI, LM, Hints, CRD"""
        
        # 1. Independent Student (Baseline)
        print("\n" + "="*80)
        print("METHOD 1/4: INDEPENDENT STUDENT (SI) - BASELINE")
        print("="*80)
        print("\n[1/4] Running Independent Student...")
        
        si = IndependentStudent(learning_rate=0.1, weight_decay=5e-4)
        si_best_val = si.train(train_loader, val_loader, num_epochs=num_epochs)
        si_test = si.evaluate(test_loader)
        self.results['Independent Student (SI)'] = {
            'val_acc': si_best_val,
            'test_acc': si_test,
            'train_accs': si.train_accs,
            'val_accs': si.val_accs,
            'train_losses': si.train_losses,
            'model': si.get_student_model(),
            'teacher': si.get_teacher_model()
        }
        
        # 2. Logit Matching (Part 1)
        print("\n" + "="*80)
        print("METHOD 2/4: LOGIT MATCHING (LM) - FROM PART 1")
        print("="*80)
        print("\n[2/4] Running Logit Matching...")
        
        lm = LogitMatchingKD(temperature=4.0, alpha=0.5, learning_rate=0.1, weight_decay=5e-4)
        lm_best_val = lm.train(train_loader, val_loader, num_epochs=num_epochs)
        lm_test = lm.evaluate(test_loader)
        self.results['Logit Matching (LM)'] = {
            'val_acc': lm_best_val,
            'test_acc': lm_test,
            'train_accs': lm.train_accs,
            'val_accs': lm.val_accs,
            'train_losses': lm.train_losses,
            'model': lm.get_student_model(),
            'teacher': lm.get_teacher_model()
        }
        
        # 3. Hints-based Distillation
        print("\n" + "="*80)
        print("METHOD 3/4: HINTS-BASED DISTILLATION (FitNet)")
        print("="*80)
        print("\n[3/4] Running Hints-based Distillation...")
        
        hints = HintsDistillation(
            hint_layer=8, learning_rate=0.1, weight_decay=5e-4, alpha=0.5
        )
        hints_best_val = hints.train(train_loader, val_loader, num_epochs=num_epochs)
        hints_test = hints.evaluate(test_loader)
        self.results['Hints (FitNet)'] = {
            'val_acc': hints_best_val,
            'test_acc': hints_test,
            'train_accs': hints.train_accs,
            'val_accs': hints.val_accs,
            'train_losses': hints.train_losses,
            'model': hints.get_student_model(),
            'teacher': hints.get_teacher_model()
        }
        
        # 4. Contrastive Representation Distillation
        print("\n" + "="*80)
        print("METHOD 4/4: CONTRASTIVE REPRESENTATION DISTILLATION (CRD)")
        print("="*80)
        print("\n[4/4] Running Contrastive Representation Distillation...")
        
        crd = ContrastiveRepDistillation(
            temperature=0.07, learning_rate=0.1, weight_decay=5e-4, alpha=0.5, feat_dim=128
        )
        crd_best_val = crd.train(train_loader, val_loader, num_epochs=num_epochs)
        crd_test = crd.evaluate(test_loader)
        self.results['Contrastive (CRD)'] = {
            'val_acc': crd_best_val,
            'test_acc': crd_test,
            'train_accs': crd.train_accs,
            'val_accs': crd.val_accs,
            'train_losses': crd.train_losses,
            'model': crd.get_student_model(),
            'teacher': crd.get_teacher_model()
        }
    
    def print_comparison_table(self):
        """Print comprehensive comparison table"""
        
        print("\n" + "="*90)
        print("PART 2: COMPREHENSIVE COMPARISON - ALL METHODS")
        print("="*90)
        print()
        
        # Sort by validation accuracy (descending)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['val_acc'],
            reverse=True
        )
        
        print(f"{'Rank':<6} {'Method':<30} {'Val Acc':<12} {'Test Acc':<12} {'Improvement*':<15}")
        print("-" * 90)
        
        # Get baseline (SI)
        si_val_acc = self.results['Independent Student (SI)']['val_acc']
        
        for rank, (method, data) in enumerate(sorted_results, 1):
            val_acc = data['val_acc']
            test_acc = data['test_acc']
            improvement = val_acc - si_val_acc
            
            # Add asterisk for SI
            method_name = method if method != 'Independent Student (SI)' else method + " (BASELINE)"
            
            print(f"{rank:<6} {method_name:<30} {val_acc:>6.2f}%        {test_acc:>6.2f}%        "
                  f"{improvement:>+6.2f}%")
        
        print("-" * 90)
        print("* Improvement relative to Independent Student (SI) baseline")
        print()
        
        # Summary statistics
        val_accs = [d['val_acc'] for d in self.results.values()]
        test_accs = [d['test_acc'] for d in self.results.values()]
        
        print(f"Best Validation Accuracy:  {max(val_accs):.2f}%")
        print(f"Worst Validation Accuracy: {min(val_accs):.2f}%")
        print(f"Average Validation Accuracy: {np.mean(val_accs):.2f}%")
        print(f"Range: {max(val_accs) - min(val_accs):.2f}%")
        print()
    
    def plot_comparison(self):
        """Generate comprehensive comparison plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Check if any method has training history
        has_training_history = any(len(data.get('train_accs', [])) > 0 for data in self.results.values())
        
        # Plot 1: Training Accuracy
        ax = axes[0, 0]
        if has_training_history:
            for method, data in self.results.items():
                if len(data.get('train_accs', [])) > 0:
                    ax.plot(data['train_accs'], label=method, linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title('Training Accuracy Comparison (All Methods)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No training history available\n(Models loaded from checkpoints)',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Training Accuracy (Not Available)', fontsize=14, fontweight='bold')
        
        # Plot 2: Validation Accuracy
        ax = axes[0, 1]
        if has_training_history:
            for method, data in self.results.items():
                if len(data.get('val_accs', [])) > 0:
                    ax.plot(data['val_accs'], label=method, linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title('Validation Accuracy Comparison (All Methods)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No training history available\n(Models loaded from checkpoints)',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Validation Accuracy (Not Available)', fontsize=14, fontweight='bold')
        
        # Plot 3: Training Loss
        ax = axes[1, 0]
        if has_training_history:
            for method, data in self.results.items():
                if len(data.get('train_losses', [])) > 0:
                    ax.plot(data['train_losses'], label=method, linewidth=2, marker='s', markersize=3)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training Loss Comparison (All Methods)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No training history available\n(Models loaded from checkpoints)',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Training Loss (Not Available)', fontsize=14, fontweight='bold')
        
        # Plot 4: Final Accuracies (Bar Chart)
        ax = axes[1, 1]
        methods = list(self.results.keys())
        val_accs = [self.results[m]['val_acc'] for m in methods]
        test_accs = [self.results[m]['test_acc'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, val_accs, width, label='Validation', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, test_accs, width, label='Test', alpha=0.8, color='darkorange')
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Final Accuracy Comparison (Bar Chart)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        fig_path = f'./results/part2/part2_comparison_{self.timestamp}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to {fig_path}")
        
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
        
        json_path = f'./results/part2/part2_comparison_{self.timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✓ Results saved to {json_path}")
    
    def generate_analysis_report(self):
        """Generate detailed analysis report"""
        
        report_path = f'./results/part2/part2_analysis_{self.timestamp}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*90 + "\n")
            f.write("PART 2: COMPREHENSIVE ANALYSIS - STATE-OF-THE-ART APPROACHES\n")
            f.write("="*90 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"Epochs per method: 30\n\n")
            
            # Summary table
            f.write("-"*90 + "\n")
            f.write("RESULTS SUMMARY\n")
            f.write("-"*90 + "\n\n")
            
            # Sort by validation accuracy
            sorted_results = sorted(
                self.results.items(),
                key=lambda x: x[1]['val_acc'],
                reverse=True
            )
            
            si_val = self.results['Independent Student (SI)']['val_acc']
            
            f.write(f"{'Rank':<6} {'Method':<30} {'Val Acc':<12} {'Test Acc':<12} {'Improvement':<12}\n")
            f.write("-"*90 + "\n")
            
            for rank, (method, data) in enumerate(sorted_results, 1):
                val_acc = data['val_acc']
                test_acc = data['test_acc']
                improvement = val_acc - si_val
                
                f.write(f"{rank:<6} {method:<30} {val_acc:>6.2f}%        {test_acc:>6.2f}%        "
                       f"{improvement:>+6.2f}%\n")
            
            f.write("\n\n")
            
            # Detailed method descriptions
            f.write("-"*90 + "\n")
            f.write("METHOD DESCRIPTIONS\n")
            f.write("-"*90 + "\n\n")
            
            f.write("1. INDEPENDENT STUDENT (SI) - BASELINE\n")
            f.write("   - Standard supervised learning\n")
            f.write("   - No teacher, no distillation\n")
            f.write("   - Only cross-entropy loss with ground truth\n")
            f.write("   - Serves as baseline for measuring KD improvements\n\n")
            
            f.write("2. LOGIT MATCHING (LM) - PART 1\n")
            f.write("   - Student matches teacher's softened logits (soft targets)\n")
            f.write("   - Uses temperature-scaled softmax (T=4.0)\n")
            f.write("   - Loss = α*KL(soft_teacher || soft_student) + (1-α)*CE\n")
            f.write("   - Well-established, direct knowledge transfer\n\n")
            
            f.write("3. HINTS-BASED DISTILLATION (FitNet)\n")
            f.write("   - Transfers knowledge via intermediate feature maps\n")
            f.write("   - Aligns student and teacher hidden layer activations\n")
            f.write("   - Loss = α*MSE(adapted_student_feat || teacher_feat) + (1-α)*CE\n")
            f.write("   - Focuses on learning useful representations\n\n")
            
            f.write("4. CONTRASTIVE REPRESENTATION DISTILLATION (CRD)\n")
            f.write("   - Uses contrastive learning to align representations\n")
            f.write("   - Student learns to match teacher's representation space\n")
            f.write("   - Loss = α*Contrastive + (1-α)*CE\n")
            f.write("   - More sophisticated, learns semantic structure\n\n")
            
            # Key insights
            f.write("-"*90 + "\n")
            f.write("KEY INSIGHTS & ANALYSIS\n")
            f.write("-"*90 + "\n\n")
            
            val_accs = [d['val_acc'] for d in self.results.values()]
            best_method = sorted_results[0][0]
            best_acc = sorted_results[0][1]['val_acc']
            worst_method = sorted_results[-1][0]
            worst_acc = sorted_results[-1][1]['val_acc']
            
            f.write(f"Best Method: {best_method} ({best_acc:.2f}%)\n")
            f.write(f"Worst Method: {worst_method} ({worst_acc:.2f}%)\n")
            f.write(f"Performance Range: {worst_acc:.2f}% - {best_acc:.2f}% (Δ {best_acc - worst_acc:.2f}%)\n\n")
            
            f.write("Improvements over Baseline (SI):\n")
            for method, data in sorted(self.results.items(), key=lambda x: x[1]['val_acc'] - si_val, reverse=True):
                if method != 'Independent Student (SI)':
                    improvement = data['val_acc'] - si_val
                    f.write(f"  {method:<35}: +{improvement:.2f}%\n")
            
            f.write("\n\n")
            
            # Performance characteristics
            f.write("-"*90 + "\n")
            f.write("PERFORMANCE CHARACTERISTICS\n")
            f.write("-"*90 + "\n\n")
            
            f.write("Training Stability (lower variance = more stable):\n")
            for method, data in self.results.items():
                val_accs_list = data['val_accs']
                variance = np.var(val_accs_list)
                f.write(f"  {method:<35}: Variance = {variance:.4f}\n")
            
            f.write("\n")
            
            f.write("Convergence Speed (epochs to reach 60% accuracy):\n")
            for method, data in self.results.items():
                val_accs_list = data['val_accs']
                epochs_to_60 = next((i for i, acc in enumerate(val_accs_list) if acc >= 60.0), len(val_accs_list))
                f.write(f"  {method:<35}: Epoch {epochs_to_60} / 30\n")
            
            f.write("\n\n")
            
            # Recommendations
            f.write("-"*90 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-"*90 + "\n\n")
            
            f.write("1. For production: Use the best performing method that offers good trade-off\n")
            f.write("   between accuracy and computational cost.\n\n")
            
            f.write("2. For accuracy: Focus on CRD or Hints if available, as they often\n")
            f.write("   provide better semantic understanding.\n\n")
            
            f.write("3. For simplicity: Logit Matching offers good accuracy with simpler\n")
            f.write("   implementation compared to CRD/Hints.\n\n")
            
            f.write("4. For resource-constrained: Independent Student may be preferred if\n")
            f.write("   teacher is unavailable or too large.\n\n")
            
            f.write("="*90 + "\n")
        
        print(f"✓ Analysis report saved to {report_path}")


def main_load_checkpoints():
    """Load all models from checkpoints and compare"""
    
    print("\n" + "="*90)
    print("PART 2: LOADING ALL MODELS FROM CHECKPOINTS AND COMPARING")
    print("Methods: Independent Student (SI), Logit Matching (LM), Hints, CRD")
    print("="*90 + "\n")
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    train_loader, val_loader, test_loader, num_classes = get_cifar100_loaders(
        batch_size=128,
        num_workers=4
    )
    print(f"✓ Dataset loaded\n")
    
    # Run comparison
    comparison = Part2Comparison()
    
    # Load all models from checkpoints
    comparison.load_and_evaluate_models(train_loader, val_loader, test_loader)
    
    # Generate comparison results
    if len(comparison.results) >= 2:
        print("\n" + "="*90)
        print("GENERATING COMPARISON RESULTS")
        print("="*90)
        comparison.print_comparison_table()
        comparison.plot_comparison()
        comparison.save_results_json()
        comparison.generate_analysis_report()
        
        print("\n" + "="*90)
        print("PART 2 COMPARISON COMPLETE!")
        print("Results saved to ./results/part2/")
        print("="*90 + "\n")
    else:
        print("\n✗ Not enough models loaded for comparison.")
        print("  Please ensure checkpoint files exist in ./checkpoints/")
        print("  Expected files: si_student_best.pth, lm_student_best.pth,")
        print("                  hints_student_best.pth, crd_student_best.pth\n")


def main_train_hints_only():
    """Train only Hints model and load others from checkpoints"""
    
    print("\n" + "="*90)
    print("PART 2: TRAINING HINTS AND LOADING OTHER MODELS FROM CHECKPOINTS")
    print("Methods: Independent Student (SI), Logit Matching (LM), Hints, CRD")
    print("="*90 + "\n")
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    train_loader, val_loader, test_loader, num_classes = get_cifar100_loaders(
        batch_size=128,
        num_workers=4
    )
    print(f"✓ Dataset loaded\n")
    
    # Run comparison
    comparison = Part2Comparison()
    
    # Step 1: Train Hints-based Distillation
    print("\n" + "="*90)
    print("STEP 1: TRAINING HINTS-BASED DISTILLATION")
    print("="*90)
    comparison.train_hints_only(train_loader, val_loader, test_loader, num_epochs=30)
    
    # Step 2: Load all models from checkpoints (including newly trained Hints)
    print("\n" + "="*90)
    print("STEP 2: LOADING ALL MODELS FROM CHECKPOINTS")
    print("="*90)
    comparison.load_and_evaluate_models(train_loader, val_loader, test_loader)
    
    # Step 3: Generate comparison results
    if len(comparison.results) >= 2:
        print("\n" + "="*90)
        print("STEP 3: GENERATING COMPARISON RESULTS")
        print("="*90)
        comparison.print_comparison_table()
        comparison.plot_comparison()
        comparison.save_results_json()
        comparison.generate_analysis_report()
        
        print("\n" + "="*90)
        print("PART 2 COMPARISON COMPLETE!")
        print("Results saved to ./results/part2/")
        print("="*90 + "\n")
    else:
        print("\n✗ Not enough models loaded for comparison.")
        print("  Please ensure checkpoint files exist in ./checkpoints/\n")


def main():
    """Main comparison script - train all from scratch"""
    
    print("\n" + "="*90)
    print("PART 2: COMPARING PERFORMANCE OF STATE-OF-THE-ART APPROACHES")
    print("Methods: Independent Student (SI), Logit Matching (LM), Hints, CRD")
    print("="*90 + "\n")
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    train_loader, val_loader, test_loader, num_classes = get_cifar100_loaders(
        batch_size=128,
        num_workers=4
    )
    
    # Run comparison
    comparison = Part2Comparison()
    
    print("\nStarting experiments (4 methods × 30 epochs)...")
    print("Estimated time: ~3-4 hours\n")
    
    comparison.run_all_experiments(
        train_loader, val_loader, test_loader,
        num_epochs=30
    )
    
    # Print and save results
    comparison.print_comparison_table()
    comparison.plot_comparison()
    comparison.save_results_json()
    comparison.generate_analysis_report()
    
    print("\n" + "="*90)
    print("PART 2 COMPARISON COMPLETE!")
    print("Results saved to ./results/part2/")
    print("="*90 + "\n")


if __name__ == "__main__":
    # Use main_load_checkpoints() to load all models from checkpoints (NO TRAINING)
    # Use main_train_hints_only() to train only Hints and load others from checkpoints
    # Use main() to train all models from scratch
    main_load_checkpoints()  # Changed to load all checkpoints by default