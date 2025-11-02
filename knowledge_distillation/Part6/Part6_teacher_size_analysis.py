"""
Part 6: Testing the Efficacy of a Larger Teacher
===================================================

Analyzes whether teacher model size affects student performance in KD.

Experiment:
1. Train VGG-11 student with VGG-16 teacher (baseline)
2. Train VGG-11 student with VGG-19 teacher (larger)
3. Compare: accuracy, training dynamics, convergence
4. Analyze: size effect on KD efficacy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class VGG11(nn.Module):
    """VGG-11 Student Model"""
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    """VGG-16 Baseline Teacher"""
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class VGG19(nn.Module):
    """VGG-19 Larger Teacher"""
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# DATA LOADING
# ============================================================================

def get_cifar100_loaders(batch_size=128, num_workers=4):
    """Get CIFAR-100 train/val loaders"""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    train_dataset = datasets.CIFAR100(
        root='/tmp/cifar100',
        train=True,
        transform=transform_train,
        download=True
    )
    
    val_dataset = datasets.CIFAR100(
        root='/tmp/cifar100',
        train=False,
        transform=transform_test,
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# KNOWLEDGE DISTILLATION (LOGIT MATCHING)
# ============================================================================

class LogitMatchingTrainer:
    """Logit Matching KD trainer"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def train_with_kd(self, teacher, student, train_loader, val_loader,
                     num_epochs=100, temperature=4.0, alpha=0.5,
                     learning_rate=0.1, teacher_name='Teacher'):
        """
        Train student with logit matching KD.
        
        Args:
            teacher: Teacher model (eval mode)
            student: Student model (trainable)
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs
            temperature: Softmax temperature
            alpha: Weight of KD loss (vs CE loss)
            learning_rate: Learning rate
            teacher_name: Name for logging
            
        Returns:
            best_student: Best trained student
            history: Training history
        """
        
        teacher = teacher.to(self.device)
        student = student.to(self.device)
        teacher.eval()
        
        criterion_ce = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            student.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_acc = 0
        best_student = student
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
        
        print(f"\n{'='*80}")
        print(f"Training Student with {teacher_name} (LM-KD)")
        print(f"{'='*80}")
        print(f"Temperature: {temperature}, Alpha: {alpha}")
        print(f"{'Epoch':<8} {'Train Loss':<15} {'Train Acc':<15} {'Val Acc':<15} {'LR':<12}")
        print(f"{'-'*80}")
        
        for epoch in range(num_epochs):
            # Training
            student.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",
                                      leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Student output
                student_logits = student(images)
                
                # Teacher output (no grad)
                with torch.no_grad():
                    teacher_logits = teacher(images)
                
                # KL divergence loss (logit matching)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                
                # CE loss
                ce_loss = criterion_ce(student_logits, labels)
                
                # Combined loss
                loss = alpha * kl_loss + (1 - alpha) * ce_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(student_logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Validation
            student.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = student(images)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = val_correct / val_total
            history['val_acc'].append(val_acc)
            
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"{epoch+1:<8} {train_loss:<15.4f} {train_acc:<15.4f} "
                      f"{val_acc:<15.4f} {optimizer.param_groups[0]['lr']:<12.2e}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_student = student
            
            scheduler.step()
        
        print(f"{'-'*80}")
        print(f"Best Validation Accuracy: {best_acc:.4f}\n")
        
        return best_student, history


# ============================================================================
# ANALYSIS
# ============================================================================

class TeacherSizeAnalyzer:
    """Analyzes effect of teacher size on KD"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        self.histories = {}
    
    def evaluate_model(self, model, val_loader, model_name):
        """Evaluate model on validation set"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def run_analysis(self, teacher16_checkpoint=None, teacher19_checkpoint=None,
                    quick_mode=False, skip_training=False):
        """
        Complete analysis pipeline.
        
        Args:
            teacher16_checkpoint: Path to VGG-16 checkpoint (optional)
            teacher19_checkpoint: Path to VGG-19 checkpoint (optional)
            quick_mode: Fewer epochs for quick testing
            skip_training: Use pre-trained models only
        """
        
        print("="*80)
        print("PART 6: TEACHER SIZE EFFECT ON KD EFFICACY")
        print("="*80)
        
        # Load data
        train_loader, val_loader = get_cifar100_loaders(batch_size=128)
        
        # ====================================================================
        # STAGE 1: LOAD/TRAIN TEACHERS
        # ====================================================================
        print("\n" + "="*80)
        print("STAGE 1: PREPARE TEACHERS")
        print("="*80)
        
        # VGG-16 teacher
        teacher16 = VGG16(num_classes=100).to(self.device)
        n_params_16 = count_parameters(teacher16)
        print(f"\nVGG-16: {n_params_16:,} parameters")
        
        if teacher16_checkpoint and Path(teacher16_checkpoint).exists():
            print(f"Loading VGG-16 from checkpoint...")
            checkpoint = torch.load(teacher16_checkpoint, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                teacher16.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                teacher16.load_state_dict(checkpoint, strict=False)
            acc16 = self.evaluate_model(teacher16, val_loader, "VGG-16")
            print(f"VGG-16 Accuracy: {acc16:.4f}")
            self.results['VGG-16'] = {'accuracy': acc16, 'params': n_params_16}
        else:
            print("No VGG-16 checkpoint provided")
        
        # VGG-19 teacher
        teacher19 = VGG19(num_classes=100).to(self.device)
        n_params_19 = count_parameters(teacher19)
        print(f"\nVGG-19: {n_params_19:,} parameters")
        
        if teacher19_checkpoint and Path(teacher19_checkpoint).exists():
            print(f"Loading VGG-19 from checkpoint...")
            checkpoint = torch.load(teacher19_checkpoint, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                teacher19.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                teacher19.load_state_dict(checkpoint, strict=False)
            acc19 = self.evaluate_model(teacher19, val_loader, "VGG-19")
            print(f"VGG-19 Accuracy: {acc19:.4f}")
            self.results['VGG-19'] = {'accuracy': acc19, 'params': n_params_19}
        else:
            print("No VGG-19 checkpoint provided")
        
        # ====================================================================
        # STAGE 2: DISTILL WITH VGG-16
        # ====================================================================
        print("\n" + "="*80)
        print("STAGE 2: DISTILL STUDENT WITH VGG-16 TEACHER")
        print("="*80)
        
        student16 = VGG11(num_classes=100)
        n_params_student = count_parameters(student16)
        print(f"VGG-11 Student: {n_params_student:,} parameters")
        
        trainer = LogitMatchingTrainer(device=self.device)
        num_epochs = 5 if quick_mode else 100
        
        student16, history16 = trainer.train_with_kd(
            teacher16, student16, train_loader, val_loader,
            num_epochs=num_epochs,
            temperature=4.0,
            alpha=0.5,
            learning_rate=0.1,
            teacher_name="VGG-16"
        )
        
        acc_student16 = self.evaluate_model(student16, val_loader, "Student (VGG-16)")
        self.results['Student + VGG-16'] = {
            'accuracy': acc_student16,
            'params': n_params_student,
            'teacher_params': n_params_16,
            'improvement': acc_student16 - self.results.get('VGG-16', {}).get('accuracy', 0)
        }
        self.histories['Student + VGG-16'] = history16
        
        print(f"Student + VGG-16 Accuracy: {acc_student16:.4f}")
        
        # ====================================================================
        # STAGE 3: DISTILL WITH VGG-19
        # ====================================================================
        print("\n" + "="*80)
        print("STAGE 3: DISTILL STUDENT WITH VGG-19 TEACHER")
        print("="*80)
        
        student19 = VGG11(num_classes=100)
        print(f"VGG-11 Student: {n_params_student:,} parameters")
        
        student19, history19 = trainer.train_with_kd(
            teacher19, student19, train_loader, val_loader,
            num_epochs=num_epochs,
            temperature=4.0,
            alpha=0.5,
            learning_rate=0.1,
            teacher_name="VGG-19"
        )
        
        acc_student19 = self.evaluate_model(student19, val_loader, "Student (VGG-19)")
        self.results['Student + VGG-19'] = {
            'accuracy': acc_student19,
            'params': n_params_student,
            'teacher_params': n_params_19,
            'improvement': acc_student19 - self.results.get('VGG-19', {}).get('accuracy', 0)
        }
        self.histories['Student + VGG-19'] = history19
        
        print(f"Student + VGG-19 Accuracy: {acc_student19:.4f}")
        
        # ====================================================================
        # STAGE 4: ANALYSIS
        # ====================================================================
        print("\n" + "="*80)
        print("STAGE 4: COMPARATIVE ANALYSIS")
        print("="*80)
        
        self._print_comparison()
        self._analyze_teacher_effect()
        
        return self.results, self.histories
    
    def _print_comparison(self):
        """Print comprehensive comparison table"""
        
        print(f"\n{'Model':<25} {'Accuracy':<12} {'Parameters':<15} {'Improvement':<12}")
        print("-"*70)
        
        for model_name, metrics in sorted(self.results.items()):
            acc = metrics['accuracy']
            params = metrics['params']
            improvement = metrics.get('improvement', 0)
            
            if 'Student' in model_name:
                print(f"{model_name:<25} {acc:<12.4f} {params:<15,} {improvement:<12.4f}")
            else:
                print(f"{model_name:<25} {acc:<12.4f} {params:<15,}")
    
    def _analyze_teacher_effect(self):
        """Analyze teacher size effects"""
        
        acc_s16 = self.results.get('Student + VGG-16', {}).get('accuracy', 0)
        acc_s19 = self.results.get('Student + VGG-19', {}).get('accuracy', 0)
        params_16 = self.results.get('VGG-16', {}).get('params', 0)
        params_19 = self.results.get('VGG-19', {}).get('params', 0)
        
        diff_acc = acc_s19 - acc_s16
        diff_params = params_19 - params_16
        param_increase_pct = (diff_params / params_16 * 100) if params_16 > 0 else 0
        
        print(f"\nTeacher Size Effect:")
        print(f"  VGG-16 → VGG-19 parameter increase: {diff_params:,} ({param_increase_pct:.1f}%)")
        print(f"  Student accuracy change: {diff_acc:+.4f}")
        
        if abs(diff_acc) < 0.01:
            print(f"  → Conclusion: Marginal/No improvement with larger teacher")
        elif diff_acc > 0.02:
            print(f"  → Conclusion: Significant improvement with larger teacher")
        else:
            print(f"  → Conclusion: Modest improvement with larger teacher")
    
    def save_results(self, filepath='teacher_size_results.json'):
        """Export results to JSON"""
        results_export = {}
        for key, val in self.results.items():
            results_export[key] = {
                'accuracy': float(val['accuracy']),
                'parameters': int(val['params']),
                'teacher_parameters': int(val.get('teacher_params', 0)),
                'improvement': float(val.get('improvement', 0))
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_export, f, indent=2)
        print(f"\n✓ Results saved to {filepath}")
    
    def plot_results(self, save_dir='part6_plots'):
        """Create comparison visualizations"""
        Path(save_dir).mkdir(exist_ok=True)
        
        if not self.results or not self.histories:
            print("No results to plot")
            return
        
        # Plot 1: Accuracy comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = []
        accuracies = []
        colors = []
        
        for model_name in sorted(self.results.keys()):
            models.append(model_name.replace('Student + ', ''))
            accuracies.append(self.results[model_name]['accuracy'])
            
            if 'VGG-16' in model_name and 'Student' not in model_name:
                colors.append('#2E86AB')  # Blue
            elif 'VGG-19' in model_name and 'Student' not in model_name:
                colors.append('#A23B72')  # Purple
            elif 'VGG-16' in model_name:
                colors.append('#06A77D')  # Green
            else:
                colors.append('#F18F01')  # Orange
        
        bars = ax.bar(range(len(models)), accuracies, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Annotate
        for i, v in enumerate(accuracies):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/accuracy_comparison.png")
        plt.close()
        
        # Plot 2: Training curves
        if len(self.histories) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            for label, history in self.histories.items():
                axes[0].plot(history['train_acc'], label=label, linewidth=2)
                axes[1].plot(history['val_acc'], label=label, linewidth=2)
            
            axes[0].set_xlabel('Epoch', fontsize=11)
            axes[0].set_ylabel('Accuracy', fontsize=11)
            axes[0].set_title('Training Accuracy', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            axes[1].set_xlabel('Epoch', fontsize=11)
            axes[1].set_ylabel('Accuracy', fontsize=11)
            axes[1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_dir}/training_curves.png")
            plt.close()
        
        # Plot 3: Parameter efficiency
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models_params = []
        accuracies_params = []
        params_list = []
        
        for model_name in sorted(self.results.keys()):
            models_params.append(model_name.replace('Student + ', ''))
            accuracies_params.append(self.results[model_name]['accuracy'])
            params_list.append(self.results[model_name]['params'])
        
        scatter = ax.scatter(params_list, accuracies_params, s=300, alpha=0.6, c=range(len(models_params)), cmap='viridis')
        
        for i, txt in enumerate(models_params):
            ax.annotate(txt, (params_list[i], accuracies_params[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Model Parameters', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy vs Model Size', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/parameter_efficiency.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/parameter_efficiency.png")
        plt.close()


if __name__ == "__main__":
    print("Part 6: Teacher Size Effect Analysis")
    print("See example_part6_main.py for usage")