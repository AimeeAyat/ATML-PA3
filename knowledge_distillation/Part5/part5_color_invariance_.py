"""
Part 5: Testing Color Invariance with CRD
===========================================

Analyzes whether CRD can transfer robustness properties (color invariance)
without explicit training of the student.

Pipeline:
1. Fine-tune teacher with color jitter augmentations
2. Evaluate teacher on color-jittered validation set
3. Distill to student using CRD (normal augmentations)
4. Evaluate student on color-jittered validation set
5. Compare with other KD methods
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns


class ColorJitterTransform:
    """Color jitter augmentation with configurable intensity."""
    
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            ),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
    
    def __call__(self, img):
        return self.transform(img)


class NormalTransform:
    """Normal CIFAR-100 transform (no color jitter)."""
    
    def __call__(self, img):
        img_tensor = transforms.ToTensor()(img)
        return transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )(img_tensor)


def get_cifar100_loaders(transform_type='normal', batch_size=128, num_workers=6):
    """
    Get CIFAR-100 loaders with specified transform.
    
    Args:
        transform_type: 'normal' or 'color_jitter'
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        train_loader, val_loader, test_loader_jittered
    """
    
    if transform_type == 'color_jitter':
        transform = ColorJitterTransform()
    else:
        transform = NormalTransform()
    
    # Training set
    train_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        transform=transform,
        download=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation set
    val_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        transform=NormalTransform(),  # Normal for validation
        download=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Color-jittered validation set
    test_jittered_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        transform=ColorJitterTransform(),  # Color-jittered
        download=False
    )
    test_jittered_loader = DataLoader(
        test_jittered_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_jittered_loader


class ColorInvarianceAnalyzer:
    """Analyzes color invariance and KD effects."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
    
    def evaluate_model(self, model, val_loader, jittered_loader, model_name):
        """
        Evaluate model on both normal and color-jittered data.
        
        Args:
            model: Neural network
            val_loader: Normal validation loader
            jittered_loader: Color-jittered validation loader
            model_name: Name for results
            
        Returns:
            results: Dict with accuracies and metrics
        """
        model.eval()
        
        # Normal validation
        correct_normal = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct_normal += (predicted == labels).sum().item()
        
        acc_normal = correct_normal / total
        
        # Color-jittered validation
        correct_jittered = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in jittered_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct_jittered += (predicted == labels).sum().item()
        
        acc_jittered = correct_jittered / total
        
        # Compute invariance metrics
        drop = acc_normal - acc_jittered  # Drop in accuracy
        invariance_ratio = acc_jittered / (acc_normal + 1e-10)  # Robustness score
        
        results = {
            'accuracy_normal': acc_normal,
            'accuracy_jittered': acc_jittered,
            'accuracy_drop': drop,
            'invariance_ratio': invariance_ratio
        }
        
        self.results[model_name] = results
        
        return results
    
    def fine_tune_teacher(self, model, train_loader, val_loader, 
                         jittered_loader, num_epochs=10, 
                         learning_rate=0.01):
        """
        Fine-tune teacher with color jitter augmentations.
        
        Args:
            model: Teacher model
            train_loader: Training loader with color jitter
            val_loader: Validation loader (normal)
            jittered_loader: Validation loader (color-jittered)
            num_epochs: Number of epochs
            learning_rate: Learning rate
            
        Returns:
            best_model: Fine-tuned model
            history: Training history
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_acc_jittered = 0
        best_model = model
        history = {
            'train_loss': [],
            'val_acc_normal': [],
            'val_acc_jittered': [],
            'val_invariance': []
        }
        
        print(f"\nFine-tuning teacher with color jitter augmentations ({num_epochs} epochs):")
        print("="*70)
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                # Normal
                correct = 0
                total = 0
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc_normal = correct / total
                
                # Jittered
                correct = 0
                total = 0
                for images, labels in jittered_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc_jittered = correct / total
            
            history['val_acc_normal'].append(acc_normal)
            history['val_acc_jittered'].append(acc_jittered)
            invariance = acc_jittered / (acc_normal + 1e-10)
            history['val_invariance'].append(invariance)
            
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, "
                  f"Normal={acc_normal:.4f}, Jittered={acc_jittered:.4f}, "
                  f"Invariance={invariance:.4f}")
            
            # Save best model
            if acc_jittered > best_acc_jittered:
                best_acc_jittered = acc_jittered
                best_model = model
            
            scheduler.step()
        
        print("="*70)
        return best_model, history
    
    def knowledge_distillation_crd(self, teacher, student, train_loader, 
                                   val_loader, jittered_loader, 
                                   num_epochs=100, temperature=4.0, 
                                   alpha=0.5, learning_rate=0.1):
        """
        CRD knowledge distillation with optional contrastive learning.
        Simplified version focusing on KL divergence loss.
        
        Args:
            teacher: Teacher model
            student: Student model
            train_loader: Training loader (normal augmentations)
            val_loader: Validation loader (normal)
            jittered_loader: Validation loader (color-jittered)
            num_epochs: Number of epochs
            temperature: Softmax temperature
            alpha: Weight for KD loss (vs regular loss)
            learning_rate: Learning rate
            
        Returns:
            best_student: Trained student
            history: Training history
        """
        teacher = teacher.to(self.device)
        student = student.to(self.device)
        
        teacher.eval()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(student.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_acc_jittered = 0
        best_student = student
        history = {
            'train_loss': [],
            'val_acc_normal': [],
            'val_acc_jittered': [],
            'val_invariance': []
        }
        
        print(f"\nCRD Knowledge Distillation ({num_epochs} epochs):")
        print("="*70)
        
        for epoch in range(num_epochs):
            student.train()
            train_loss = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Student output
                student_logits = student(images)
                
                # Teacher output (no grad)
                with torch.no_grad():
                    teacher_logits = teacher(images)
                
                # KL divergence loss (main distillation)
                student_probs = F.log_softmax(student_logits / temperature, dim=1)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
                kd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
                
                # Regular CE loss
                ce_loss = criterion(student_logits, labels)
                
                # Combined loss
                total_loss = alpha * kd_loss + (1 - alpha) * ce_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            student.eval()
            with torch.no_grad():
                # Normal
                correct = 0
                total = 0
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = student(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc_normal = correct / total
                
                # Jittered
                correct = 0
                total = 0
                for images, labels in jittered_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = student(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc_jittered = correct / total
            
            history['val_acc_normal'].append(acc_normal)
            history['val_acc_jittered'].append(acc_jittered)
            invariance = acc_jittered / (acc_normal + 1e-10)
            history['val_invariance'].append(invariance)
            
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, "
                      f"Normal={acc_normal:.4f}, Jittered={acc_jittered:.4f}, "
                      f"Invariance={invariance:.4f}")
            
            if acc_jittered > best_acc_jittered:
                best_acc_jittered = acc_jittered
                best_student = student
            
            scheduler.step()
        
        print("="*70)
        return best_student, history
    
    def compare_methods(self, save_path='color_invariance_results.json'):
        """Export comparison results to JSON."""
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to {save_path}")
    
    def plot_comparison(self, save_dir='part5_plots'):
        """Create comparison visualizations."""
        Path(save_dir).mkdir(exist_ok=True)
        
        if not self.results:
            print("No results to plot")
            return
        
        # Extract data
        models = list(self.results.keys())
        acc_normal = [self.results[m]['accuracy_normal'] for m in models]
        acc_jittered = [self.results[m]['accuracy_jittered'] for m in models]
        drops = [self.results[m]['accuracy_drop'] for m in models]
        invariance = [self.results[m]['invariance_ratio'] for m in models]
        
        # Plot 1: Accuracy comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.arange(len(models))
        width = 0.35
        
        ax = axes[0]
        ax.bar(x - width/2, acc_normal, width, label='Normal', alpha=0.8)
        ax.bar(x + width/2, acc_jittered, width, label='Color-Jittered', alpha=0.8)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy: Normal vs Color-Jittered', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 2: Accuracy drop
        ax = axes[1]
        colors = ['red' if d > 0.1 else 'orange' if d > 0.05 else 'green' for d in drops]
        ax.bar(x, drops, color=colors, alpha=0.7)
        ax.set_ylabel('Accuracy Drop', fontsize=12)
        ax.set_title('Performance Drop on Color-Jittered Data', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/accuracy_comparison.png")
        plt.close()
        
        # Plot 3: Invariance ratio
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(x, invariance, alpha=0.7, color='steelblue')
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Perfect Invariance')
        ax.set_ylabel('Invariance Ratio (Jittered Acc / Normal Acc)', fontsize=12)
        ax.set_title('Color Invariance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Annotate
        for i, v in enumerate(invariance):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/invariance_ratio.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/invariance_ratio.png")
        plt.close()


if __name__ == "__main__":
    print("Part 5: Color Invariance Analysis with CRD")
    print("See example_part5_main.py for usage")