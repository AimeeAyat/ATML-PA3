"""
Independent Student (SI) - Baseline
Student model trained without any knowledge distillation.
This serves as baseline to measure improvement from KD methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # When imported as part of package
    from Part1.utils import (
        get_device, get_cifar100_loaders, vgg11, vgg16,
        validate, save_checkpoint, accuracy
    )
except ImportError:
    # When run as standalone script
    from .utils import (
        get_device, get_cifar100_loaders, vgg11, vgg16,
        validate, save_checkpoint, accuracy
    )


class IndependentStudent:
    """Independent Student (SI) - Baseline without distillation"""
    
    def __init__(self, learning_rate=0.1, weight_decay=5e-4):
        """
        Initialize Independent Student
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for L2 regularization
        """
        self.device = get_device()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize model
        print("\n" + "="*70)
        print("INDEPENDENT STUDENT (SI) - BASELINE")
        print("="*70)
        
        print("\nLoading Student (VGG-11, no teacher)...")
        self.student = vgg11(num_classes=100).to(self.device)
        
        # We still load teacher for consistency (but won't use it)
        self.teacher = vgg16(num_classes=100).to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Loss function - standard cross entropy
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Optimizer and scheduler
        self.optimizer = optim.SGD(
            self.student.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        # For storing metrics
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.student.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass through student ONLY
            student_logits = self.student(images)
            
            # Standard cross-entropy loss (no distillation)
            loss = self.ce_loss(student_logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            # Progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                acc = 100 * total_correct / total_samples
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100 * total_correct / total_samples
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        
        return avg_loss, avg_acc
    
    def train(self, train_loader, val_loader, num_epochs=30):
        """Train student without distillation"""
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        print(f"\nTraining parameters:")
        print(f"  Method: Baseline (no distillation)")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs: {num_epochs}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch [{epoch}/{num_epochs}]")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_acc = validate(self.student, val_loader, self.device)
            
            self.val_accs.append(val_acc)
            
            scheduler.step()
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Acc: {val_acc:.2f}%\n")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_checkpoint(
                    self.student,
                    self.optimizer,
                    epoch,
                    val_acc,
                    f'./checkpoints/si_student_best.pth'
                )
        
        return self.best_val_acc
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.student.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.student(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * total_correct / total_samples
        return test_acc
    
    def get_student_model(self):
        """Return the trained student model"""
        return self.student
    
    def get_teacher_model(self):
        """Return the teacher model (unused)"""
        return self.teacher


def main():
    """Main training script"""
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = get_cifar100_loaders(
        batch_size=128,
        num_workers=4
    )
    
    # Initialize Independent Student
    si = IndependentStudent(
        learning_rate=0.1,
        weight_decay=5e-4
    )
    
    # Train
    best_val_acc = si.train(train_loader, val_loader, num_epochs=30)
    
    # Evaluate
    test_acc = si.evaluate(test_loader)
    
    print("\n" + "="*70)
    print("RESULTS - INDEPENDENT STUDENT (SI)")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()