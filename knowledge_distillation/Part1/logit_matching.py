"""
Basic Logit Matching (LM) Knowledge Distillation
The student learns to match the softened output logits of the teacher.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from .utils import (
    get_device, get_cifar100_loaders, vgg11, vgg16,
    validate, save_checkpoint, load_checkpoint, accuracy, KLDivergenceLoss
)


class LogitMatchingKD:
    """Basic Logit Matching Knowledge Distillation"""
    
    def __init__(self, temperature=4.0, alpha=0.5, learning_rate=0.1, weight_decay=5e-4):
        """
        Initialize Logit Matching KD
        
        Args:
            temperature: Temperature for softening probabilities
            alpha: Weight for distillation loss (1-alpha for CE loss)
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.device = get_device()
        self.temperature = temperature
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize models
        print("\n" + "="*70)
        print("BASIC LOGIT MATCHING KNOWLEDGE DISTILLATION")
        print("="*70)
        
        print("\nLoading Teacher (VGG-16)...")
        self.teacher = vgg16(num_classes=100).to(self.device)
        self._load_pretrained_teacher()
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        print("Loading Student (VGG-11)...")
        self.student = vgg11(num_classes=100).to(self.device)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = KLDivergenceLoss(temperature=temperature)
        
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
    
    def _load_pretrained_teacher(self):
        """Load pretrained teacher weights if available"""
        checkpoint_path = './checkpoints/teacher_vgg16.pth'
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained teacher from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.teacher.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("WARNING: No pretrained teacher found. Using randomly initialized teacher.")
            print("For best results, pretrain the teacher first.")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.student.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass through teacher (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher(images)
            
            # Forward pass through student
            student_logits = self.student(images)
            
            # Compute losses
            # Distillation loss (KL divergence with temperature)
            kd_loss = self.kl_loss(student_logits, teacher_logits)
            
            # Cross entropy loss with ground truth labels
            ce_loss = self.ce_loss(student_logits, labels)
            
            # Combined loss
            loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
            
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
    
    def train(self, train_loader, val_loader, num_epochs=200):
        """Train student with knowledge distillation"""
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        print(f"\nTraining parameters:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Alpha (KD weight): {self.alpha}")
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
                    f'./checkpoints/lm_student_best.pth'
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
        """Return the teacher model"""
        return self.teacher


def main():
    """Main training script"""
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = get_cifar100_loaders(
        batch_size=128,
        num_workers=4
    )
    
    # Initialize Knowledge Distillation
    kd = LogitMatchingKD(
        temperature=4.0,
        alpha=0.5,
        learning_rate=0.1,
        weight_decay=5e-4
    )
    
    # Train
    best_val_acc = kd.train(train_loader, val_loader, num_epochs=200)
    
    # Evaluate
    test_acc = kd.evaluate(test_loader)
    
    print("\n" + "="*70)
    print("RESULTS - BASIC LOGIT MATCHING")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()