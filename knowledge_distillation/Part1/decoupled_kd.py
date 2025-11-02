"""
Decoupled Knowledge Distillation (DKD)
Separates distillation loss into two components:
- Target class knowledge: How to emphasize the correct class
- Non-target class knowledge: How to differentiate between wrong classes
Reference: https://arxiv.org/abs/2203.08679
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from .utils import (
    get_device, get_cifar100_loaders, vgg11, vgg16,
    validate, save_checkpoint, load_checkpoint, accuracy
)


class DecoupledKDLoss(nn.Module):
    """Decoupled Knowledge Distillation Loss"""
    
    def __init__(self, temperature=4.0, alpha=1.0, beta=1.0):
        """
        Initialize Decoupled KD Loss
        
        Args:
            temperature: Temperature for softening probabilities
            alpha: Weight for target class knowledge
            beta: Weight for non-target class knowledge
        """
        super(DecoupledKDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Compute decoupled KD loss
        
        DKD = alpha * L_tckd + beta * L_nckd
        where:
        - L_tckd: Target class knowledge distillation loss
        - L_nckd: Non-target class knowledge distillation loss
        
        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            labels: Ground truth labels
        
        Returns:
            Combined DKD loss
        """
        
        # Get softmax probabilities
        student_softmax = torch.nn.functional.softmax(
            student_logits / self.temperature, dim=1
        )
        teacher_softmax = torch.nn.functional.softmax(
            teacher_logits / self.temperature, dim=1
        )
        
        # Get log softmax for student (for KL divergence)
        student_log_softmax = torch.nn.functional.log_softmax(
            student_logits / self.temperature, dim=1
        )
        
        # Create one-hot encoded labels
        num_classes = student_logits.size(1)
        one_hot_labels = torch.zeros_like(student_softmax)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1.0)
        
        # ==============================================================
        # Target Class Knowledge Distillation (TCKD)
        # ==============================================================
        # Focus on how well the student learns the target class
        # Only compute loss for the correct class
        
        # Extract target class probabilities
        student_target = (student_softmax * one_hot_labels).sum(dim=1, keepdim=True)
        teacher_target = (teacher_softmax * one_hot_labels).sum(dim=1, keepdim=True)
        
        # KL divergence for target class: encourages student to put probability mass
        # on the target class similar to teacher
        # L_tckd = KL(teacher_target || student_target)
        tckd_loss = -teacher_target * torch.log(student_target + 1e-8)
        tckd_loss = tckd_loss.mean()
        
        # ==============================================================
        # Non-Target Class Knowledge Distillation (NCKD)
        # ==============================================================
        # Focus on how well the student learns to differentiate non-target classes
        # Mask out the target class
        
        # Create masks for non-target classes
        non_target_mask = 1.0 - one_hot_labels
        
        # Extract non-target probabilities
        student_non_target = (student_softmax * non_target_mask) / (1.0 - student_target + 1e-8)
        teacher_non_target = (teacher_softmax * non_target_mask) / (1.0 - teacher_target + 1e-8)
        
        # KL divergence for non-target classes
        # L_nckd = KL(teacher_non_target || student_non_target)
        student_log_non_target = torch.log(student_non_target + 1e-8)
        nckd_loss = -(teacher_non_target * student_log_non_target).sum(dim=1).mean()
        
        # ==============================================================
        # Combined DKD Loss
        # ==============================================================
        dkd_loss = self.alpha * tckd_loss + self.beta * nckd_loss
        
        return dkd_loss, tckd_loss, nckd_loss


class DecoupledKD:
    """Decoupled Knowledge Distillation"""
    
    def __init__(self, temperature=4.0, alpha=1.0, beta=1.0, 
                 ce_weight=1.0, learning_rate=0.1, weight_decay=5e-4):
        """
        Initialize Decoupled KD
        
        Args:
            temperature: Temperature for softening probabilities
            alpha: Weight for target class knowledge
            beta: Weight for non-target class knowledge
            ce_weight: Weight for cross-entropy loss with ground truth
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.device = get_device()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_weight = ce_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize models
        print("\n" + "="*70)
        print("DECOUPLED KNOWLEDGE DISTILLATION (DKD)")
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
        self.dkd_loss = DecoupledKDLoss(
            temperature=temperature,
            alpha=alpha,
            beta=beta
        )
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
        total_tckd = 0
        total_nckd = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass through teacher (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher(images)
            
            # Forward pass through student
            student_logits = self.student(images)
            
            # Compute DKD loss
            dkd_loss, tckd_loss, nckd_loss = self.dkd_loss(
                student_logits, teacher_logits, labels
            )
            
            # Cross entropy loss with ground truth labels
            ce_loss = self.ce_loss(student_logits, labels)
            
            # Combined loss
            loss = dkd_loss + self.ce_weight * ce_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_tckd += tckd_loss.item()
            total_nckd += nckd_loss.item()
            
            _, predicted = torch.max(student_logits.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            # Progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_tckd = total_tckd / (batch_idx + 1)
                avg_nckd = total_nckd / (batch_idx + 1)
                acc = 100 * total_correct / total_samples
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f}, TCKD: {avg_tckd:.4f}, NCKD: {avg_nckd:.4f}, "
                      f"Acc: {acc:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100 * total_correct / total_samples
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        
        return avg_loss, avg_acc
    
    def train(self, train_loader, val_loader, num_epochs=200):
        """Train student with decoupled knowledge distillation"""
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        print(f"\nTraining parameters:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Alpha (TCKD weight): {self.alpha}")
        print(f"  Beta (NCKD weight): {self.beta}")
        print(f"  CE Loss weight: {self.ce_weight}")
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
                    f'./checkpoints/dkd_student_best.pth'
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
    
    # Initialize Decoupled KD
    kd = DecoupledKD(
        temperature=4.0,
        alpha=1.0,
        beta=1.0,
        ce_weight=1.0,
        learning_rate=0.1,
        weight_decay=5e-4
    )
    
    # Train
    best_val_acc = kd.train(train_loader, val_loader, num_epochs=200)
    
    # Evaluate
    test_acc = kd.evaluate(test_loader)
    
    print("\n" + "="*70)
    print("RESULTS - DECOUPLED KNOWLEDGE DISTILLATION (DKD)")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()