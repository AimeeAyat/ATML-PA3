"""
Contrastive Representation Distillation (CRD)
Aligns student and teacher representations using contrastive learning.
The student learns to match the teacher's representation space.
Reference: https://arxiv.org/abs/1910.10699 (Contrastive Representation Distillation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os

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


class CRDLoss(nn.Module):
    """
    Contrastive Representation Distillation Loss
    
    Uses memory bank and contrastive learning to align student and teacher representations.
    Simplified version focusing on core idea.
    """
    
    def __init__(self, temperature=0.07, contrast_size=1024):
        super(CRDLoss, self).__init__()
        self.temperature = temperature
        self.contrast_size = contrast_size
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_feat, teacher_feat):
        """
        Compute CRD loss
        
        Args:
            student_feat: Student feature vectors [batch_size, feat_dim]
            teacher_feat: Teacher feature vectors [batch_size, feat_dim]
        
        Returns:
            CRD loss value
        """
        # Normalize features
        student_feat = torch.nn.functional.normalize(student_feat, dim=1)
        teacher_feat = torch.nn.functional.normalize(teacher_feat, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(student_feat, teacher_feat.t()) / self.temperature
        
        # Create labels: diagonal should be positive (matching pairs)
        batch_size = student_feat.shape[0]
        labels = torch.arange(batch_size, device=student_feat.device)
        
        # Contrastive loss (InfoNCE loss)
        loss = self.ce_loss(similarity, labels)
        
        return loss


class ProjectionHead(nn.Module):
    """Projection head to reduce feature dimensionality for contrastive learning"""
    
    def __init__(self, input_dim, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, x):
        return self.projection(x)


class VGGWithProjection(nn.Module):
    """VGG model with projection head for contrastive learning"""
    
    def __init__(self, vgg_model, feat_dim=128):
        super(VGGWithProjection, self).__init__()
        self.features = vgg_model.features
        self.avgpool = vgg_model.avgpool
        
        # Feature extractor (without classifier)
        self.feat_dim = 512  # VGG last conv layer
        self.projection = ProjectionHead(512, feat_dim)
        
        # Full classifier
        self.classifier = vgg_model.classifier
    
    def forward_features(self, x):
        """Return projected features for contrastive learning"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        return x
    
    def forward(self, x):
        """Return classification logits"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ContrastiveRepDistillation:
    """Contrastive Representation Distillation (CRD)"""
    
    def __init__(self, temperature=0.07, contrast_size=1024, 
                 learning_rate=0.1, weight_decay=5e-4, alpha=0.5, feat_dim=128):
        """
        Initialize Contrastive Representation Distillation
        
        Args:
            temperature: Temperature for contrastive learning
            contrast_size: Size of contrast (simplified, not full memory bank)
            learning_rate: Learning rate
            weight_decay: Weight decay
            alpha: Balance between CRD loss and CE loss
            feat_dim: Feature dimension for projection
        """
        self.device = get_device()
        self.temperature = temperature
        self.contrast_size = contrast_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.feat_dim = feat_dim
        
        print("\n" + "="*70)
        print("CONTRASTIVE REPRESENTATION DISTILLATION (CRD)")
        print("="*70)
        
        print("\nLoading Teacher (VGG-16)...")
        self.teacher_base = vgg16(num_classes=100).to(self.device)
        self._load_pretrained_teacher()
        self.teacher = VGGWithProjection(self.teacher_base, feat_dim=feat_dim).to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        print("Loading Student (VGG-11)...")
        self.student_base = vgg11(num_classes=100).to(self.device)
        self.student = VGGWithProjection(self.student_base, feat_dim=feat_dim).to(self.device)
        
        # Loss functions
        self.crd_loss = CRDLoss(temperature=temperature, contrast_size=contrast_size).to(self.device)
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        
        # Optimizer (for student and its projection head)
        self.optimizer = optim.SGD(
            list(self.student_base.parameters()) + list(self.student.projection.parameters()),
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
            self.teacher_base.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("WARNING: No pretrained teacher found. Using randomly initialized teacher.")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.student_base.train()
        self.student.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass through teacher (no gradients)
            with torch.no_grad():
                teacher_features = self.teacher.forward_features(images)
            
            # Forward pass through student
            student_features = self.student.forward_features(images)
            student_logits = self.student_base(images)
            
            # Compute CRD loss (contrastive loss on representations)
            crd_loss = self.crd_loss(student_features, teacher_features)
            
            # Compute cross-entropy loss
            ce_loss = self.ce_loss(student_logits, labels)
            
            # Combined loss
            loss = self.alpha * crd_loss + (1 - self.alpha) * ce_loss
            
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
        """Train student with contrastive representation distillation"""
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        print(f"\nTraining parameters:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Feature dimension: {self.feat_dim}")
        print(f"  Alpha (CRD weight): {self.alpha}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs: {num_epochs}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch [{epoch}/{num_epochs}]")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_acc = validate(self.student_base, val_loader, self.device)
            
            self.val_accs.append(val_acc)
            
            scheduler.step()
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Acc: {val_acc:.2f}%\n")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_checkpoint(
                    self.student_base,
                    self.optimizer,
                    epoch,
                    val_acc,
                    f'./checkpoints/crd_student_best.pth'
                )
        
        return self.best_val_acc
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.student_base.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.student_base(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * total_correct / total_samples
        return test_acc
    
    def get_student_model(self):
        """Return the trained student model"""
        return self.student_base
    
    def get_teacher_model(self):
        """Return the teacher model"""
        return self.teacher_base
    
    def get_train_acc(self):
        """Return the latest training accuracy"""
        return self.train_accs[-1] if self.train_accs else 0.0


def main():
    """Main training script"""
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = get_cifar100_loaders(
        batch_size=128,
        num_workers=4
    )
    
    # Initialize Contrastive Representation Distillation
    kd = ContrastiveRepDistillation(
        temperature=0.07,
        contrast_size=1024,
        learning_rate=0.1,
        weight_decay=5e-4,
        alpha=0.5,
        feat_dim=128
    )
    
    # Train
    best_val_acc = kd.train(train_loader, val_loader, num_epochs=30)
    
    # Evaluate
    test_acc = kd.evaluate(test_loader)
    
    print("\n" + "="*70)
    print("RESULTS - CONTRASTIVE REPRESENTATION DISTILLATION (CRD)")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()