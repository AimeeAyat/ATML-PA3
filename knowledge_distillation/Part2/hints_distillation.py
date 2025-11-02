"""
Hints-based Distillation (FitNet)
Transfers knowledge through intermediate feature representations (hidden layers).
Student learns to match teacher's intermediate layer activations.
Reference: https://arxiv.org/abs/1412.4203 (FitNet: Hints for Thin Deep Nets)
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
    from utils import (
        get_device, get_cifar100_loaders, vgg11, vgg16,
        validate, save_checkpoint, accuracy
    )


class HintLoss(nn.Module):
    """Hint Loss - L2 loss between intermediate feature maps"""
    
    def __init__(self):
        super(HintLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_features, teacher_features):
        """
        Compute hint loss (L2 distance between feature maps)
        
        Args:
            student_features: Student intermediate feature maps
            teacher_features: Teacher intermediate feature maps
        
        Returns:
            MSE loss between features
        """
        # Ensure spatial dimensions match via interpolation if needed
        if student_features.shape[2:] != teacher_features.shape[2:]:
            student_features = nn.functional.interpolate(
                student_features,
                size=teacher_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        loss = self.mse_loss(student_features, teacher_features)
        return loss


class FeatureAdapter(nn.Module):
    """Adapter to match feature dimensions between student and teacher"""
    
    def __init__(self, student_channels, teacher_channels, student_spatial=None, teacher_spatial=None):
        super(FeatureAdapter, self).__init__()
        # Channel adapter (1x1 conv)
        self.channel_adapter = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=True)
        
        # Store spatial dimensions for adaptive pooling/upsampling
        self.student_spatial = student_spatial
        self.teacher_spatial = teacher_spatial
    
    def forward(self, x):
        # First adapt channels
        x = self.channel_adapter(x)
        
        # Then adapt spatial dimensions if needed
        if self.teacher_spatial is not None and x.shape[2:] != self.teacher_spatial:
            x = nn.functional.interpolate(
                x,
                size=self.teacher_spatial,
                mode='bilinear',
                align_corners=False
            )
        
        return x


class VGGWithHooks(nn.Module):
    """VGG model that outputs intermediate features"""
    
    def __init__(self, vgg_model, layer_idx=8):
        """
        Wrap VGG model to extract features from specific layer
        
        Args:
            vgg_model: VGG model
            layer_idx: Which layer to extract features from (0-based in features)
        """
        super(VGGWithHooks, self).__init__()
        self.features = nn.Sequential(*list(vgg_model.features.children())[:layer_idx+1])
        self.avgpool = vgg_model.avgpool
        self.classifier = vgg_model.classifier
    
    def forward_intermediate(self, x):
        """Return intermediate features (before final layers)"""
        x = self.features(x)
        return x
    
    def forward(self, x):
        """Return final logits"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class HintsDistillation:
    """Hints-based Knowledge Distillation (FitNet)"""
    
    def __init__(self, hint_layer=8, learning_rate=0.1, weight_decay=5e-4, alpha=0.5):
        """
        Initialize Hints-based KD
        
        Args:
            hint_layer: Which layer to extract hints from (0-based index in features)
            learning_rate: Learning rate
            weight_decay: Weight decay
            alpha: Balance between hint loss and CE loss
        """
        self.device = get_device()
        self.hint_layer = hint_layer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.alpha = alpha
        
        print("\n" + "="*70)
        print("HINTS-BASED DISTILLATION (FitNet)")
        print("="*70)
        
        print("\nLoading Teacher (VGG-16)...")
        self.teacher_base = vgg16(num_classes=100).to(self.device)
        self._load_pretrained_teacher()
        self.teacher = VGGWithHooks(self.teacher_base, layer_idx=hint_layer)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        print("Loading Student (VGG-11)...")
        self.student_base = vgg11(num_classes=100).to(self.device)
        self.student = VGGWithHooks(self.student_base, layer_idx=hint_layer)
        
        # Get feature dimensions for adapter
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        with torch.no_grad():
            student_feat = self.student.forward_intermediate(dummy_input)
            teacher_feat = self.teacher.forward_intermediate(dummy_input)
        
        student_channels = student_feat.shape[1]
        teacher_channels = teacher_feat.shape[1]
        student_spatial = tuple(student_feat.shape[2:])
        teacher_spatial = tuple(teacher_feat.shape[2:])
        
        print(f"  Student feature shape: {student_feat.shape}")
        print(f"  Teacher feature shape: {teacher_feat.shape}")
        
        # Adapter to match dimensions (channels and spatial)
        self.adapter = FeatureAdapter(
            student_channels, teacher_channels,
            student_spatial, teacher_spatial
        ).to(self.device)
        
        # Loss functions
        self.hint_loss = HintLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Optimizer (for both student and adapter)
        self.optimizer = optim.SGD(
            list(self.student_base.parameters()) + list(self.adapter.parameters()),
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
        self.adapter.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass through teacher (no gradients)
            with torch.no_grad():
                teacher_features = self.teacher.forward_intermediate(images)
            
            # Forward pass through student
            student_features = self.student.forward_intermediate(images)
            student_logits = self.student_base(images)
            
            # Adapt student features to match teacher dimensions
            adapted_student_features = self.adapter(student_features)
            
            # Compute hint loss (L2 distance between feature maps)
            hint_loss = self.hint_loss(adapted_student_features, teacher_features)
            
            # Compute cross-entropy loss
            ce_loss = self.ce_loss(student_logits, labels)
            
            # Combined loss
            loss = self.alpha * hint_loss + (1 - self.alpha) * ce_loss
            
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
        """Train student with hints distillation"""
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        print(f"\nTraining parameters:")
        print(f"  Hint layer: {self.hint_layer}")
        print(f"  Alpha (hint weight): {self.alpha}")
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
                    f'./checkpoints/hints_student_best.pth'
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
    
    # Initialize Hints Distillation
    kd = HintsDistillation(
        hint_layer=8,
        learning_rate=0.1,
        weight_decay=5e-4,
        alpha=0.5
    )
    
    # Train
    best_val_acc = kd.train(train_loader, val_loader, num_epochs=30)
    
    # Evaluate
    test_acc = kd.evaluate(test_loader)
    
    print("\n" + "="*70)
    print("RESULTS - HINTS-BASED DISTILLATION")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()