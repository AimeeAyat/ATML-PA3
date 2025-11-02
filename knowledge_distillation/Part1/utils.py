"""
Common utilities for Knowledge Distillation experiments
Includes data loading, model architectures, and shared functions
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import os
from datetime import datetime

# ============================================================================
# DEVICE SETUP
# ============================================================================

def get_device():
    """Get device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# ============================================================================
# DATA LOADING
# ============================================================================

def get_cifar100_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Load CIFAR-100 dataset with standard preprocessing
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of workers for DataLoader
        data_dir: Directory to store dataset
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Normalization constants for CIFAR-100
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load full training set
    full_train_set = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Split into train and validation (80-20 split)
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(
        full_train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply test transform to validation set
    val_set.dataset.transform = test_transform
    
    # Load test set
    test_set = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = 100
    
    print(f"CIFAR-100 loaded successfully")
    print(f"  Train samples: {len(train_set)}")
    print(f"  Val samples: {len(val_set)}")
    print(f"  Test samples: {len(test_set)}")
    print(f"  Number of classes: {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes


# ============================================================================
# VGG ARCHITECTURES
# ============================================================================

class VGG(nn.Module):
    """VGG architecture for CIFAR-100"""
    
    def __init__(self, features, num_classes=100, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    """Create VGG feature layers"""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11(pretrained=False, num_classes=100, **kwargs):
    """VGG 11-layer network"""
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    model = VGG(make_layers(cfg, batch_norm=True), num_classes=num_classes, **kwargs)
    return model


def vgg16(pretrained=False, num_classes=100, **kwargs):
    """VGG 16-layer network"""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = VGG(make_layers(cfg, batch_norm=True), num_classes=num_classes, **kwargs)
    return model


def vgg19(pretrained=False, num_classes=100, **kwargs):
    """VGG 19-layer network"""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    model = VGG(make_layers(cfg, batch_norm=True), num_classes=num_classes, **kwargs)
    return model


# ============================================================================
# TRAINING & EVALUATION UTILITIES
# ============================================================================

def accuracy(output, target, topk=(1,)):
    """Compute accuracy for top-k predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    print(f"Checkpoint loaded from {filepath}")
    return model, optimizer, epoch, accuracy


def validate(model, val_loader, device):
    """Validate model on validation set"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
    accuracy_val = 100 * total_correct / total_samples
    return accuracy_val


# ============================================================================
# TEMPERATURE-BASED SOFTMAX
# ============================================================================

class TemperatureSoftmax(nn.Module):
    """Apply softmax with temperature"""
    def __init__(self, temperature=4.0):
        super(TemperatureSoftmax, self).__init__()
        self.temperature = temperature
    
    def forward(self, logits):
        return torch.nn.functional.softmax(logits / self.temperature, dim=1)


# ============================================================================
# KL DIVERGENCE LOSS
# ============================================================================

class KLDivergenceLoss(nn.Module):
    """KL Divergence loss for knowledge distillation"""
    def __init__(self, temperature=4.0):
        super(KLDivergenceLoss, self).__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits):
        student_probs = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence loss scaled by temperature squared
        loss = self.kl_loss(student_probs, teacher_probs) * (self.temperature ** 2)
        return loss


print("Utils module loaded successfully!")