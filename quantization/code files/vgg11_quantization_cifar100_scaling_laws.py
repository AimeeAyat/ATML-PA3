import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from collections import OrderedDict
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg11, VGG11_Weights
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import copy
from torch.autograd import grad
import os
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def measure_inference_latency(model, test_loader, device, dtype=torch.float32, num_batches=50):
    """Measure average inference latency"""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device).to(dtype)
            start = time.time()
            _ = model(inputs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
    return np.mean(times) * 1000  # ms

# Data loading and preprocessing
def get_cifar100_dataloaders(batch_size=128):
    # KEY FIX: Resize to 224x224 to match ImageNet pretraining
    transform_train = transforms.Compose([
        transforms.Resize(224),  # ADD THIS!
        transforms.RandomCrop(224, padding=28),  # Adjust padding
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), 
        #                    (0.2675, 0.2565, 0.2761))
        transforms.Normalize(((0.485, 0.456, 0.406)), 
                           (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),  # ADD THIS!
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), 
        #                    (0.2675, 0.2565, 0.2761)
        transforms.Normalize(((0.485, 0.456, 0.406)), 
                           (0.2675, 0.2565, 0.2761))
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return trainloader, testloader

def get_pretrained_vgg11_cifar100():
    print("\n📦 Loading VGG-11 with ImageNet pretrained weights...")
    model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, 100)
    nn.init.normal_(model.classifier[6].weight, 0, 0.01)
    nn.init.constant_(model.classifier[6].bias, 0)
    print("✅ Loaded pretrained weights and adapted for CIFAR-100")
    return model

import json
import pickle
from datetime import datetime

def save_results(results, filename='quantization_results_v2.json'):
    """Save results dictionary to file"""
    # Add metadata
    results_with_metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'results': results
    }
    
    # Save as JSON
    with open(filename, 'w') as f:
        json.dump(results_with_metadata, f, indent=4)
    print(f"\n💾 Results saved to: {filename}")
    
    # Also save as pickle (preserves exact float precision)
    pkl_filename = filename.replace('.json', '.pkl')
    with open(pkl_filename, 'wb') as f:
        pickle.dump(results_with_metadata, f)
    print(f"💾 Results also saved to: {pkl_filename}")
    
    return filename, pkl_filename


def load_results(filename='quantization_results_v2.json'):
    """Load results from file"""
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(filename, 'r') as f:
            data = json.load(f)
    
    print(f"\n📂 Loaded results from: {filename}")
    print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"   Device: {data.get('device', 'N/A')}")
    return data['results']

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {running_loss/(batch_idx+1):.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# Evaluation function
def evaluate(model, test_loader, device, dtype=torch.float32):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device).to(dtype)
            targets = targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


# Post-Training Quantization Functions
def apply_fp16_quantization(model):
    """Convert model to FP16"""
    return model.half()


def apply_bf16_quantization(model):
    """Convert model to BF16"""
    return model.to(torch.bfloat16)


def quantize_tensor_int8(tensor):
    """Quantize tensor to INT8"""
    scale = tensor.abs().max() / 127.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return quantized, scale


def dequantize_tensor_int8(quantized, scale):
    """Dequantize INT8 tensor back to float"""
    return quantized.float() * scale


def quantize_tensor_int4(tensor):
    """Quantize tensor to INT4 (stored in int8)"""
    scale = tensor.abs().max() / 7.0
    quantized = torch.clamp(torch.round(tensor / scale), -7, 7).to(torch.int8)
    return quantized, scale

def quantize_tensor_int4_packed(tensor):
    """Quantize tensor to INT4 and pack two 4-bit values per byte"""
    scale = tensor.abs().max() / 7.0
    q = torch.clamp(torch.round(tensor / scale), -7, 7).to(torch.int8)

    # Shift [-7,7] → [0,15]
    q = (q + 8).to(torch.uint8)

    # Pad if odd number of elements
    if q.numel() % 2 != 0:
        q = torch.cat([q, torch.zeros(1, dtype=torch.uint8, device=q.device)])

    # Pack two 4-bit values into one byte
    q_reshaped = q.view(-1, 2)
    packed = (q_reshaped[:, 0] & 0x0F) | ((q_reshaped[:, 1] & 0x0F) << 4)

    return packed, scale

def dequantize_tensor_int4(quantized, scale):
    """Dequantize INT4 tensor back to float"""
    return quantized.float() * scale

def dequantize_tensor_int4_packed(packed, scale):
    """Unpack 4-bit values and dequantize back to float"""
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    unpacked[0::2] = packed & 0x0F
    unpacked[1::2] = (packed >> 4) & 0x0F
    unpacked = unpacked.to(torch.int8) - 8
    return unpacked.float() * scale

class QuantizedLinear(nn.Module):
    """Quantized Linear Layer"""
    def __init__(self, weight, bias, scale, bit_width='int8'):
        super().__init__()
        self.register_buffer('weight_q', weight)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self.register_buffer('scale', scale)
        self.bit_width = bit_width
    
    def forward(self, x):
        if self.bit_width == 'int8':
            weight = dequantize_tensor_int8(self.weight_q, self.scale)
        else:  # int4
            weight = dequantize_tensor_int4(self.weight_q, self.scale)
        return nn.functional.linear(x, weight, self.bias)


class QuantizedConv2d(nn.Module):
    """Quantized Conv2d Layer"""
    def __init__(self, weight, bias, scale, stride, padding, bit_width='int8'):
        super().__init__()
        self.register_buffer('weight_q', weight)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self.register_buffer('scale', scale)
        self.stride = stride
        self.padding = padding
        self.bit_width = bit_width
    
    def forward(self, x):
        if self.bit_width == 'int8':
            weight = dequantize_tensor_int8(self.weight_q, self.scale)
        else:  # int4
            weight = dequantize_tensor_int4(self.weight_q, self.scale)
        return nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding)


def apply_int_quantization(model, bit_width='int8'):
    """Apply INT8 or INT4 quantization to model"""
    quantize_fn = quantize_tensor_int8 if bit_width == 'int8' else quantize_tensor_int4
    
    # Create a new model with quantized layers
    new_model = copy.deepcopy(model)
    
    # Quantize convolutional layers in features
    for i, layer in enumerate(new_model.features):
        if isinstance(layer, nn.Conv2d):
            weight_q, scale = quantize_fn(layer.weight.data)
            bias = layer.bias.data if layer.bias is not None else None
            new_model.features[i] = QuantizedConv2d(
                weight_q, bias, scale, layer.stride, layer.padding, bit_width
            )
    
    # Quantize linear layers in classifier
    for i, layer in enumerate(new_model.classifier):
        if isinstance(layer, nn.Linear):
            weight_q, scale = quantize_fn(layer.weight.data)
            bias = layer.bias.data if layer.bias is not None else None
            new_model.classifier[i] = QuantizedLinear(
                weight_q, bias, scale, bit_width
            )
    
    return new_model


# Quantization-Aware Training
class StraightThroughEstimator(torch.autograd.Function):
    """Custom STE: forward quantizes, backward passes gradient straight through"""
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient of round(x) ≈ 1
        return grad_output


class QAT_QuantizedLayer(nn.Module):
    """Layer with fake quantization for QAT"""
    def __init__(self, bit_width='int8'):
        super().__init__()
        self.bit_width = bit_width
        if bit_width == 'int8':
            self.qmin, self.qmax = -127, 127  # Fixed: symmetric
            self.scale_factor = 127.0  # Fixed: /127 not /127.5
        else:  # int4
            self.qmin, self.qmax = -7, 7  # Fixed: symmetric
            self.scale_factor = 7.0  # Fixed: /7 not /7.5
    
    def fake_quantize(self, tensor):
        """Simulate quantization with STE for gradient flow"""
        scale = tensor.abs().max() / self.scale_factor
        scale = torch.clamp(scale, min=1e-8)
        
        # Use STE for round operation
        normalized = tensor / scale
        quantized = StraightThroughEstimator.apply(normalized)
        quantized = torch.clamp(quantized, self.qmin, self.qmax)
        
        return quantized * scale


class QAT_FP16Layer(nn.Module):
    """Fake FP16 quantization for QAT (educational - no real benefit)"""
    def __init__(self):
        super().__init__()
    
    def fake_quantize(self, tensor):
        """Simulate FP16 precision"""
        return tensor.half().float()  # Convert to FP16 and back


class QAT_BF16Layer(nn.Module):
    """Fake BF16 quantization for QAT (educational - no real benefit)"""
    def __init__(self):
        super().__init__()
    
    def fake_quantize(self, tensor):
        """Simulate BF16 precision"""
        return tensor.bfloat16().float()  # Convert to BF16 and back
    
class QAT_Conv2d(nn.Conv2d):
    """Conv2d with Quantization-Aware Training"""
    def __init__(self, *args, bit_width='int8', **kwargs):
        super().__init__(*args, **kwargs)
        if bit_width in ['fp16', 'bf16']:
            self.quantizer = QAT_FP16Layer() if bit_width == 'fp16' else QAT_BF16Layer()
        else:
            self.quantizer = QAT_QuantizedLayer(bit_width)
    
    def forward(self, x):
        weight_q = self.quantizer.fake_quantize(self.weight)
        return nn.functional.conv2d(x, weight_q, self.bias, self.stride, self.padding)


class QAT_Linear(nn.Linear):
    """Linear layer with Quantization-Aware Training"""
    def __init__(self, *args, bit_width='int8', **kwargs):
        super().__init__(*args, **kwargs)
        if bit_width in ['fp16', 'bf16']:
            self.quantizer = QAT_FP16Layer() if bit_width == 'fp16' else QAT_BF16Layer()
        else:
            self.quantizer = QAT_QuantizedLayer(bit_width)
    
    def forward(self, x):
        weight_q = self.quantizer.fake_quantize(self.weight)
        return nn.functional.linear(x, weight_q, self.bias)


def convert_to_qat_model(model, bit_width='int8'):
    """Convert regular model to QAT model"""
    qat_model = copy.deepcopy(model)
    
    # Replace Conv2d layers
    for i, layer in enumerate(qat_model.features):
        if isinstance(layer, nn.Conv2d):
            qat_layer = QAT_Conv2d(
                layer.in_channels, layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=layer.bias is not None,
                bit_width=bit_width
            )
            qat_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                qat_layer.bias.data = layer.bias.data.clone()
            qat_model.features[i] = qat_layer
    
    # Replace Linear layers
    for i, layer in enumerate(qat_model.classifier):
        if isinstance(layer, nn.Linear):
            qat_layer = QAT_Linear(
                layer.in_features, layer.out_features,
                bias=layer.bias is not None,
                bit_width=bit_width
            )
            qat_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                qat_layer.bias.data = layer.bias.data.clone()
            qat_model.classifier[i] = qat_layer
    
    return qat_model

def convert_qat_to_quantized(qat_model, bit_width='int8'):
    """Convert QAT model with fake quantization to actual quantized model"""
    print(f"   Converting QAT model to actual {bit_width.upper()} quantized weights...")
    
    if bit_width in ['fp16', 'bf16']:
        # Create a clean model without QAT layers
        regular_model = copy.deepcopy(qat_model)
        
        # Replace QAT_Conv2d with regular Conv2d
        for i, layer in enumerate(regular_model.features):
            if isinstance(layer, QAT_Conv2d):
                regular_conv = nn.Conv2d(
                    layer.in_channels, layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=layer.bias is not None
                )
                regular_conv.weight.data = layer.weight.data.clone()
                if layer.bias is not None:
                    regular_conv.bias.data = layer.bias.data.clone()
                regular_model.features[i] = regular_conv
        
        # Replace QAT_Linear with regular Linear
        for i, layer in enumerate(regular_model.classifier):
            if isinstance(layer, QAT_Linear):
                regular_linear = nn.Linear(
                    layer.in_features, layer.out_features,
                    bias=layer.bias is not None
                )
                regular_linear.weight.data = layer.weight.data.clone()
                if layer.bias is not None:
                    regular_linear.bias.data = layer.bias.data.clone()
                regular_model.classifier[i] = regular_linear
        
        # Now convert to FP16/BF16
        if bit_width == 'fp16':
            return regular_model.half()
        else:
            return regular_model.bfloat16()
    
    # For INT8/INT4, extract weights and quantize
    quantize_fn = quantize_tensor_int8 if bit_width == 'int8' else quantize_tensor_int4
    
    # Create quantized model
    quantized_model = copy.deepcopy(qat_model)
    
    # Replace Conv2d layers
    for i, layer in enumerate(quantized_model.features):
        if isinstance(layer, QAT_Conv2d):
            # Extract trained weights
            weight_q, scale = quantize_fn(layer.weight.data)
            bias = layer.bias.data if layer.bias is not None else None
            
            # Replace with quantized layer
            quantized_model.features[i] = QuantizedConv2d(
                weight_q, bias, scale, 
                layer.stride, layer.padding, bit_width
            )
    
    # Replace Linear layers
    for i, layer in enumerate(quantized_model.classifier):
        if isinstance(layer, QAT_Linear):
            # Extract trained weights
            weight_q, scale = quantize_fn(layer.weight.data)
            bias = layer.bias.data if layer.bias is not None else None
            
            # Replace with quantized layer
            quantized_model.classifier[i] = QuantizedLinear(
                weight_q, bias, scale, bit_width
            )
    
    return quantized_model

def get_model_size(model):
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.getbuffer().nbytes / (1024 ** 2)
    return size_mb

# Main experiment function
def run_quantization_experiments():
    """Run complete PTQ and QAT experiments"""
    
    print("="*80)
    print("VGG11 Quantization Experiments on CIFAR-100")
    print("="*80)
    
    # Load data
    print("\nLoading CIFAR-100 dataset...")
    train_loader, test_loader = get_cifar100_dataloaders(batch_size=128)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Train baseline model or load pretrained
    print("\n" + "="*80)
    print("Step 1: Training Baseline FP32 Model")
    print("="*80)
    
    model_fp32 = get_pretrained_vgg11_cifar100().to(device)
    
    # Check if pretrained model exists
    import os
    if os.path.exists('vgg11_cifar100_fp32.pth'):
        print("Loading pretrained FP32 model...")
        model_fp32.load_state_dict(torch.load('vgg11_cifar100_fp32.pth'))
    else:
        print("Training FP32 model from scratch...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_fp32.parameters(), lr=0.001, 
                            momentum=0.9, weight_decay=5e-4)

    
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

        num_epochs = 100
        best_acc = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            train_loss, train_acc = train_epoch(model_fp32, train_loader, 
                                               criterion, optimizer, device)
            test_acc = evaluate(model_fp32, test_loader, device)
            scheduler.step()
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test Acc: {test_acc:.2f}%')
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model_fp32.state_dict(), 'vgg11_cifar100_fp32.pth')
                print(f'Saved best model with accuracy: {best_acc:.2f}%')
    
    # Evaluate baseline
    baseline_acc = evaluate(model_fp32, test_loader, device)
    print(f"\nBaseline FP32 Accuracy: {baseline_acc:.2f}%")
    
    # Results storage
    results = {
    'baseline': {
        'accuracy': baseline_acc,
        'latency_ms': measure_inference_latency(model_fp32, test_loader, device)
    },
    'ptq': {},
    'qat': {}
    }
    
    # ========================================================================
    # POST-TRAINING QUANTIZATION (PTQ)
    # ========================================================================
    print("\n" + "="*80)
    print("Step 2: Post-Training Quantization (PTQ)")
    print("="*80)
    
    # FP16 PTQ
    print("\n--- FP16 PTQ ---")
    model_fp16 = copy.deepcopy(model_fp32).to(device)
    model_fp16 = apply_fp16_quantization(model_fp16)
    acc_fp16 = evaluate(model_fp16, test_loader, device, dtype=torch.float16)
    lat_fp16 = measure_inference_latency(model_fp16, test_loader, device, dtype=torch.float16)
    results['ptq']['FP16'] = {'accuracy': acc_fp16, 'latency_ms': lat_fp16, 'drop': baseline_acc - acc_fp16}

    # results['FP16 PTQ'] = acc_fp16
    print(f"FP16 PTQ Accuracy: {acc_fp16:.2f}%")
    print(f"Accuracy Drop: {baseline_acc - acc_fp16:.2f}%")
    print(f"FP16 PTQ Accuracy: {acc_fp16:.2f}%, Latency: {lat_fp16:.2f}ms")

    # BF16 PTQ
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("\n--- BF16 PTQ ---")
        model_bf16 = copy.deepcopy(model_fp32).to(device)
        model_bf16 = apply_bf16_quantization(model_bf16)
        acc_bf16 = evaluate(model_bf16, test_loader, device, dtype=torch.bfloat16)
        lat_bf16 = measure_inference_latency(model_bf16, test_loader, device, dtype=torch.bfloat16)
        results['ptq']['BF16'] = {'accuracy': acc_bf16, 'latency_ms': lat_bf16, 'drop': baseline_acc - acc_bf16}

        # results['BF16 PTQ'] = acc_bf16
        print(f"BF16 PTQ Accuracy: {acc_bf16:.2f}%")
        print(f"Accuracy Drop: {baseline_acc - acc_bf16:.2f}%")
        print(f"BF16 PTQ Accuracy: {acc_bf16:.2f}%, Latency: {lat_bf16:.2f}ms")

    else:
        print("\n--- BF16 PTQ ---")
        print("BF16 not supported on this device, skipping...")
        # results['BF16 PTQ'] = None
        results['ptq']['BF16'] = None  # Store None in the new structure

    
    # INT8 PTQ
    print("\n--- INT8 PTQ ---")
    model_int8 = apply_int_quantization(model_fp32.cpu(), bit_width='int8').to(device)
    acc_int8 = evaluate(model_int8, test_loader, device)
    lat_int8 = measure_inference_latency(model_int8, test_loader, device, dtype=torch.float32)
    results['ptq']['INT8'] = {'accuracy': acc_int8, 'latency_ms': lat_int8, 'drop': baseline_acc - acc_int8}

    # results['INT8 PTQ'] = acc_int8
    print(f"INT8 PTQ Accuracy: {acc_int8:.2f}%")
    print(f"Accuracy Drop: {baseline_acc - acc_int8:.2f}%")
    print(f"INT8 PTQ Accuracy: {acc_int8:.2f}%, Latency: {lat_int8:.2f}ms")

    # INT4 PTQ
    print("\n--- INT4 PTQ ---")
    model_int4 = apply_int_quantization(model_fp32.cpu(), bit_width='int4').to(device)
    acc_int4 = evaluate(model_int4, test_loader, device)
    lat_int4 = measure_inference_latency(model_int4, test_loader, device, dtype=torch.float32)
    results['ptq']['INT4'] = {'accuracy': acc_int4, 'latency_ms': lat_int4, 'drop': baseline_acc - acc_int4}

    # results['INT4 PTQ'] = acc_int4
    print(f"INT4 PTQ Accuracy: {acc_int4:.2f}%")
    print(f"Accuracy Drop: {baseline_acc - acc_int4:.2f}%")
    print(f"INT4 PTQ Accuracy: {acc_int4:.2f}%, Latency: {lat_int4:.2f}ms")

    # ========================================================================
    # QUANTIZATION-AWARE TRAINING (QAT)
    # ========================================================================
    print("\n" + "="*80)
    print("Step 3: Quantization-Aware Training (QAT)")
    print("="*80)
    
    qat_epochs = 100  # Number of QAT fine-tuning epochs
    criterion = nn.CrossEntropyLoss()
    
    for bit_width, name in [('fp16', 'FP16'), ('bf16', 'BF16'), ('int8', 'INT8'), ('int4', 'INT4')]:
    # Skip BF16 if not supported
        if bit_width == 'bf16' and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            print(f"\n--- {name} QAT ---")
            print("BF16 not supported, skipping...")
            continue
        
        print(f"\n--- {name} QAT ---")
        model_pretrained_imagenet = get_pretrained_vgg11_cifar100().to(device)  # Fresh ImageNet weights
        # Convert to QAT model
        model_qat = convert_to_qat_model(model_pretrained_imagenet, bit_width=bit_width).to(device)

        # Use smaller learning rate for fine-tuning
        optimizer = optim.SGD(model_qat.parameters(), lr=0.001, 
                            momentum=0.9, weight_decay=5e-4)
        qat_start_time = time.time()
        initial_acc = evaluate(model_qat, test_loader, device)
        recovery_epoch = None
        recovery_time = None
        epoch_history = []

        print(f"Initial: {initial_acc:.2f}% | Baseline: {baseline_acc:.2f}% | PTQ: {results['ptq'][name]['accuracy']:.2f}%")


        for epoch in range(qat_epochs):
            epoch_start = time.time()

            train_loss, train_acc = train_epoch(model_qat, train_loader, criterion, optimizer, device)
            test_acc = evaluate(model_qat, test_loader, device)
            epoch_time = time.time() - epoch_start

            epoch_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'time_sec': epoch_time
            })

            print(f"Epoch {epoch+1}: "
                f"TrainAcc={train_acc:.2f}%, TestAcc={test_acc:.2f}% "
                f"(+{test_acc - results['ptq'][name]['accuracy']:.2f}% from PTQ)")

            # ✅ Early stop condition — once recovered baseline
            if test_acc >= baseline_acc:
                recovery_epoch = epoch + 1
                recovery_time = time.time() - qat_start_time
                print(f"\n✅ {name} QAT recovered baseline accuracy at epoch {recovery_epoch} "
                    f"after {recovery_time:.2f}s. Stopping early.\n")
                break
        
        total_training_time = time.time() - qat_start_time
        # *** ADD THIS: Convert QAT model to actual quantized weights ***
        print(f"\n   Converting {name} QAT model to quantized weights...")
        model_qat_quantized = convert_qat_to_quantized(model_qat, bit_width=bit_width).to(device)


        final_acc = evaluate(model_qat_quantized, test_loader, device,
                         dtype=torch.float16 if bit_width=='fp16' else torch.bfloat16 if bit_width=='bf16' else torch.float32)
        final_latency = measure_inference_latency(model_qat_quantized, test_loader, device,
                         dtype=torch.float16 if bit_width=='fp16' else torch.bfloat16 if bit_width=='bf16' else torch.float32)
        
        # Measure model size
        model_size = get_model_size(model_qat_quantized)

        # Save model
        save_path = f"qat_model_{bit_width}.pkl"
        torch.save(model_qat_quantized.state_dict(), save_path)
        print(f"Saved {name} QAT model ({model_size:.2f} MB) → {save_path}")

       
        results['qat'][name] = {
        'initial_accuracy': initial_acc,
        'final_accuracy': final_acc,
        'baseline_accuracy': baseline_acc,
        'ptq_accuracy': results['ptq'][name]['accuracy'],
        'recovery_from_ptq': final_acc - results['ptq'][name]['accuracy'],
        'vs_baseline': baseline_acc - final_acc,
        'recovered_baseline': recovery_epoch is not None,
        'recovery_epoch': recovery_epoch,
        'recovery_time_sec': recovery_time,
        'total_training_time_sec': total_training_time,
        'latency_ms': final_latency,
        'epoch_history': epoch_history,
        'model_size': model_size
    }
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print("\n{:<20} {:<15} {:<15}".format("Method", "Accuracy (%)", "vs Baseline"))
    print("-" * 50)
    
    for method, acc in results.items():
        if acc is not None:
            drop = baseline_acc - acc
            print("{:<20} {:<15.2f} {:<15.2f}".format(method, acc, drop))
        else:
            print("{:<20} {:<15} {:<15}".format(method, "N/A", "N/A"))
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS")
    print("="*80)
    
    print("\n1. PTQ Results:")
    print(f"   - FP16: Minimal accuracy loss ({baseline_acc - results['FP16 PTQ']:.2f}%)")
    if results['BF16 PTQ'] is not None:
        print(f"   - BF16: Minimal accuracy loss ({baseline_acc - results['BF16 PTQ']:.2f}%)")
    print(f"   - INT8: Moderate accuracy loss ({baseline_acc - results['INT8 PTQ']:.2f}%)")
    print(f"   - INT4: Significant accuracy loss ({baseline_acc - results['INT4 PTQ']:.2f}%)")
    
    print("\n2. QAT Recovery:")
    print(f"   - INT8: Recovered {results['INT8 QAT'] - results['INT8 PTQ']:.2f}% in {qat_epochs} epochs")
    print(f"   - INT4: Recovered {results['INT4 QAT'] - results['INT4 PTQ']:.2f}% in {qat_epochs} epochs")
    
    print("\n3. Trade-offs:")
    print("   - PTQ: Fast, simple, but accuracy loss at low bit-widths")
    print("   - QAT: More compute intensive, but recovers performance significantly")
    print("   - Lower bit-widths require more QAT epochs for full recovery")


    save_results(results, filename='quantization_results_v2.json')
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    # Extract baseline
    baseline_acc = results['baseline']['accuracy']
    baseline_lat = results['baseline']['latency_ms']

    print(f"\n{'Method':<20} {'Accuracy (%)':<15} {'Latency (ms)':<15} {'vs Baseline':<15}")
    print("-" * 65)
    print(f"{'FP32 Baseline':<20} {baseline_acc:<15.2f} {baseline_lat:<15.2f} {0.0:<15.2f}")

    # PTQ Results
    print("\nPTQ Results:")
    for name, data in results['ptq'].items():
        if data is not None:
            acc = data['accuracy']
            lat = data['latency_ms']
            drop = baseline_acc - acc
            print(f"{name + ' PTQ':<20} {acc:<15.2f} {lat:<15.2f} {drop:<15.2f}")

    # QAT Results
    print("\nQAT Results:")
    for name, data in results['qat'].items():
        if data is not None:
            acc = data['final_accuracy']
            lat = data['latency_ms']
            drop = baseline_acc - acc
            recovery = data['recovery_from_ptq']
            rec_time = data.get('recovery_time_sec', 'N/A')
            rec_epoch = data.get('recovery_epoch', 'N/A')
            print(f"{name + ' QAT':<20} {acc:<15.2f} {lat:<15.2f} {drop:<15.2f}")
            # print(f"  → Recovered: {recovery:+.2f}% | Epoch: {rec_epoch} | Time: {rec_time if rec_time == 'N/A' else f'{rec_time:.1f}s'}")
            print(f"  → Recovered: {recovery:+.2f}% | Epoch: {rec_epoch or 'N/A'} | Time: {f'{rec_time:.1f}s' if rec_time else 'N/A'}")
    print("\n" + "="*80)
    print("KEY OBSERVATIONS")
    print("="*80)

    print("\n1. PTQ Results:")
    for name in ['FP16', 'BF16', 'INT8', 'INT4']:
        if results['ptq'].get(name):
            drop = results['ptq'][name]['drop']
            print(f"   - {name}: {drop:.2f}% accuracy loss")

    print("\n2. QAT Recovery:")
    for name in ['INT8', 'INT4']:
        if results['qat'].get(name):
            recovery = results['qat'][name]['recovery_from_ptq']
            epochs = results['qat'][name].get('recovery_epoch', 'N/A')
            print(f"   - {name}: Recovered {recovery:.2f}% | Baseline at epoch {epochs}")

    print("\n3. Trade-offs:")
    print("   - PTQ: Fast, simple, but accuracy loss at low bit-widths")
    print("   - QAT: More compute intensive, but recovers performance significantly")
    print("   - FP16/BF16: Minimal loss, QAT not needed")
    print("   - INT8/INT4: Benefit most from QAT")

    save_results(results, filename='quantization_results_v2.json')

    return results


if __name__ == "__main__":
    results = run_quantization_experiments()
    print("\n🔍 Testing packed INT4 size reduction without retraining...")

    # Load pretrained FP32 model (must exist)
    model = get_pretrained_vgg11_cifar100().to('cpu')
    model.load_state_dict(torch.load('vgg11_cifar100_fp32.pth', map_location='cpu'))

    # Quantize only the classifier weights for a quick test
    fc_layer = model.classifier[6]
    print(f"Original weight shape: {fc_layer.weight.shape}")

    weight = fc_layer.weight.data.clone()
    quant_packed, scale = quantize_tensor_int4_packed(weight)

    # Measure storage sizes
    import io
    fp32_bytes = io.BytesIO()
    torch.save(weight, fp32_bytes)
    fp32_size = fp32_bytes.getbuffer().nbytes / (1024**2)

    int4_bytes = io.BytesIO()
    torch.save(quant_packed, int4_bytes)
    int4_size = int4_bytes.getbuffer().nbytes / (1024**2)

    print(f"FP32 weight size: {fp32_size:.2f} MB")
    print(f"Packed INT4 weight size: {int4_size:.2f} MB (↓ {fp32_size/int4_size:.2f}x smaller)")

    # Save and check actual disk file sizes
    torch.save(weight, "fc_fp32_weights.pth")
    torch.save(quant_packed, "fc_int4_packed.pth")

    import os
    print(f"Disk FP32: {os.path.getsize('fc_fp32_weights.pth')/1e6:.2f} MB")
    print(f"Disk INT4: {os.path.getsize('fc_int4_packed.pth')/1e6:.2f} MB")
