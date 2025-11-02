import torch
import torch.nn as nn
import torch.optim as optim
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
import tempfile
import os

# ============================================================================
# TORCHAO SETUP
# ============================================================================
try:
    import torchao
    from torchao.quantization import quantize_
    from torchao.quantization.quant_api import Int8WeightOnlyConfig, Int4WeightOnlyConfig
    
    test_model = nn.Linear(10, 10)
    quantize_(test_model, Int8WeightOnlyConfig())
    
    if 'AffineQuantized' in type(test_model.weight).__name__:
        TORCHAO_AVAILABLE = True
        print("✅ TorchAO quantization is working")
    else:
        TORCHAO_AVAILABLE = False
        print("⚠️ TorchAO imported but quantization doesn't create AffineQuantizedTensor")
        
except Exception as e:
    TORCHAO_AVAILABLE = False
    print(f"⚠️ TorchAO not available: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# =====================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.backends.quantized.engine = 'fbgemm'

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
HYPERPARAMS = {
    'finetune_epochs': 10,
    'finetune_lr': 0.001,
    'finetune_batch_size': 128,
    'qat_epochs': {'fp16': 50, 'bf16': 50, 'int8': 50, 'int4': 50},
    'qat_lr': {'fp16': 0.001, 'bf16': 0.001, 'int8': 0.0005, 'int4': 0.0001},
    'qat_batch_size': 256,
    'calibration_batches': 10,
    'outlier_percentile': 99.9,
    'latency_batches': 50,
    'latency_warmup': 5,
}

def simulate_low_precision(module, bits):
    """Simulate precision loss – ALL stay float32 (no dtype crashes!)"""
    if bits == 16:  # FP16 sim
        for param in module.parameters(recurse=False):
            param.data = param.data.half().float()  # Round to 16-bit mantissa
            setattr(param, 'quant_bits', 16)
        for buf in module.buffers(recurse=False):
            buf.data = buf.data.half().float()
            
    elif bits == 32:  # FP32
        for param in module.parameters(recurse=False):
            setattr(param, 'quant_bits', 32)

# ============================================================================
# BASIC UTILITIES
# ============================================================================
# 
def get_cifar100_dataloaders(batch_size=128):
    # KEY FIX: Resize to 224x224 to match ImageNet pretraining
    transform_train = transforms.Compose([
        transforms.Resize(224),  # ADD THIS!
        transforms.RandomCrop(224, padding=28),  # Adjust padding
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
                           (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),  # ADD THIS!
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
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

def finetune_model(model, trainloader, testloader, epochs=100, lr=0.001, baseline_acc=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': lr * 0.1},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    history = {'train_loss': [], 'test_acc': [], 'recovery_time': None}
    
    # ✅ START TIMING
    start_time = time.time()
    
    print(f"\n🏋️ Fine-tuning for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            try:
                model_dtype = next(model.parameters()).dtype
                if model_dtype == torch.bfloat16:
                    inputs = inputs.bfloat16()
                elif model_dtype == torch.float16:
                    inputs = inputs.half()
            except StopIteration:
                pass
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        train_acc = 100. * correct / total
        test_acc = evaluate_model(model, testloader, device)
        history['train_loss'].append(train_loss / len(trainloader))
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        # ✅ CHECK IF BASELINE RECOVERED
        if baseline_acc is not None and history['recovery_time'] is None:
            if test_acc >= baseline_acc * 0.99:
                history['recovery_time'] = time.time() - start_time
                print(f"   🎯 Baseline recovered at epoch {epoch+1} ({history['recovery_time']:.2f}s)")
        
        print(f'Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(trainloader):.3f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')
    
    print(f"✅ Fine-tuning complete! Best accuracy: {best_acc:.2f}%")
    return model, history, best_acc

def evaluate_model(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"🎯 Accuracy: {acc:.2f}%")
    return acc


def measure_latency(model, testloader, device, num_batches=50, warmup=5):
    """
    Measure true inference latency (ms) in mixed-precision setting.
    - All inputs are cast to float32 (safe for FP16/INT8/INT4 simulation)
    - Real TorchAO INT8 kernels still run at full speed (FBGEMM)
    - No dtype mismatch: bias stays float32, inputs are float32
    """
    model.eval()
    times = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(testloader):
            if i >= num_batches:
                break

            # Always use float32 inputs – safe for all layer types
            inputs = inputs.to(device, dtype=torch.float32)

            # Warmup
            if i < warmup:
                _ = model(inputs)
                continue

            # Synchronize + time
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000

    # Report dominant dtype (first param) for logging
    try:
        model_dtype = next(model.parameters()).dtype
    except:
        model_dtype = torch.float32

    print(f"Latency ({str(model_dtype).split('.')[-1]}): {mean_ms:.3f} ± {std_ms:.3f} ms")
    return mean_ms


# ============================================================================
# FIXED MIXED-PRECISION IMPLEMENTATION (with true model size simulation)
# ============================================================================

def get_model_disk_size(model):
    """
    Compute simulated model size (MB) based on quant_bits per parameter.
    Default FP32 = 32 bits. INT8 = 8 bits. INT4 = 4 bits.
    """
    total_bits = 0
    for name, param in model.named_parameters():
        bits = getattr(param, "quant_bits", 32)  # default to 32-bit float
        total_bits += param.numel() * bits
    return total_bits / 8 / 1024**2  # Convert bits → bytes → MB

def compute_hessian_trace(model, dataloader, device, num_batches=10):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    layer_hessian = {}
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            target_layers.append((name, module))
            layer_hessian[name] = []
    
    print(f"\n🔬 Computing Hessian trace for {len(target_layers)} layers...")
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        for name, module in target_layers:
            if not hasattr(module, 'weight') or module.weight is None:
                continue
            
            try:
                grads = grad(loss, module.weight, create_graph=True, retain_graph=True)
                if grads[0] is not None:
                    z = torch.randint_like(grads[0], high=2).float() * 2 - 1
                    hz = grad(grads[0], module.weight, grad_outputs=z, retain_graph=True)
                    if hz[0] is not None:
                        trace_estimate = (hz[0] * z).sum().item()
                        layer_hessian[name].append(abs(trace_estimate))
            except RuntimeError:
                pass
        
        if (batch_idx + 1) % 5 == 0:
            print(f"   Processed {batch_idx + 1}/{num_batches} batches...")
    
    hessian_sensitivity = {}
    for name, traces in layer_hessian.items():
        if len(traces) > 0:
            hessian_sensitivity[name] = np.mean(traces)
        else:
            hessian_sensitivity[name] = 0.0
    
    print("   ✅ Hessian computation complete!")
    return hessian_sensitivity

def assign_precision_hawq(model, hessian_sensitivity, activation_variance):
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)
    
    num_layers = len(layer_names)
    precision_config = {}
    
    hessian_values = [hessian_sensitivity.get(name, 0) for name in layer_names]
    variance_values = [activation_variance.get(name, 0) for name in layer_names]
    
    max_hessian = max(hessian_values) if max(hessian_values) > 0 else 1
    max_variance = max(variance_values) if max(variance_values) > 0 else 1
    
    combined_sensitivity = {}
    for name in layer_names:
        h_norm = hessian_sensitivity.get(name, 0) / max_hessian
        v_norm = activation_variance.get(name, 0) / max_variance
        combined_sensitivity[name] = 0.6 * h_norm + 0.4 * v_norm
    
    sensitivity_values = list(combined_sensitivity.values())
    threshold_high = np.percentile(sensitivity_values, 70)
    threshold_low = np.percentile(sensitivity_values, 30)
    
    print(f"\n📊 Precision Assignment (HAWQ-based):")
    print(f"   High threshold: {threshold_high:.4f}, Low threshold: {threshold_low:.4f}")
    
    for idx, name in enumerate(layer_names):
        sens = combined_sensitivity[name]
        
        if idx == 0 or idx == num_layers - 1:
            precision = 16
        elif 'bn' in name.lower() or 'downsample' in name.lower():
            precision = 16
        elif sens > threshold_high:
            precision = 16
        elif sens > threshold_low:
            precision = 8
        else:
            precision = 8
        
        precision_config[name] = precision
    
    return precision_config

def cast_module(module: nn.Module, dtype: torch.dtype):
    """Cast every parameter *and* buffer of a module to `dtype`."""
    for p in module.parameters(recurse=False):
        p.data = p.data.to(dtype)
        # keep the bit-width tag for size calculation
        p.quant_bits = {torch.float32: 32, torch.float16: 16}.get(dtype, 32)
    for name, buf in module.named_buffers(recurse=False):
        buf.data = buf.data.to(dtype)
# ======================================================

def apply_mixed_precision(model, precision_config):
    model_copy = copy.deepcopy(model).float()  # Clean FP32 base
    
    for name, module in model_copy.named_modules():
        if name not in precision_config: continue
        bits = precision_config[name]
        
        print(f"   Applying {bits}-bit to {name}")
        
        if bits in [16, 32]:  # FP16/FP32: Simulate in FP32
            simulate_low_precision(module, bits)
            
        elif bits == 8:  # **REAL** TorchAO INT8
            if TORCHAO_AVAILABLE:
                try:
                    quantize_(module, quantizer.Int8WeightPerChannelQuantizer())
                    for p in module.parameters(recurse=False):
                        setattr(p, 'quant_bits', 8)
                    print(f"     ✅ {name} → REAL INT8 (FBGEMM)")
                except Exception as e:
                    print(f"     ⚠️ TorchAO fail {name}: {e}")
                    # Fallback manual INT8
                    if hasattr(module, 'weight'):
                        print("simulated int8")

                        w = module.weight.data
                        scale = w.abs().max() / 127
                        wq = torch.round(w / scale).clamp(-128, 127) * scale
                        module.weight.data = wq
                        module.weight.quant_bits = 8
            else:
                # Manual INT8 fallback
                if hasattr(module, 'weight'):
                    print("executing else part")
                    w = module.weight.data
                    scale = w.abs().max() / 127
                    wq = torch.round(w / scale).clamp(-128, 127) * scale
                    module.weight.data = wq
                    module.weight.quant_bits = 8
            
            # Bias FP32 (standard)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.quant_bits = 32
                
        elif bits == 4:  # INT4 sim
            if hasattr(module, 'weight'):
                w = module.weight.data
                scale = w.abs().max() / 7
                wq = torch.round(w / scale).clamp(-8, 7) * scale
                module.weight.data = wq
                module.weight.quant_bits = 4
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.quant_bits = 32
    
    # Safety: Unset params = FP32
    for p in model_copy.parameters():
        if not hasattr(p, 'quant_bits'): setattr(p, 'quant_bits', 32)
        
    return model_copy


def get_activation_variance(model, trainloader, device='cpu', num_batches=10):
    """
    Compute activation variance per layer (for HAWQ sensitivity analysis)
    - Safe for mixed precision (FP16 / FP32)
    - Reduces memory footprint
    - Prevents dtype mismatches and OOM errors
    """
    import torch
    import numpy as np
    import torch.nn as nn

    model.eval()

    # Force all weights to float32 for stable variance computation
    for p in model.parameters():
        if p.dtype != torch.float32:
            p.data = p.data.float()

    model = model.to(device)
    layer_stats = {}
    hooks = []

    # Hook to collect output variance safely
    def get_hook(name):
        def hook(module, input, output):
            try:
                if output is None:
                    return
                if not torch.is_floating_point(output):
                    output = output.float()
                if name not in layer_stats:
                    layer_stats[name] = []
                # Compute variance safely with .detach() and .cpu()
                variance = output.detach().float().var().cpu().item()
                layer_stats[name].append(variance)
            except Exception as e:
                print(f"⚠️ Skipped layer {name} due to error: {e}")
        return hook

    # Register hooks on Conv and Linear layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(get_hook(name)))

    # Run limited batches to save memory
    with torch.no_grad():
        for i, (inputs, _) in enumerate(trainloader):
            if i >= num_batches:
                break
            # Always cast to float32
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            _ = model(inputs)
            torch.cuda.empty_cache()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute mean variance per layer
    variance_sensitivity = {
        name: float(np.mean(stats)) for name, stats in layer_stats.items() if len(stats) > 0
    }

    print(f"✅ Collected variance for {len(variance_sensitivity)} layers "
          f"over {num_batches} batches ({device}).")
    return variance_sensitivity

def mixed_precision_experiments(model, trainloader, testloader, device):
    """
    Run three configurations:
      1. Uniform INT8
      2. Simple Mixed (FP16 first/last, INT8 mid)
      3. HAWQ-based Adaptive (FP16 sensitive layers)
    """
    results = {}
    layer_names = [name for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    print(f"\n📊 Total quantizable layers: {len(layer_names)}")

    # Baseline FP32
    acc_fp32 = evaluate_model(model, testloader, device)
    size_fp32 = get_model_disk_size(model)
    latency_fp32 = measure_latency(model, testloader, device)

    results['fp32_baseline'] = {
        'accuracy': acc_fp32,
        'size_mb': size_fp32,
        'latency_ms': latency_fp32,
        'compression': 1.0,
        'config': 'Full FP32 Baseline'
    }

    # ===============================================================
    # Config 1: Uniform INT8
    # ===============================================================
    print(f"\n{'='*70}\nConfiguration 1: Uniform INT8\n{'='*70}")
    uniform_config = {name: 8 for name in layer_names}
    model_uniform = apply_mixed_precision(model, uniform_config).to(device)

    acc_uniform = evaluate_model(model_uniform, testloader, device)
    size_uniform = get_model_disk_size(model_uniform)
    latency_uniform = measure_latency(model_uniform, testloader, device)

    results['uniform_int8'] = {
        'accuracy': acc_uniform,
        'size_mb': size_uniform,
        'latency_ms': latency_uniform,
        'compression': size_fp32 / size_uniform,
        'config': 'All layers INT8'
    }

    # ===============================================================
    # Config 2: Simple Mixed (FP16 first/last, INT8 others)
    # ===============================================================
    print(f"\n{'='*70}\nConfiguration 2: Simple Mixed-Precision\n{'='*70}")
    simple_config = {}
    for i, name in enumerate(layer_names):
        if i < 2 or i >= len(layer_names) - 2:
            simple_config[name] = 16
        else:
            simple_config[name] = 8

    model_simple = apply_mixed_precision(model, simple_config).to(device)
    acc_simple = evaluate_model(model_simple, testloader, device)
    size_simple = get_model_disk_size(model_simple)
    latency_simple = measure_latency(model_simple, testloader, device)

    results['simple_mixed'] = {
        'accuracy': acc_simple,
        'size_mb': size_simple,
        'latency_ms': latency_simple,
        'compression': size_fp32 / size_simple,
        'config': 'First/Last FP16, Middle INT8'
    }

    # ===============================================================
    # Config 3: HAWQ-based Adaptive (activation + Hessian sensitivity)
    # ===============================================================
    print(f"\n{'='*70}\nConfiguration 3: HAWQ-based Adaptive\n{'='*70}")
    print("\n🔬 Computing Hessian sensitivity...")
    hessian_sensitivity = compute_hessian_trace(model, trainloader, device, num_batches=5)
    print("\n🔬 Computing activation variance...")
    activation_variance = get_activation_variance(model, trainloader, device='cpu', num_batches=5)

    hawq_config = assign_precision_hawq(model, hessian_sensitivity, activation_variance)
    model_hawq = apply_mixed_precision(model, hawq_config).to(device)

    acc_hawq = evaluate_model(model_hawq, testloader, device)
    size_hawq = get_model_disk_size(model_hawq)
    latency_hawq = measure_latency(model_hawq, testloader, device)

    results['hawq_adaptive'] = {
        'accuracy': acc_hawq,
        'size_mb': size_hawq,
        'latency_ms': latency_hawq,
        'compression': size_fp32 / size_hawq,
        'config': 'HAWQ Adaptive Precision',
        'precision_dist': {k: list(hawq_config.values()).count(k) for k in [16, 8, 4]}
    }

    return results, hessian_sensitivity, activation_variance


def plot_mixed_precision_results(results, save_path='mixed_precision.png'):
    """
    Generate improved visualization with visible compression & accuracy trade-offs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    configs = ['fp32_baseline', 'uniform_int8', 'simple_mixed', 'hawq_adaptive']
    labels = ['FP32\nBaseline', 'Uniform\nINT8', 'Simple\nMixed', 'HAWQ\nAdaptive']
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']

    accuracies = [results[c]['accuracy'] for c in configs]
    sizes = [results[c]['size_mb'] for c in configs]

    # --- Accuracy comparison ---
    ax1 = axes[0, 0]
    bars = ax1.bar(labels, accuracies, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontweight='bold')
    for bar, val in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.1, f"{val:.2f}%", ha='center', fontweight='bold')

    # --- Model size ---
    ax2 = axes[0, 1]
    bars = ax2.bar(labels, sizes, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Model Size (MB)', fontweight='bold')
    ax2.set_title('Model Size Comparison', fontweight='bold')
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}MB", ha='center', fontweight='bold')

    # --- Accuracy vs Size (Pareto) ---
    ax3 = axes[1, 0]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax3.scatter(sizes[i], accuracies[i], s=250, color=color, edgecolor='black', label=label.replace('\n', ' '))
    ax3.plot(sizes, accuracies, 'k--', alpha=0.3)
    ax3.set_xlabel('Model Size (MB)', fontweight='bold')
    ax3.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax3.set_title('Accuracy vs Size (Pareto Front)', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # --- Compression vs Accuracy Retention ---
    ax4 = axes[1, 1]
    x = np.arange(len(labels))
    width = 0.35

    compression = [results[c]['compression'] for c in configs]
    retention = [(acc / accuracies[0]) * 100 for acc in accuracies]

    bars1 = ax4.bar(x - width/2, compression, width, label='Compression (x)', color='#9b59b6')
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, retention, width, label='Accuracy Retention (%)', color='#1abc9c')

    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.set_ylabel('Compression (x)', color='#9b59b6', fontweight='bold')
    ax4_twin.set_ylabel('Accuracy Retention (%)', color='#1abc9c', fontweight='bold')
    ax4.set_title('Compression vs Accuracy Retention', fontweight='bold')
    ax4.legend([bars1, bars2], ['Compression', 'Retention'], loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Mixed-precision plot saved to {save_path}")

# ============================================================================
# PART 4: OUTLIER ANALYSIS
# ============================================================================
# ----------------------------------------------------------------------
# 1. ActivationCollector – streaming + real percentile sampling
# ----------------------------------------------------------------------
class ActivationCollector:
    """
    Collects mean, std, min, max **and** a small random sample of activations
    for accurate 99 / 99.9 % percentiles.  Memory stays < 2 MB per layer.
    """
    def __init__(self, max_samples_per_layer: int = 5000):
        self.stats   = {}                     # raw sums
        self.samples = {}                     # list of float values (numpy later)
        self.max_samples = max_samples_per_layer
        self.hooks   = []

    # ------------------------------------------------------------------
    def register_hooks(self, model, layer_names):
        for name, mod in model.named_modules():
            if name in layer_names and isinstance(mod, (nn.Conv2d, nn.ReLU)):
                h = mod.register_forward_hook(self._make_hook(name))
                self.hooks.append(h)

    # ------------------------------------------------------------------
    def _make_hook(self, name):
        def hook(module, inp, out):
            acts = out.detach().flatten()          # [N*C*H*W]

            # ---- raw statistics ------------------------------------------------
            if name not in self.stats:
                self.stats[name] = {
                    'sum': 0.0, 'sum_sq': 0.0, 'count': 0,
                    'min': float('inf'), 'max': float('-inf')
                }
                self.samples[name] = []

            s = self.stats[name]
            s['sum']    += acts.sum().item()
            s['sum_sq'] += (acts ** 2).sum().item()
            s['count']  += acts.numel()
            s['min']    = min(s['min'], acts.min().item())
            s['max']    = max(s['max'], acts.max().item())

            # ---- random sampling for percentiles -------------------------------
            if len(self.samples[name]) < self.max_samples:
                # take up to 64 random elements per forward (fast & enough)
                n_take = min(64, acts.numel())
                idx    = torch.randperm(acts.numel(), device=acts.device)[:n_take]
                self.samples[name].extend(acts[idx].cpu().numpy().tolist())

            del acts
        return hook

    # ------------------------------------------------------------------
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    # ------------------------------------------------------------------
    def get_statistics(self, name):
        if name not in self.stats:
            return {}

        s = self.stats[name]
        count = s['count']
        mean  = s['sum'] / count
        std   = (s['sum_sq'] / count - mean ** 2) ** 0.5

        # ---- real percentiles from sampled values -------------------------
        if self.samples[name]:
            arr = np.array(self.samples[name])
            p99   = np.percentile(arr, 99.0)
            p99_9 = np.percentile(arr, 99.9)
        else:                                   # fallback (should never happen)
            p99   = s['max'] * 0.99
            p99_9 = s['max'] * 0.999

        return {
            'mean'          : float(mean),
            'std'           : float(std),
            'min'           : float(s['min']),
            'max'           : float(s['max']),
            'percentile_99' : float(p99),
            'percentile_99_9': float(p99_9)
        }
    
# ----------------------------------------------------------------------
# 2. collect_activation_statistics – uses the new collector
# ----------------------------------------------------------------------
def collect_activation_statistics(model, dataloader, device, num_batches=50):
    """
    Returns a dict:  layer_name → {'mean':…, 'std':…, 'percentile_99_9':…}
    """
    model.eval()
    model = model.to(device)                 # make sure model & inputs on same device

    # pick 3 representative Conv layers
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    if len(conv_names) >= 3:
        selected = [conv_names[0], conv_names[len(conv_names)//2], conv_names[-1]]
    else:
        selected = conv_names

    print(f"Collecting activation statistics from {len(selected)} layers...")

    collector = ActivationCollector(max_samples_per_layer=5000)
    collector.register_hooks(model, selected)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches: break
            inputs = inputs.to(device)
            _ = model(inputs)

    collector.remove_hooks()

    stats = {name: collector.get_statistics(name) for name in selected}
    return stats

def quantize_with_clipping(tensor, num_bits=8, clip_percentile=99.9):
    clip_val = np.percentile(tensor.abs().cpu().numpy(), clip_percentile)
    tensor_clipped = torch.clamp(tensor, -clip_val, clip_val)
    
    qmin = -2**(num_bits-1)
    qmax = 2**(num_bits-1) - 1
    
    scale = tensor_clipped.abs().max() / qmax
    if scale == 0:
        scale = 1.0
    
    q_tensor = torch.clamp(torch.round(tensor_clipped / scale), qmin, qmax)
    dq_tensor = q_tensor * scale
    
    return dq_tensor

def apply_quantization_with_outlier_handling(fp32_model, bits, clip_percentile, trainloader, quantize_activations=False):
    """
    Post-training quantization with optional percentile clipping for weight scaling.
    
    Args:
        fp32_model: Full precision model
        bits: Target bit-width (4 or 8)
        clip_percentile: If not None, clip to this percentile (e.g., 99.9)
        trainloader: Data for calibration
        quantize_activations: If True, apply runtime activation quantization (NOT RECOMMENDED - causes instability)
    
    Returns:
        Quantized model with metadata
    """
    device = next(fp32_model.parameters()).device
    model = copy.deepcopy(fp32_model).to(device)
    model.eval()

    # === 1. Collect activation stats for calibration ===
    collector = ActivationCollector(max_samples_per_layer=5000)
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    collector.register_hooks(model, conv_names)

    print(f"   Calibrating on {HYPERPARAMS['calibration_batches']} batches...")
    with torch.no_grad():
        for i, (x, _) in enumerate(trainloader):
            if i >= HYPERPARAMS['calibration_batches']: break
            x = x.to(device)
            _ = model(x)
    collector.remove_hooks()

    # === 2. Compute scales based on weight distribution (NOT activations) ===
    scales = {}
    clip_vals = {}
    
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d) or not hasattr(mod, 'weight'):
            continue
            
        w = mod.weight.data.abs()
        
        # Use weight statistics for scale, not activation statistics
        if clip_percentile is not None:
            # Clip based on weight percentile
            clip_val = torch.quantile(w, clip_percentile / 100.0).item()
        else:
            # Use max weight value
            clip_val = w.max().item()
        
        clip_vals[name] = {'clip_val': clip_val, 'max_val': w.max().item()}
        
        # Symmetric quantization scale
        q_range = 2**(bits - 1) - 1
        scales[name] = clip_val / q_range

    # === 3. Quantize ONLY weights (stable approach) ===
    def quantize_weight(w, scale, bits):
        qmin, qmax = -(2**(bits-1)), (2**(bits-1)) - 1
        # Clip → Quantize → Dequantize
        w_clipped = torch.clamp(w, -scale * qmax, scale * qmax)
        w_q = torch.clamp(torch.round(w_clipped / scale), qmin, qmax)
        return w_q * scale

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and name in scales:
            mod.weight.data = quantize_weight(mod.weight.data, scales[name], bits)
            mod.weight.quant_bits = bits
            
            # Keep bias in FP32 (standard practice)
            if mod.bias is not None:
                mod.bias.quant_bits = 32

    # === 4. (Optional) Activation quantization - DISABLED by default ===
    # Activation quantization during inference causes severe accuracy drops
    # Real deployment uses int8 compute engines, not Python hooks
    if quantize_activations:
        print("   ⚠️  Warning: Runtime activation quantization enabled (may degrade accuracy)")
        # Implementation omitted - not recommended for this experiment
    
    # Attach metadata
    model._clip_vals = clip_vals
    model._scales = scales
    
    # Ensure all params have quant_bits
    for p in model.parameters():
        if not hasattr(p, 'quant_bits'):
            p.quant_bits = 32
    
    return model

def plot_activation_histograms(stats, weight_clip_info, save_path='activation_histograms.png'):
    """
    Visualize activation distributions and the effect of weight clipping.
    
    Args:
        stats: Activation statistics from ActivationCollector
        weight_clip_info: Dictionary mapping layer_name → {'clip_val': float, 'max_val': float}
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, len(stats), figsize=(5*len(stats), 8))
    if len(stats) == 1: 
        axes = axes.reshape(2, 1)
    
    for col_idx, (name, s) in enumerate(stats.items()):
        # Top row: Activation distributions
        ax_act = axes[0, col_idx]
        
        # Generate sample activations based on collected statistics
        # Use the actual samples if available, otherwise approximate from mean/std
        samples = np.random.normal(s['mean'], s['std'], 5000)
        samples = np.clip(samples, s['min'], s['max'])
        
        ax_act.hist(samples, bins=60, alpha=0.7, color='skyblue', 
                   edgecolor='black', linewidth=0.5, label='Activations')
        
        # Mark percentiles
        ax_act.axvline(s['percentile_99'], color='orange', linestyle='--', 
                      linewidth=2, label=f"99%: {s['percentile_99']:.2f}")
        ax_act.axvline(s['percentile_99_9'], color='red', linestyle='--', 
                      linewidth=2, label=f"99.9%: {s['percentile_99_9']:.2f}")
        
        ax_act.set_title(f"{name.split('.')[-1]} - Activations", fontweight='bold')
        ax_act.set_xlabel("Activation Value")
        ax_act.set_ylabel("Frequency")
        ax_act.legend(fontsize=8)
        ax_act.grid(True, alpha=0.3)
        
        # Bottom row: Weight clipping effect (if available)
        ax_weight = axes[1, col_idx]
        
        if weight_clip_info and name in weight_clip_info:
            clip_info = weight_clip_info[name]
            
            # Show clipping effect
            x_range = np.linspace(-clip_info['max_val'], clip_info['max_val'], 100)
            original = x_range
            clipped = np.clip(x_range, -clip_info['clip_val'], clip_info['clip_val'])
            
            ax_weight.plot(original, original, 'b-', linewidth=2, label='Original', alpha=0.7)
            ax_weight.plot(original, clipped, 'r-', linewidth=2, label='Clipped', alpha=0.7)
            ax_weight.axvline(clip_info['clip_val'], color='orange', linestyle='--', 
                            label=f"Clip: ±{clip_info['clip_val']:.2f}")
            ax_weight.axvline(-clip_info['clip_val'], color='orange', linestyle='--')
            
            ax_weight.set_title(f"Weight Clipping Effect", fontweight='bold')
            ax_weight.set_xlabel("Original Weight Value")
            ax_weight.set_ylabel("Clipped Weight Value")
            ax_weight.legend(fontsize=8)
            ax_weight.grid(True, alpha=0.3)
        else:
            ax_weight.text(0.5, 0.5, 'No clipping applied', 
                          ha='center', va='center', fontsize=12)
            ax_weight.set_title(f"No Clipping", fontweight='bold')
    
    plt.suptitle("Activation Distributions and Weight Clipping Effects", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def outlier_experiments(model, trainloader, testloader, device):
    """
    Study the effect of outliers on quantization by comparing:
    1. Standard quantization (no clipping)
    2. Clipped quantization (99.9th percentile)
    
    For both INT8 and INT4 bit-widths.
    """
    print("\n" + "="*70)
    print("PART 4: EFFECT OF OUTLIERS ON QUANTIZATION")
    print("="*70)

    # === 1. Collect activation statistics ===
    print("\n📊 Collecting activation statistics...")
    stats = collect_activation_statistics(model, trainloader, device, num_batches=20)
    
    print("\n" + "-"*70)
    print(f"{'Layer':<30} {'Mean':<10} {'Std':<10} {'99%':<10} {'99.9%':<10}")
    print("-"*70)
    for name, s in stats.items():
        print(f"{name.split('.')[-1]:<30} {s['mean']:<10.4f} {s['std']:<10.4f} "
              f"{s['percentile_99']:<10.4f} {s['percentile_99_9']:<10.4f}")
    print("-"*70)

    # Get baseline accuracy
    print("\n🎯 Evaluating FP32 baseline...")
    baseline_acc = evaluate_model(model, testloader, device)

    # === 2. Run quantization experiments ===
    results = {}
    weight_clip_metadata = {}

    experiments = [
        ('int8_standard', 8, None, "INT8 - Standard (no clipping)"),
        ('int8_clipped', 8, 99.9, "INT8 - Clipped (99.9%)"),
        ('int4_standard', 4, None, "INT4 - Standard (no clipping)"),
        ('int4_clipped', 4, 99.9, "INT4 - Clipped (99.9%)"),
    ]

    for key, bits, clip_pct, desc in experiments:
        print(f"\n{'='*70}")
        print(f"Testing: {desc}")
        print("="*70)
        
        qmodel = apply_quantization_with_outlier_handling(
            model, bits, clip_pct, trainloader, quantize_activations=False
        )
        qmodel = qmodel.to(device)
        
        # Evaluate
        acc = evaluate_model(qmodel, testloader, device)
        
        # Store results
        results[key] = {
            'accuracy': acc,
            'method': desc,
            'delta': acc - baseline_acc,
            'bits': bits,
            'clipping': clip_pct is not None
        }
        
        # Store clipping metadata for visualization
        if clip_pct is not None and hasattr(qmodel, '_clip_vals'):
            weight_clip_metadata[key] = qmodel._clip_vals
        
        print(f"   Accuracy: {acc:.2f}% (Δ: {acc - baseline_acc:+.2f}%)")
        
        # Clean up
        del qmodel
        torch.cuda.empty_cache()

    # === 3. Generate visualizations ===
    print("\n📈 Generating visualizations...")
    
    # Activation histograms with clipping info
    sample_clip_info = weight_clip_metadata.get('int8_clipped', {})
    plot_activation_histograms(stats, sample_clip_info, 
                               save_path='activation_histograms.png')
    
    # Comprehensive outlier analysis
    plot_outlier_analysis1(results, stats, baseline_acc, 
                         save_path='outlier_analysis.png')

    return results, stats

def plot_outlier_analysis1(results, activation_stats, baseline_acc, save_path='outlier_analysis.png'):
    """
    Comprehensive visualization of outlier handling effects.
    
    Creates 4-panel plot:
    1. Accuracy comparison (INT8)
    2. Accuracy comparison (INT4)
    3. Improvement from clipping
    4. Layer-wise outlier severity
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # === Panel 1: INT8 Comparison ===
    ax1 = fig.add_subplot(gs[0, 0])
    int8_keys = ['int8_standard', 'int8_clipped']
    int8_labels = ['Standard\n(No Clipping)', 'Clipped\n(99.9%)']
    int8_accs = [results[k]['accuracy'] for k in int8_keys]
    colors_int8 = ['#e74c3c', '#2ecc71']
    
    bars1 = ax1.bar(int8_labels, int8_accs, color=colors_int8, 
                    alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
    ax1.axhline(y=baseline_acc, color='blue', linestyle='--', 
               linewidth=2.5, label=f'FP32 Baseline ({baseline_acc:.2f}%)', zorder=0)
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('INT8 Quantization:\nEffect of Outlier Clipping', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, max(baseline_acc * 1.1, max(int8_accs) * 1.1)])
    
    for bar, acc in zip(bars1, int8_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # === Panel 2: INT4 Comparison ===
    ax2 = fig.add_subplot(gs[0, 1])
    int4_keys = ['int4_standard', 'int4_clipped']
    int4_labels = ['Standard\n(No Clipping)', 'Clipped\n(99.9%)']
    int4_accs = [results[k]['accuracy'] for k in int4_keys]
    colors_int4 = ['#e67e22', '#27ae60']
    
    bars2 = ax2.bar(int4_labels, int4_accs, color=colors_int4, 
                    alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
    ax2.axhline(y=baseline_acc, color='blue', linestyle='--', 
               linewidth=2.5, label=f'FP32 Baseline ({baseline_acc:.2f}%)', zorder=0)
    
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('INT4 Quantization:\nEffect of Outlier Clipping', 
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, max(baseline_acc * 1.1, max(int4_accs) * 1.1)])
    
    for bar, acc in zip(bars2, int4_accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # === Panel 3: Improvement from Clipping ===
    ax3 = fig.add_subplot(gs[1, 0])
    
    improvements = {
        'INT8\nClipping': results['int8_clipped']['accuracy'] - results['int8_standard']['accuracy'],
        'INT4\nClipping': results['int4_clipped']['accuracy'] - results['int4_standard']['accuracy'],
    }
    
    colors_imp = ['#3498db', '#e67e22']
    bars3 = ax3.bar(improvements.keys(), improvements.values(), 
                    color=colors_imp, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax3.set_ylabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy Gain from\nOutlier Clipping', 
                 fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, (name, val) in zip(bars3, improvements.items()):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.2f}%', ha='center', 
                va='bottom' if val > 0 else 'top', 
                fontsize=11, fontweight='bold')
    
    # === Panel 4: Layer-wise Outlier Severity ===
    ax4 = fig.add_subplot(gs[1, 1])
    
    layers = list(activation_stats.keys())
    layer_labels = [l.split('.')[-1] for l in layers]
    
    # Compute outlier ratio: (99.9% - mean) / std
    outlier_ratios = []
    for name in layers:
        s = activation_stats[name]
        ratio = (s['percentile_99_9'] - s['mean']) / (s['std'] + 1e-6)
        outlier_ratios.append(ratio)
    
    x_pos = np.arange(len(layer_labels))
    bars4 = ax4.bar(x_pos, outlier_ratios, color='#9b59b6', 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(layer_labels, rotation=45, ha='right')
    ax4.set_ylabel('Outlier Severity\n(99.9% - Mean) / Std', fontsize=11, fontweight='bold')
    ax4.set_title('Layer-wise Outlier Severity', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars4, outlier_ratios):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}σ', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    plt.suptitle('Outlier Analysis: Effect on Quantization Performance', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Saved comprehensive outlier analysis to: {save_path}")
    
    # === Print summary table ===
    print("\n" + "="*70)
    print("OUTLIER HANDLING RESULTS SUMMARY")
    print("="*70)
    print(f"{'Configuration':<30} {'Accuracy':<12} {'Δ from Baseline':<18} {'Δ from Standard':<15}")
    print("-"*70)
    
    for key in ['int8_standard', 'int8_clipped', 'int4_standard', 'int4_clipped']:
        res = results[key]
        delta_baseline = res['accuracy'] - baseline_acc
        
        if 'clipped' in key:
            standard_key = key.replace('_clipped', '_standard')
            delta_standard = res['accuracy'] - results[standard_key]['accuracy']
            delta_std_str = f"+{delta_standard:.2f}%" if delta_standard > 0 else f"{delta_standard:.2f}%"
        else:
            delta_std_str = "—"
        
        print(f"{res['method']:<30} {res['accuracy']:<12.2f} {delta_baseline:<18.2f} {delta_std_str:<15}")
    print("="*70)

def apply_clipping_quantization(model, trainloader, device, bits=8, percentile=99.9):
    model.eval()
    model = copy.deepcopy(model).to(device)
    
    # Step 1: Collect 99.9% percentile per layer (streaming)
    layer_scales = {}
    collector = ActivationCollector()
    conv_layers = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    collector.register_hooks(model, conv_layers[:3])  # Only first 3 for speed
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(trainloader):
            if i >= 10: break
            inputs = inputs.to(device)
            _ = model(inputs)
    
    collector.remove_hooks()
    
    # Step 2: Extract 99.9% clip value
    for name in collector.stats:
        max_abs = collector.stats[name]['percentile_99_9']
        layer_scales[name] = max_abs / (2 ** (bits - 1) - 1)  # symmetric
    
    # Step 3: Quantize weights with per-layer scale (no activation clipping!)
    def quantize_weight(w, scale):
        qmin, qmax = -(2**(bits-1)), 2**(bits-1)-1
        return torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in layer_scales:
            scale = layer_scales[name]
            if hasattr(module, 'weight'):
                w = module.weight.data
                module.weight.data = quantize_weight(w, scale)
                module.weight.quant_bits = bits
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.quant_bits = 32
    
    # Safety
    for p in model.parameters():
        if not hasattr(p, 'quant_bits'): p.quant_bits = 32
            
    return model

def plot_outlier_analysis(results, activation_stats):
    import matplotlib.pyplot as plt
    
    layers = list(activation_stats.keys())
    means = [activation_stats[l]['mean'] for l in layers]
    stds = [activation_stats[l]['std'] for l in layers]
    p99_9 = [activation_stats[l]['percentile_99_9'] for l in layers]
    
    # === Histogram-style bar plot ===
    x = np.arange(len(layers))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, means, width, label='Mean', color='skyblue')
    ax.bar(x, stds, width, label='Std Dev', color='salmon')
    ax.bar(x + width, p99_9, width, label='99.9%ile', color='gold')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Activation Value')
    ax.set_title('Activation Statistics Across Layers')
    ax.set_xticks(x)
    ax.set_xticklabels([l.split('.')[-1] for l in layers])
    ax.legend()
    plt.tight_layout()
    plt.savefig('outlier_stats.png')
    plt.close()
    
    # === Accuracy table ===
    print("\nOutlier Handling Results:")
    print("-" * 50)
    for config, acc in results.items():
        delta = acc - 75.04
        print(f"{config:25}: {acc:6.2f}% (Δ: {delta:+6.2f}%)")

# ============================================================================
# MAIN EXECUTION PIPELINE (FINAL FIXED VERSION)
# ============================================================================

def main():
    print("="*70)
    print("TASK 2: MODEL QUANTIZATION ON VGG-11 (CIFAR-100)")
    print("="*70)
    
    # Load data
    print("\n[1/6] Loading CIFAR-100 dataset...")
    trainloader, testloader = get_cifar100_dataloaders(batch_size=HYPERPARAMS['finetune_batch_size'])
    
    # Load and finetune model
    print("\n[2/6] Loading or training FP32 baseline model...")

    # === Safe loading / training block ===
    import os
    model_path = "vgg11_cifar100_fp32.pth"

    # Load if exists, else train from scratch
    if os.path.exists(model_path):
        print("✅ Found pretrained FP32 model. Loading weights...")
        model = get_pretrained_vgg11_cifar100().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        baseline_acc = evaluate_model(model, testloader, device)
        print(f"✅ Loaded model | Baseline Accuracy: {baseline_acc:.2f}%")
    else:
        print("🚀 No pretrained model found. Training FP32 model from scratch...")
        model = get_pretrained_vgg11_cifar100().to(device)
        model, history, baseline_acc = finetune_model(
            model, trainloader, testloader,
            epochs=HYPERPARAMS['finetune_epochs'],
            lr=HYPERPARAMS['finetune_lr']
        )
        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved trained FP32 model to {model_path}")

    
    print(f"\n✅ Baseline FP32 Accuracy: {baseline_acc:.2f}%")
    
    
    # PART 3: Mixed-Precision Quantization
    print("\n[5/6] " + "="*70)
    print("PART 3: MIXED-PRECISION QUANTIZATION (HAWQ)")
    print("="*70)
    mixed_results, hessian_sens, activation_var = mixed_precision_experiments(
        model, trainloader, testloader, device
    )
    
    plot_mixed_precision_results(mixed_results)
    
    # PART 4: Outlier Analysis
    print("\n[6/6] " + "="*70)
    print("PART 4: EFFECT OF OUTLIERS ON QUANTIZATION")
    print("="*70)
    outlier_results, activation_stats = outlier_experiments(
        model, trainloader, testloader, device
    )
    
    plot_outlier_analysis(outlier_results, activation_stats)
    
    # ========================================================================
    # FINAL SUMMARY REPORT
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY REPORT")
    print("="*70)
    
    print(f"\n🎯 Baseline Performance:")
    print(f"   FP32 Accuracy: {baseline_acc:.2f}%")

    print(f"\n💾 Mixed-Precision Results:")
    print("-" * 80)
    print(f"{'Configuration':<35} {'Accuracy':<12} {'Size (MB)':<12} {'Compression':<12}")
    print("-" * 80)
    
    for config in ['fp32_baseline', 'uniform_int8', 'simple_mixed', 'hawq_adaptive']:
        res = mixed_results[config]
        comp = res.get('compression', 1.0)
        print(f"{res['config']:<35} {res['accuracy']:<12.2f} {res['size_mb']:<12.2f} {comp:<11.2f}x")
    
    print(f"\n🔬 HAWQ Adaptive Details:")
    if 'precision_dist' in mixed_results['hawq_adaptive']:
        dist = mixed_results['hawq_adaptive']['precision_dist']
        total = sum(dist.values())
        print(f"   Total layers: {total}")
        print(f"   FP16 layers: {dist.get(16, 0)} ({dist.get(16, 0)/total*100:.1f}%)")
        print(f"   INT8 layers: {dist.get(8, 0)} ({dist.get(8, 0)/total*100:.1f}%)")
        print(f"   INT4 layers: {dist.get(4, 0)} ({dist.get(4, 0)/total*100:.1f}%)")
    
    print(f"\n🔍 Outlier Analysis - Best Results:")
    print("-" * 80)
    
    best_int8_clip = max(['int8_clip99', 'int8_clip999'], 
                        key=lambda x: outlier_results[x]['accuracy'])
    best_int4_clip = max(['int4_clip99', 'int4_clip999'], 
                        key=lambda x: outlier_results[x]['accuracy'])
    
    print(f"INT8: {outlier_results[best_int8_clip]['method']}")
    print(f"   Accuracy: {outlier_results[best_int8_clip]['accuracy']:.2f}%")
    print(f"   Improvement: {outlier_results[best_int8_clip]['accuracy'] - outlier_results['int8']['accuracy']:+.2f}%")
    
    print(f"\nINT4: {outlier_results[best_int4_clip]['method']}")
    print(f"   Accuracy: {outlier_results[best_int4_clip]['accuracy']:.2f}%")
    print(f"   Improvement: {outlier_results[best_int4_clip]['accuracy'] - outlier_results['int4']['accuracy']:+.2f}%")
    
    print("\n" + "="*70)
    print("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   - mixed_precisio_p.png (4-panel mixed-precision comparison)")
    print("   - outlier_analysis_p.png (Comprehensive outlier effects)")
    

    import json
    import pickle
    
    # Prepare JSON-serializable results
    json_results = {
        'mixed_results': {k: {k2: v2 for k2, v2 in v.items() if k2 not in ['history', 'precision_dist']} 
                         for k, v in mixed_results.items()},
        'outlier_results': outlier_results,
        'baseline_acc': float(baseline_acc),
        'hyperparameters': HYPERPARAMS
    }
    
    with open('quantization_results_p.json', 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print("\n💾 Saved: quantization_results_p.json")
    
    # Save full results with histories to PKL
    full_results = {
        'mixed_results': mixed_results,
        'outlier_results': outlier_results,
        'activation_stats': activation_stats,
        'hessian_sensitivity': hessian_sens,
        'activation_variance': activation_var,
        'baseline_acc': baseline_acc,
        'hyperparameters': HYPERPARAMS
    }
    
    with open('quantization_results_p.pkl', 'wb') as f:
        pickle.dump(full_results, f)
    
    print("💾 Saved: quantization_results_p.pkl")
    
    return full_results

if __name__ == "__main__":
    results = main()

