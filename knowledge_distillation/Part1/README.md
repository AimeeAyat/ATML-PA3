# Knowledge Distillation Assignment - Part 1: Logit Matching

## Project Structure

```
KD_Assignment/
├── utils.py                    # Common utilities for all experiments
├── logit_matching.py          # Basic Logit Matching implementation
├── label_smoothing.py         # Label Smoothing implementation
├── decoupled_kd.py            # Decoupled KD implementation
├── comparison_part1.py        # Comparison script (runs all three)
├── checkpoints/               # Saved model checkpoints
├── results/                   # Results, plots, and analysis
└── README.md                  # This file
```

## Installation

Before running, install required packages:

```bash
# Install PyTorch with CUDA support (for RTX 5000 with 16GB VRAM)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CPU (slower):
pip install torch torchvision torchaudio

# Install other dependencies
pip install matplotlib numpy
```

## Quick Start

### Option 1: Run All Three Methods Together (Recommended)

```bash
cd KD_Assignment
python comparison_part1.py
```

This will:
1. Train Logit Matching (LM)
2. Train Label Smoothing (LS)
3. Train Decoupled KD (DKD)
4. Generate comparison plots and analysis
5. Save results to `./results/`

### Option 2: Run Individual Methods

```bash
# Basic Logit Matching
python logit_matching.py

# Label Smoothing
python label_smoothing.py

# Decoupled Knowledge Distillation
python decoupled_kd.py
```

## Method Overview

### 1. Basic Logit Matching (LM)

**How it works:**
- Student learns to match teacher's softened probability distributions
- Uses temperature-scaled softmax: `softmax(logits / T)`
- Loss combines:
  - **Distillation Loss**: KL divergence between teacher and student distributions
  - **Cross-Entropy Loss**: Standard CE with hard labels
- Balanced by parameter `alpha` (default 0.5)

**Key Formula:**
```
Loss = alpha * KL(T_softened || S_softened) + (1-alpha) * CE(S, ground_truth)
```

**Hyperparameters:**
- `temperature`: Higher = softer distributions (default 4.0)
- `alpha`: Balance between KD and CE loss (default 0.5)
- `learning_rate`: 0.1
- `weight_decay`: 5e-4

---

### 2. Label Smoothing (LS)

**How it works:**
- Regularization technique that prevents overconfidence
- Replaces one-hot labels with smoothed targets
- Example: Instead of `[0, 1, 0]`, use `[0.05, 0.9, 0.05]`
- Does NOT directly use teacher (implicit knowledge transfer)

**Key Formula:**
```
Smoothed_Label = (1 - epsilon) * one_hot + epsilon / (num_classes - 1)
```

**Hyperparameters:**
- `smoothing`: Distribution of probability to other classes (default 0.1)
- `learning_rate`: 0.1
- `weight_decay`: 5e-4

**Intuition:**
- Prevents model from being 100% confident
- Encourages learning of similar classes
- Acts as implicit regularization

---

### 3. Decoupled Knowledge Distillation (DKD)

**How it works:**
- Separates knowledge transfer into two components:
  - **TCKD (Target Class KD)**: Teach student to emphasize correct class
  - **NCKD (Non-Target Class KD)**: Teach student to differentiate wrong classes
- More nuanced than basic logit matching
- Better control over what knowledge is transferred

**Key Formulas:**
```
TCKD Loss: -sum(teacher_target * log(student_target))
NCKD Loss: -sum(teacher_non_target * log(student_non_target))
Total Loss = alpha * TCKD + beta * NCKD + CE_weight * CE(S, ground_truth)
```

**Hyperparameters:**
- `temperature`: 4.0
- `alpha`: Weight for TCKD (default 1.0)
- `beta`: Weight for NCKD (default 1.0)
- `ce_weight`: Weight for CE loss (default 1.0)
- `learning_rate`: 0.1
- `weight_decay`: 5e-4

---

## Expected Results

All experiments use:
- **Student**: VGG-11 (25M parameters)
- **Teacher**: VGG-16 (pretrained, 38M parameters)
- **Dataset**: CIFAR-100 (100 classes)
- **Epochs**: 200
- **Batch Size**: 128

### Typical Performance Range:
```
Method                  Best Val Acc    Test Acc
─────────────────────────────────────────────────
Logit Matching          ~75-78%         ~74-77%
Label Smoothing         ~72-75%         ~71-74%
Decoupled KD            ~76-79%         ~75-78%
```

> Note: Exact numbers depend on teacher pretraining quality

## GPU Considerations (RTX 5000 - 16GB VRAM)

Your GPU has plenty of memory for this task. Optimization settings:

```python
# Current settings (safe for 16GB):
batch_size = 128
num_workers = 4

# Can be increased if needed:
batch_size = 256  # Faster training
num_workers = 8   # More parallel data loading
```

**Memory usage per training:**
- ~8-10 GB (batch size 128)
- ~12-14 GB (batch size 256)

**Training time estimates:**
- Per epoch: ~2-3 minutes (with GPU)
- Full training (200 epochs): ~7-10 hours per method
- All three methods: ~24-30 hours total

## Output Files

After running, you'll find:

### In `./checkpoints/`:
- `lm_student_best.pth` - Best LM student model
- `ls_student_best.pth` - Best LS student model
- `dkd_student_best.pth` - Best DKD student model

### In `./results/`:
- `kd_comparison_YYYYMMDD_HHMMSS.png` - Comparison plots
- `kd_comparison_YYYYMMDD_HHMMSS.json` - Raw results data
- `kd_comparison_analysis_YYYYMMDD_HHMMSS.txt` - Detailed analysis

## Comparison Plots

The generated plot shows:
1. **Training Accuracy**: How accuracy increases during training
2. **Validation Accuracy**: How well models generalize
3. **Training Loss**: How loss decreases (higher loss at start)
4. **Final Accuracy Bars**: Direct comparison of validation vs test accuracy

## What to Analyze

### 1. Accuracy Comparison
- Which method performs best overall?
- Is the improvement statistically significant?
- What are the trade-offs?

### 2. Training Curves
- Does one method converge faster?
- Does one overfit more (big gap between train and val)?
- Are there oscillations or instability?

### 3. Loss Convergence
- Which method has smoothest loss curve?
- Do losses stabilize or oscillate?
- Any sudden jumps or drops?

### 4. Expected Insights

**Logit Matching (LM):**
- ✓ Direct knowledge transfer from teacher
- ✓ Well-established baseline method
- ✗ May require careful tuning of alpha and temperature

**Label Smoothing (LS):**
- ✓ Simple to implement
- ✓ Acts as regularization
- ✗ Doesn't directly use teacher knowledge
- ✗ Usually underperforms LM

**Decoupled KD (DKD):**
- ✓ More nuanced knowledge transfer
- ✓ Better control over learning process
- ✗ More hyperparameters to tune
- ✓ Usually best performance if tuned well

## Customization

### Modify Training Parameters

Edit the methods in `comparison_part1.py`:

```python
# Change number of epochs
comparison.run_all_experiments(..., num_epochs=100)

# Change batch size in utils.py
train_loader, val_loader, test_loader = get_cifar100_loaders(
    batch_size=256,  # Larger batch
    num_workers=8    # More workers
)
```

### Tune Hyperparameters

In `comparison_part1.py`:

```python
# Logit Matching
lm_kd = LogitMatchingKD(
    temperature=6.0,   # Increase for softer targets
    alpha=0.7,         # More weight on KD loss
    learning_rate=0.05  # Lower for more stable training
)

# Label Smoothing
ls_kd = LabelSmoothingKD(
    smoothing=0.2,     # More aggressive smoothing
    learning_rate=0.1
)

# Decoupled KD
dkd_kd = DecoupledKD(
    temperature=4.0,
    alpha=0.5,         # Different weights for TCKD
    beta=1.5,          # More weight on NCKD
)
```

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size in utils.py
batch_size=64  # Instead of 128
```

### Training is too slow
```python
# Use fewer workers
num_workers=0  # Disable parallel loading

# Or reduce epochs for testing
num_epochs=50
```

### Teacher model not found
Make sure to train the teacher first or ensure `checkpoints/teacher_vgg16.pth` exists

### CUDA not available
```python
# Force CPU usage (slow)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

## References

- [1] Hinton et al., "Distilling the Knowledge in a Neural Network" (Original KD paper)
- [2] Szegedy et al., "Rethinking the Inception Architecture for Computer Vision" (Label Smoothing)
- [3] Zhao et al., "Decoupled Knowledge Distillation" (DKD paper)

## Questions/Next Steps

After Part 1, you'll move to:
- **Part 2**: Implement Hints-based and Contrastive Representation Distillation (CRD)
- **Part 3**: Compare probability distributions between models
- **Part 4**: Visualize learned features with GradCAM
- **Part 5**: Test color invariance with CRD
- **Part 6**: Compare teacher size effects

Good luck! 🚀