# Part 5: Testing Color Invariance with CRD

## Overview

Part 5 analyzes whether **Knowledge Distillation can transfer robustness properties** (specifically color invariance) **without explicit training**. 

**Key Question:** Can a student learn to be color-invariant by mimicking a color-invariant teacher, without seeing color-jittered data itself?

---

## What is Color Invariance?

**Color Invariance** = Model's ability to make correct predictions despite color variations in images.

**Example:**
- Image of a dog: Normal colors → Predicted correctly
- Same dog image: Color-jittered (shifted hues, brightness, etc.) → Should still predict dog

**Why it matters:**
- Real-world images have color variations
- Brightness changes, lighting conditions vary
- Model should be robust, not rely on exact colors

---

## Complete Pipeline

### Stage 1: Original Teacher Evaluation
- Load pre-trained VGG-16 teacher
- Evaluate on normal validation set
- Evaluate on color-jittered validation set
- Measure accuracy drop

### Stage 2: Fine-tune Teacher with Color Jitter
- Train teacher on CIFAR-100 with color jitter augmentations
- Optimize for color invariance
- Evaluate on color-jittered data
- Should have LOWER accuracy drop than original

### Stage 3: CRD Knowledge Distillation
- Use color-invariant teacher from Stage 2
- Distill to VGG-11 student
- **Important:** Student training uses NORMAL augmentations (no color jitter)
- Student should inherit color invariance from teacher

### Stage 4: Compare with Other Methods
- Evaluate SI (independent student)
- Evaluate LM (logit matching student)
- Compare color invariance across methods

### Stage 5: Analysis
- Compare invariance ratios
- Analyze transfer of robustness
- Discuss effectiveness of CRD

---

## Quick Start

### 1. Update Checkpoint Paths

Edit `example_part5_main.py`, update `CHECKPOINT_PATHS`:

```python
CHECKPOINT_PATHS = {
    'original_teacher': './path/to/teacher.pth',
    'si_student': './path/to/si.pth',
    'lm_student': './path/to/lm.pth',
    'crd_student': './path/to/crd.pth',
}
```

### 2. Run Analysis

```bash
# Quick test (faster, fewer epochs)
python example_part5_main.py --quick

# Full analysis
python example_part5_main.py

# Skip teacher fine-tuning (use original)
python example_part5_main.py --skip-finetuning
```

### 3. Review Results

Output files:
- **color_invariance_results.json** - All metrics
- **part5_plots/accuracy_comparison.png** - Normal vs Jittered accuracy
- **part5_plots/invariance_ratio.png** - Invariance comparison

---

## Key Metrics

### Accuracy (Normal vs Jittered)
- **Normal:** Accuracy on standard validation set
- **Jittered:** Accuracy on color-jittered validation set
- **Lower jittered = More variance-sensitive**

### Accuracy Drop
- Drop = Normal Accuracy - Jittered Accuracy
- **Lower drop = More robust**
- **Green bars** (<5%): Excellent
- **Orange bars** (5-10%): Good
- **Red bars** (>10%): Poor

### Invariance Ratio
- Ratio = Jittered Accuracy / Normal Accuracy
- **1.0:** Perfect invariance (same on jittered)
- **0.9+:** Excellent invariance
- **0.8-0.9:** Good invariance
- **<0.8:** Poor invariance

---

## Expected Results

### Original Teacher
```
Normal Accuracy:     ~76%
Jittered Accuracy:   ~60%
Accuracy Drop:       ~16%
Invariance Ratio:    0.79
```
→ Not color-invariant (large drop)

### Color-Invariant Teacher
```
Normal Accuracy:     ~74%
Jittered Accuracy:   ~71%
Accuracy Drop:       ~3%
Invariance Ratio:    0.96
```
→ Learned to ignore color variations

### CRD Student (From Color-Invariant Teacher)
```
Normal Accuracy:     ~72%
Jittered Accuracy:   ~68%
Accuracy Drop:       ~4%
Invariance Ratio:    0.94
```
→ Inherited color invariance!

### Other Methods (For Comparison)
```
SI (Independent):    Invariance ~0.79 (not trained with color invariance)
LM (Logit Matching): Invariance ~0.82 (some transfer)
CRD:                 Invariance ~0.94 (best transfer!)
```

---

## Interpretation

### Success Indicators

✓ **CRD has highest invariance ratio** (closest to 1.0)
✓ **CRD invariance > SI and other methods**
✓ **CRD student invariance approaches teacher invariance**
✓ **Student never saw color-jittered data but still invariant**

### Analysis Points

1. **Knowledge Transfer Success?**
   - Yes: CRD invariance high even without explicit training
   - No: CRD invariance similar to SI

2. **Better than Other Methods?**
   - Compare CRD vs LM vs SI
   - Is CRD significantly better?

3. **Robustness Transfer Mechanism?**
   - Did student learn color-invariant features?
   - Or just follow teacher outputs?

4. **Generalization?**
   - Does it work on other color variations?
   - What about other transformations?

---

## Code Structure

### `part5_color_invariance.py`

**Classes:**
- `ColorJitterTransform` - Color augmentation
- `NormalTransform` - Standard normalization
- `ColorInvarianceAnalyzer` - Main analyzer

**Key Methods:**
- `evaluate_model()` - Test on normal + jittered data
- `fine_tune_teacher()` - Train teacher with color jitter
- `knowledge_distillation_crd()` - CRD distillation pipeline
- `compare_methods()` - Export results
- `plot_comparison()` - Create visualizations

**Helpers:**
- `get_cifar100_loaders()` - Load CIFAR-100 with transforms

### `example_part5_main.py`

**Pipeline:**
1. Load original teacher → Evaluate
2. Fine-tune teacher with color jitter → Evaluate
3. Distill to student with CRD → Evaluate
4. Load & evaluate other methods
5. Compare results

**Usage:**
```bash
python example_part5_main.py [--quick] [--skip-finetuning]
```

---

## Customization

### Different Color Jitter Intensity

In `part5_color_invariance.py`:

```python
class ColorJitterTransform:
    def __init__(self, brightness=0.4, contrast=0.4, 
                 saturation=0.4, hue=0.2):
        # Adjust these values
        # Higher = more aggressive augmentation
```

### Different Training Epochs

In `example_part5_main.py`:

```python
# For fine-tuning
num_epochs = 10  # Change this

# For CRD
num_epochs = 100  # Change this
```

### Different Distillation Parameters

In `analyzer.knowledge_distillation_crd()`:

```python
temperature=4.0      # How soft are teacher probabilities?
alpha=0.5           # Weight of KD loss (0.5 = balanced)
learning_rate=0.1   # Training rate
```

### Test Different Augmentations

In `get_cifar100_loaders()`, modify transforms:

```python
transforms.ColorJitter(
    brightness=0.5,   # ← Increase
    contrast=0.3,
    saturation=0.2,
    hue=0.1
)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Accuracy on jittered = 0% | Check color jitter transform |
| Teacher fine-tuning not improving | Increase learning rate or epochs |
| CRD student accuracy drops | Verify teacher loading |
| Results file not created | Check write permissions |
| Out of memory | Reduce batch_size, use --quick |

---

## Output Interpretation

### color_invariance_results.json

```json
{
  "Original Teacher": {
    "accuracy_normal": 0.76,
    "accuracy_jittered": 0.60,
    "accuracy_drop": 0.16,
    "invariance_ratio": 0.79
  },
  "Color-Invariant Teacher": {
    "accuracy_normal": 0.74,
    "accuracy_jittered": 0.71,
    "accuracy_drop": 0.03,
    "invariance_ratio": 0.96
  },
  "CRD Student (From Color-Invariant Teacher)": {
    "accuracy_normal": 0.72,
    "accuracy_jittered": 0.68,
    "accuracy_drop": 0.04,
    "invariance_ratio": 0.94
  }
}
```

### Charts

**accuracy_comparison.png:**
- Shows normal vs jittered accuracy side-by-side
- Gap between bars = lack of color invariance

**invariance_ratio.png:**
- Horizontal bar chart
- Higher = more invariant
- Green line at 1.0 = perfect invariance

---

## Key Findings Discussion

### What Should Happen?

1. **Original teacher:** Not color-invariant (drop ~15-20%)
2. **Color-invariant teacher:** Drop reduced to ~3-5%
3. **CRD student:** Inherits invariance (drop ~3-5%)
4. **SI student:** No improvement (drop ~15-20%)

### Success Criteria

✓ CRD invariance significantly > SI invariance
✓ CRD invariance close to teacher invariance
✓ CRD student did NOT see color-jittered training data
✓ Other methods (LM) show some improvement but less than CRD

### Failure Indicators

✗ CRD invariance similar to SI
✗ Student accuracy collapses on jittered data
✗ Fine-tuning doesn't improve teacher invariance
✗ CRD worse than expected

---

## Running Time

| Mode | Duration |
|------|----------|
| Quick (small epochs) | ~5-10 min |
| Full (default epochs) | ~20-30 min |
| With GPU | 3-5x faster |

---

## File Sizes

| File | Size |
|------|------|
| part5_color_invariance.py | ~22 KB |
| example_part5_main.py | ~14 KB |
| color_invariance_results.json | ~1 KB |

---

## Dependencies

```bash
pip install torch torchvision numpy matplotlib seaborn
```

---

## Advanced Analysis

### Analyze Per-Class Invariance

Modify `evaluate_model()` to track per-class accuracy:

```python
for class_id in range(100):
    class_mask = labels == class_id
    # Compute accuracy for this class
```

### Test Multiple Augmentations

Create separate evaluation functions for different transformations:

```python
def evaluate_with_augmentation(model, loader, augmentation_fn):
    # Apply custom augmentation
    # Evaluate and return accuracy
```

### Visualize Examples

Save misclassified examples:

```python
# Find images model fails on when jittered
# Save side-by-side: original vs jittered
```

---

## Related Parts

- **Part 3:** Distribution alignment (probability transfer)
- **Part 4:** Localization alignment (spatial attention transfer)
- **Part 5:** Robustness transfer (invariance properties)

---

## Summary

Part 5 demonstrates whether CRD can transfer implicit robustness properties. The key insight is that **students can learn to be robust without explicit robustness training**, purely by mimicking robust teachers.

This is a strong indicator that knowledge distillation transfers not just predictions, but also reasoning and robustness properties.