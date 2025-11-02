# Part 4: Examining Localization Knowledge Transfer via GradCAM

## Overview

Part 4 analyzes whether Knowledge Distillation transfers not only **predictive power** but also the ability to **focus on relevant features**. Using GradCAM visualizations, we compare spatial attention patterns between teacher and student models.

**Key Question:** Do students learn to look at the same image regions as teachers?

---

## What is GradCAM?

**Grad-weighted Class Activation Mapping** visualizes which parts of an input a neural network considers important for its predictions.

Based on: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02055) (Selvaraju et al., ICCV 2017)

### How It Works

1. **Forward pass:** Get feature maps from target layer
2. **Backward pass:** Compute gradients for predicted class
3. **Weight:** Combine with gradient magnitudes
4. **Result:** Heatmap showing "important regions"

---

## Files Included

| File | Purpose |
|------|---------|
| `part4_gradcam_localization.py` | Core analyzer & metrics |
| `example_part4_main.py` | Complete working example |
| `PART4_GUIDE.md` | Comprehensive reference |
| `PART4_QUICKSTART.md` | Quick start guide |
| `PART4_README.md` | This file |

---

## Quick Start

### 1. Update Checkpoint Paths

Edit `example_part4_main.py`, update `CHECKPOINT_PATHS`:

```python
CHECKPOINT_PATHS = {
    'teacher': {'path': './path/to/teacher.pth', 'class': VGG16},
    'LM': {'path': './path/to/lm.pth', 'class': VGG11},
    'CRD': {'path': './path/to/crd.pth', 'class': VGG11},
    # ...
}
```

### 2. Run Analysis

```bash
# Quick test (20 samples, ~2 min)
python example_part4_main.py --quick

# Full analysis (~10-15 min)
python example_part4_main.py

# Custom sample count
python example_part4_main.py --num-samples 500
```

### 3. Review Results

Output files:
- `gradcam_analysis_results.json` - All metrics
- `part4_gradcam_viz/` - Visualizations:
  - `image_*.png` - CAM side-by-side comparisons
  - `cam_pearson_correlation.png` - Spatial correlation
  - `cam_iou_comparison.png` - Region overlap
  - `cam_l2_kl_comparison.png` - Distance metrics

---

## Key Metrics

### 1. Pearson Correlation (0-1, higher better)
Measures if two heatmaps have similar spatial patterns.
- **1.0:** Perfect pattern correlation
- **0.85:** Excellent agreement
- **0.5:** Moderate agreement
- **0.0:** No correlation

### 2. L2 Similarity (0-1, higher better)
Pixel-by-pixel spatial accuracy.
- **1.0:** Identical heatmaps
- **0.8:** Very similar
- **0.5:** Moderately similar
- **0.0:** Completely different

### 3. Intersection over Union (0-1, higher better)
Binary overlap of "important regions" (threshold >0.5).
- **1.0:** Complete region overlap
- **0.75:** Excellent overlap
- **0.5:** Moderate overlap
- **0.0:** No overlap

### 4. KL Divergence (0-∞, lower better)
Distribution-based difference (treating CAMs as probability distributions).
- **0.0:** Identical distributions
- **0.15:** Excellent agreement
- **0.5:** Moderate agreement
- **>1.0:** Very different

---

## Interpretation Guide

### Excellent Results
```
Pearson Corr: > 0.85
L2 Similarity: > 0.85
IoU: > 0.75
KL Divergence: < 0.15
```
✓ Student focuses on same regions as teacher
✓ Strong knowledge transfer
✓ Likely good generalization

### Good Results
```
Pearson Corr: 0.70-0.85
L2 Similarity: 0.70-0.85
IoU: 0.60-0.75
KL Divergence: 0.15-0.30
```
✓ Student similar to teacher with variations
✓ Reasonable knowledge transfer
✓ Probably generalizes okay

### Acceptable Results
```
Pearson Corr: 0.50-0.70
L2 Similarity: 0.50-0.70
IoU: 0.40-0.60
KL Divergence: 0.30-0.60
```
⚠ Student different from teacher
⚠ Limited knowledge transfer
⚠ Risk of poor generalization

### Problematic Results
```
Pearson Corr: < 0.50
L2 Similarity: < 0.50
IoU: < 0.40
KL Divergence: > 0.60
```
✗ Student very different from teacher
✗ Little knowledge transfer
✗ High risk of poor generalization

---

## Expected Results (CIFAR-100)

Typical pattern you should see:

| Model | Pearson | L2 Sim | IoU | KL Div | Status |
|-------|---------|--------|-----|--------|--------|
| Independent | 0.45 | 0.52 | 0.35 | 0.65 | Baseline |
| Logit Match | 0.65 | 0.70 | 0.58 | 0.28 | Good |
| Hints | 0.72 | 0.76 | 0.65 | 0.22 | Very Good |
| CRD | 0.78 | 0.82 | 0.72 | 0.15 | Excellent |

**Key insight:** Better KD methods → higher CAM similarity → better knowledge transfer

---

## Visualization Examples

### Side-by-side CAM Comparison
```
[Original] [Teacher CAM] [Student CAM]

Red = high focus
Blue = low focus

If CAMs similar → Student learned teacher's reasoning ✓
If CAMs different → Student guessed differently ✗
```

### Similarity Metrics
```
Bar charts comparing:
- Pearson Correlation (spatial pattern)
- L2 Similarity (pixel accuracy)
- IoU (region overlap)
- KL Divergence (distribution)

Higher/lower (as appropriate) = better KD ✓
```

---

## Correct vs Incorrect Predictions

### For Correct Predictions
Student should have **HIGH similarity** to teacher:
- Both predicted same class
- Should focus on same features
- Low similarity = student got lucky

### For Incorrect Predictions
Student can have **LOWER similarity**:
- Different errors are okay
- Both just confused
- Still interesting if same mistakes

### Key Pattern
- Good KD: Similar on correct, different on incorrect
- Bad KD: Similar on both (no reasoning difference)

---

## Customization

### Analyze Specific Students Only
```python
student_models = ['CRD', 'Hints']  # Skip others
```

### Use Different Target Layer
```python
def target_layer_getter(model):
    return model.features[-2]  # Not the last layer
```

### Different Batch Size
```python
loader = get_cifar100_loader('val', batch_size=8)  # Smaller batch
```

### Custom Image Denormalization
```python
def my_denorm(img):
    return img * my_std + my_mean

analyzer.visualize_cams(
    models,
    indices,
    denormalization_fn=my_denorm
)
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| Uniform/noisy CAMs | Check if ReLU applied, gradients computed |
| Zero CAMs | Verify target layer, backward hook registered |
| Memory error | Reduce batch size, use `--quick` |
| Wrong checkpoint | Verify path, check file exists |
| Results unexpected | Debug checkpoint loading, model architecture |

---

## Connecting to Part 3

**Part 3:** Probability distributions
- Measures: Do students output similar probabilities?
- Metric: KL divergence (probability domain)
- Result: Predictions alignment

**Part 4:** Spatial attention
- Measures: Do students focus on same regions?
- Metric: CAM correlation (spatial domain)
- Result: Reasoning alignment

Both should show same ranking:
- CRD best
- Hints close
- Logit Matching okay
- Independent worst

---

## What You'll Learn

✓ How to visualize model attention using GradCAM
✓ How to measure spatial alignment quantitatively
✓ Why accuracy alone is incomplete
✓ How knowledge transfer includes "reasoning transfer"
✓ How to interpret neural network visualizations
✓ When students learn vs memorize

---

## Requirements

### Libraries
```bash
pip install torch torchvision numpy matplotlib seaborn scipy opencv-python
```

### Hardware
- GPU recommended (5-10x faster)
- 4GB+ RAM (8GB recommended)

### Data
- CIFAR-100 (auto-downloaded)

---

## Performance Notes

| Mode | Time | Memory | Use Case |
|------|------|--------|----------|
| Quick (20 samples) | 2 min | ~2GB | Setup test |
| Medium (100 samples) | 5 min | ~3GB | Quick analysis |
| Full (10K samples) | 15 min | ~4GB | Complete analysis |

---

## Advanced Topics

### Per-Layer Analysis
Generate CAMs from different layers to see how attention evolves.

### Robustness Analysis
Generate CAMs after perturbations (noise, rotation) to test robustness.

### Ensemble Comparison
Compare teacher ensemble vs student single model.

### Adversarial Robustness
Do adversarial examples transfer between models? (Advanced)

---

## References

1. Selvaraju et al. (2017): Grad-CAM
   [arXiv:1610.02055](https://arxiv.org/abs/1610.02055)

2. Hinton et al. (2015): Knowledge Distillation
   [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

3. Additional references from assignment document

---

## Next: Part 5

Part 5 analyzes whether robustness properties transfer:
- Color invariance with CRD
- Can students be robust without explicit training?
- Does knowledge transfer include robustness?

---

## Support

For issues:
1. Check PART4_QUICKSTART.md for quick answers
2. Read PART4_GUIDE.md for detailed explanations
3. Check example_part4_main.py comments
4. Verify model checkpoints and paths

---

**Ready to start?** → Run `python example_part4_main.py --quick`

Begin with quick mode, verify setup, then run full analysis!