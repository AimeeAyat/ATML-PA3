"""
Part 3: Probability Distribution Alignment Analysis
COMPREHENSIVE IMPLEMENTATION GUIDE
======================================

This guide explains:
1. What Part 3 measures
2. All metrics implemented
3. How to customize the analysis
4. Interpreting results
"""

# ============================================================================
# CONCEPTUAL OVERVIEW
# ============================================================================

"""
GOAL: Understand if Knowledge Distillation transfers HOW teachers think,
not just WHAT they predict.

KEY INSIGHT:
Two models can achieve the same accuracy but for different reasons:
- Teacher predicts class A with 90% confidence for image X
- Student also predicts A but with only 60% confidence
- Both get it right, but they "think" differently

SOLUTION:
Compare probability distributions using divergence metrics.
Lower divergence = better knowledge transfer (thinking alike)
"""

# ============================================================================
# METRICS EXPLAINED
# ============================================================================

"""
1. KL DIVERGENCE (Kullback-Leibler)
   ──────────────────────────────────
   Formula: KL(P||Q) = Σ p(x) * log(p(x) / q(x))
   
   What it measures:
   - How different Q is from P (asymmetric!)
   - P = Teacher distribution
   - Q = Student distribution
   
   Interpretation:
   - KL = 0: Perfect match
   - KL > 0: Student differs from teacher
   - Larger values = worse alignment
   
   Why it matters:
   - Captures divergence in probability mass across all classes
   - Emphasizes where teacher and student disagree most
   
   Limitations:
   - Asymmetric: KL(T||S) ≠ KL(S||T)
   - Infinite if Q=0 where P>0 (handled with epsilon)


2. JENSEN-SHANNON DIVERGENCE
   ─────────────────────────────
   Formula: JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)
            where M = 0.5*(P+Q)
   
   What it measures:
   - Symmetric version of KL divergence
   - Average of both directions
   
   Interpretation:
   - Range: [0, log(2)] ≈ [0, 0.693]
   - Better for symmetrical comparison
   - 0 = perfect match
   
   Why it matters:
   - More intuitive than KL (symmetric)
   - Bounded and less sensitive to extreme values
   - Sqrt(JS) has nice properties (metric space)


3. HELLINGER DISTANCE
   ──────────────────
   Formula: H(P,Q) = sqrt(0.5 * Σ (sqrt(p) - sqrt(q))^2)
   
   What it measures:
   - Square root of average squared difference in sqrt-probabilities
   - Emphasizes differences more uniformly
   
   Interpretation:
   - Range: [0, 1]
   - 0 = identical distributions
   - 1 = completely different (orthogonal)
   
   Why it matters:
   - Bounded [0,1] - easy to interpret as "distance"
   - Symmetric
   - Less sensitive to extreme probability values


4. WASSERSTEIN DISTANCE (EMD - Earth Mover's Distance)
   ──────────────────────────────────────────────────
   Formula: W(P,Q) = average |P_cdf - Q_cdf|
   
   What it measures:
   - Minimum effort to transform one distribution to another
   - Uses cumulative distributions
   
   Interpretation:
   - Range: [0, 1]
   - 0 = identical
   - Intuitive: "cost of transportation"
   
   Why it matters:
   - Captures "global" vs "local" differences
   - Natural for comparing probability distributions
   - Less affected by outliers


5. COSINE SIMILARITY
   ─────────────────
   Formula: cos_sim = (P · Q) / (||P|| * ||Q||)
   
   What it measures:
   - Angle between probability vectors
   - How aligned the directions are
   
   Interpretation:
   - Range: [-1, 1]
   - 1 = parallel (same direction/shape)
   - 0 = orthogonal (no similarity)
   - -1 = opposite directions (rare in probabilities)
   
   Why it matters:
   - Measures distributional shape similarity
   - Invariant to magnitude (just direction)
   - Complements distance metrics


# ============================================================================
# TEMPERATURE ANALYSIS
# ============================================================================

"""
Why we test at multiple temperatures T:
- T=1.0: Original probabilities (no smoothing)
- T>1.0: Softer distributions (peaks smoothed, valleys raised)
- T→∞: Uniform distribution

Key insight:
- At high T, all distributions become similar (high entropy)
- At low T, differences are more pronounced
- Plotting across temperatures shows confidence structure

Example:
Teacher at T=1.0: [0.8, 0.1, 0.05, 0.05] (confident)
Student at T=1.0: [0.5, 0.3, 0.1, 0.1] (uncertain)
KL = higher (big difference in confidence)

Teacher at T=8.0: [0.27, 0.26, 0.23, 0.24] (uncertain)
Student at T=8.0: [0.27, 0.26, 0.23, 0.24] (uncertain)
KL = lower (look similar at high temp)

This reveals: Student has lower confidence, not just different class probs.
"""

# ============================================================================
# ANALYZING BY CORRECTNESS
# ============================================================================

"""
Why split by correct vs incorrect predictions:

Correct predictions:
- Student agrees with teacher on the label
- But does it have similar confidence?
- Low KL here = good knowledge transfer
- High KL here = student is guessing right for wrong reasons

Incorrect predictions:
- Student predicts differently than teacher
- How different are their distributions?
- High KL here = fundamental disagreement
- This should typically be higher

Good KD: High gap between correct and incorrect
Bad KD: Distributions look similar regardless of correctness
"""

# ============================================================================
# INTERPRETATION GUIDE
# ============================================================================

"""
WHAT DO DIFFERENT RESULTS MEAN?

Scenario 1: High accuracy, Low KL divergence
─────────────────────────────────────────────
✓ BEST CASE: Student learned like teacher
- Student predicts correctly
- Student is confident similarly
- True knowledge transfer occurred
- This is the goal of KD!

Example:
- Teacher: [0.85, 0.10, 0.03, 0.02]
- Student: [0.80, 0.12, 0.04, 0.04]
- Both confident about same class


Scenario 2: High accuracy, High KL divergence
──────────────────────────────────────────────
⚠ POTENTIAL ISSUE: Student got lucky
- Student predicts correctly
- But distributes confidence differently
- Different decision boundaries
- May not generalize as well
- Indicates weak knowledge transfer

Example:
- Teacher: [0.85, 0.10, 0.03, 0.02]
- Student: [0.50, 0.35, 0.10, 0.05]
- Same prediction, different reasoning


Scenario 3: Low accuracy, Low KL divergence
────────────────────────────────────────────
❌ PROBLEM: Teacher-Student mismatch
- Both are confident about wrong things
- Student learned to be confident incorrectly
- Something went wrong in training
- Re-examine training procedure

Example:
- Teacher: [0.85, 0.10, 0.03, 0.02]
- Student: [0.80, 0.12, 0.04, 0.04]
- Both predict class A... which is wrong!


Scenario 4: Low accuracy, High KL divergence
─────────────────────────────────────────────
❓ UNCERTAIN: Student diverged from teacher
- Student doesn't match teacher at all
- Different predictions
- Both low accuracy
- Something failed in distillation


# ============================================================================
# USING THE ANALYZER CLASS
# ============================================================================

from part3_distribution_alignment import DistributionAnalyzer
import torch

# Initialize
analyzer = DistributionAnalyzer(device='cuda')

# Load models
analyzer.load_model('teacher', 'path/to/teacher.pth', TeacherClass)
analyzer.load_model('student', 'path/to/student.pth', StudentClass)

# Get distributions
teacher_dist = analyzer.get_output_distributions(teacher_model, val_loader)
student_dist = analyzer.get_output_distributions(student_model, val_loader)

# Compute metrics
kl = analyzer.compute_kl_divergence(teacher_dist['distributions'], 
                                     student_dist['distributions'])
print(f"Mean KL: {kl['mean']:.6f}")
print(f"Median KL: {kl['median']:.6f}")
print(f"Std KL: {kl['std']:.6f}")

# Analyze alignment
results = analyzer.analyze_alignment(teacher_dist, 
                                     {'student': student_dist},
                                     ['student'])

# Create plots
analyzer.plot_divergence_comparison(results, ['student'], save_dir='plots')


# ============================================================================
# CUSTOMIZATION OPTIONS
# ============================================================================

# 1. Use different metric
js_div = analyzer.compute_js_divergence(p_dist, q_dist)
hellinger = analyzer.compute_hellinger_distance(p_dist, q_dist)

# 2. Analyze specific temperature
teacher_probs_t4 = F.softmax(logits / 4.0, dim=1)
student_probs_t4 = F.softmax(s_logits / 4.0, dim=1)

# 3. Sample-level analysis
kl_results = analyzer.compute_kl_divergence(p_dist, q_dist)
per_sample_kl = kl_results['per_sample']
# Now you can sort, filter, visualize individual samples

# 4. Per-class analysis
for class_id in range(num_classes):
    class_mask = teacher_outputs['targets'] == class_id
    teacher_class = teacher_dist['distributions'][class_mask]
    student_class = student_dist['distributions'][class_mask]
    kl_class = analyzer.compute_kl_divergence(teacher_class, student_class)
    print(f"Class {class_id}: KL = {kl_class['mean']:.6f}")


# ============================================================================
# EXPECTED RESULTS AND BENCHMARKS
# ============================================================================

"""
CIFAR-100 with VGG-11 student, VGG-16 teacher (typical results):

Model               Accuracy    KL Div      JS Div      Hellinger
────────────────────────────────────────────────────────────────
Independent (SI)    ~70-72%     N/A         N/A         N/A
Logit Matching      ~73-75%     0.15-0.25   0.10-0.15   0.15-0.20
Label Smoothing     ~72-74%     0.20-0.30   0.12-0.18   0.18-0.25
DKD                 ~74-76%     0.10-0.18   0.08-0.12   0.12-0.18
Hints               ~75-77%     0.12-0.22   0.08-0.14   0.13-0.19
CRD                 ~76-78%     0.08-0.16   0.06-0.11   0.10-0.16

Key observations:
- Better KD methods show lower divergence
- Lower divergence ≈ better knowledge transfer
- Not perfectly correlated with accuracy (some variance)
- Temperature analysis shows more structure in better methods


# ============================================================================
# SAVING AND REPORTING RESULTS
# ============================================================================

The analysis generates:

1. distribution_alignment_results.json
   - All computed metrics in structured format
   - Can be used for further analysis
   - Load with: json.load(open(...))

2. part3_plots/
   - kl_divergence_comparison.png: Bar chart of KL means
   - accuracy_vs_kl_divergence.png: Scatter plot (key visualization!)
   - temperature_effect.png: Line plot across temperatures
   - multiple_metrics_comparison.png: Comparison of all 4 metrics

3. Console output
   - Detailed statistics for each model
   - Accuracy comparisons
   - Metric breakdowns


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

Q: All KL divergences are very high (>1.0)?
A: Check if distributions are properly normalized.
   Ensure softmax is applied correctly.
   Verify no temperature is being applied unexpectedly.

Q: KL divergence is near 0 for all models?
A: This might mean softmax output is nearly uniform.
   Check if models are trained (not random initialization).
   Verify checkpoints loaded correctly.

Q: Why is teacher accuracy different in Part 3 vs Part 1/2?
A: Different subset of data being used.
   Different preprocessing/augmentations.
   Compare using same validation set.

Q: Student with lower accuracy has lower KL divergence?
A: Possible: Both learning similar wrong distribution.
   Or: Student is more confident/uniform distribution.
   Check per-sample analysis and visualization.

Q: How to compare different distillation methods?
A: Use the JSON results file:
   - Sort by KL divergence
   - Plot accuracy vs divergence
   - Use this as secondary metric to accuracy
"""

# ============================================================================
# ADVANCED USAGE
# ============================================================================

"""
1. PER-CLASS ANALYSIS
   Analyze each class separately to find which classes have worst alignment

2. SAMPLE HARDNESS
   Compute difficulty based on teacher logit magnitude
   Correlate with divergence (harder samples harder to distill?)

3. CONFIDENCE CALIBRATION
   Check if student is overconfident vs teacher
   Use ECE (Expected Calibration Error) alongside divergence

4. ENSEMBLE ANALYSIS
   Compare single teacher vs ensemble teacher
   Does ensemble lead to better/worse student alignment?

5. TRANSFER LEARNING
   How do distributions change across layers?
   Which intermediate layers transfer best?
"""