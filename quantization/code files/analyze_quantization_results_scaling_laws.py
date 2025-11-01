import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ==============================================================
# Setup
# ==============================================================
plt.style.use("seaborn-v0_8-paper")
sns.set_theme(style="whitegrid", font_scale=1.2)
output_dir = "/workspace/task1/plots"
os.makedirs(output_dir, exist_ok=True)

# ==============================================================
# Load JSON Data
# ==============================================================
with open("/workspace/task1/quantization_results_v2.json", "r") as f:
    data = json.load(f)["results"]

baseline = data["baseline"]
ptq = data["ptq"]
qat = data["qat"]
models = list(ptq.keys())

baseline_acc = baseline["accuracy"]
baseline_latency = baseline["latency_ms"]

# ==============================================================
# Create Summary Tables
# ==============================================================
summary_ptq = pd.DataFrame([
    {
        "Model": m,
        "Accuracy (PTQ)": ptq[m]["accuracy"],
        "Δ vs Baseline (%)": baseline_acc - ptq[m]["accuracy"],
        "Latency (ms)": ptq[m]["latency_ms"],
        "Accuracy Drop (%)": ptq[m]["drop"],
    }
    for m in models
])

summary_qat = pd.DataFrame([
    {
        "Model": m,
        "Final Accuracy (QAT)": qat[m]["final_accuracy"],
        "Recovery Epoch": qat[m]["recovery_epoch"],
        "Recovery Time (min)": qat[m]["recovery_time_sec"] / 60,
        "Model Size (MB)": qat[m]["model_size"],
        "Δ vs Baseline (%)": qat[m]["final_accuracy"] - baseline_acc,
    }
    for m in models
])

summary_ptq.to_csv(f"{output_dir}/ptq_summary.csv", index=False)
summary_qat.to_csv(f"{output_dir}/qat_summary.csv", index=False)

print("✅ Tables saved → ptq_summary.csv, qat_summary.csv")

# ==============================================================
# Initialize Multi-page PDF
# ==============================================================
pdf_path = f"{output_dir}/quantization_report.pdf"
pdf = PdfPages(pdf_path)

# Helper to save both individual PNG and PDF page
def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(f"{output_dir}/{name}.png", dpi=300)
    pdf.savefig(fig)
    plt.close(fig)

# ==============================================================
# A. PTQ vs QAT Accuracy Comparison
# ==============================================================
df_acc = pd.DataFrame({
    "Model": models,
    "Baseline": [baseline_acc]*len(models),
    "PTQ": [ptq[m]["accuracy"] for m in models],
    "QAT": [qat[m]["final_accuracy"] for m in models],
})
df_acc_melt = df_acc.melt(id_vars="Model", var_name="Stage", value_name="Accuracy")

fig = plt.figure(figsize=(8, 5))
sns.barplot(data=df_acc_melt, x="Model", y="Accuracy", hue="Stage", palette="Set2")
plt.title("A. Baseline vs PTQ vs QAT Accuracy")
save_fig(fig, "A_ptq_qat_accuracy")

# ==============================================================
# B. Accuracy Recovery Curves
# ==============================================================
fig = plt.figure(figsize=(10, 6))
for m in models:
    epochs = [e["epoch"] for e in qat[m]["epoch_history"]]
    accs = [e["test_acc"] for e in qat[m]["epoch_history"]]
    plt.plot(epochs, accs, label=f"{m} QAT")
    plt.axvline(qat[m]["recovery_epoch"], color="gray", linestyle="--", alpha=0.4)
plt.axhline(baseline_acc, color="k", linestyle="--", label="Baseline")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("B. Accuracy Recovery over Epochs")
plt.legend()
save_fig(fig, "B_accuracy_recovery_curves")

# ==============================================================
# C. Recovery Epoch vs PTQ Drop
# ==============================================================
drop = [baseline_acc - ptq[m]["accuracy"] for m in models]
epochs = [qat[m]["recovery_epoch"] for m in models]
fig = plt.figure(figsize=(7, 5))
sns.scatterplot(x=drop, y=epochs, hue=models, s=150, palette="viridis")
plt.xlabel("PTQ Accuracy Drop (%)")
plt.ylabel("QAT Recovery Epoch")
plt.title("C. Drop vs Recovery Epochs")
save_fig(fig, "C_drop_vs_recovery_epoch")

# ==============================================================
# D. Size–Latency Trade-off
# ==============================================================
sizes = [qat[m]["model_size"] for m in models]
lat_qat = [qat[m]["latency_ms"] for m in models]
fig = plt.figure(figsize=(8, 6))
sns.scatterplot(x=sizes, y=lat_qat, hue=models, s=200, style=models, palette="muted")
plt.xlabel("Model Size (MB)")
plt.ylabel("Latency (ms)")
plt.title("D. Size–Latency Trade-off (QAT Models)")
save_fig(fig, "D_size_latency_tradeoff")

# ==============================================================
# E. QAT Recovery Time
# ==============================================================
fig = plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=[qat[m]["recovery_time_sec"]/60 for m in models], palette="mako")
plt.ylabel("Recovery Time (minutes)")
plt.title("E. Time Required to Recover Baseline Accuracy")
save_fig(fig, "E_qat_recovery_time")

# ==============================================================
# F. Accuracy vs Compression Ratio
# ==============================================================


baseline_size = 491  # MB, fixed reference
models = ["FP16", "BF16", "INT8", "INT4"]

compression = [baseline_size / qat[m]["model_size"] for m in models]

fig = plt.figure(figsize=(8, 5))
sns.lineplot(x=compression, y=[qat[m]["final_accuracy"] for m in models], marker="o")

# Label each point with model name
for i, m in enumerate(models):
    plt.text(compression[i] + 0.05, qat[m]["final_accuracy"], m)

plt.xlabel("Compression Ratio (Baseline / Quantized)")
plt.ylabel("Final Accuracy (%)")
plt.title("F. Accuracy vs Model Compression")
plt.grid(True, alpha=0.3)

save_fig(fig, "F_accuracy_vs_compression")


# ==============================================================
# G. Speed-up vs Accuracy Retention
# ==============================================================
speedup = [baseline_latency / qat[m]["latency_ms"] for m in models]
acc_retention = [qat[m]["final_accuracy"]/baseline_acc*100 for m in models]
fig = plt.figure(figsize=(8, 6))
sns.scatterplot(x=speedup, y=acc_retention, hue=models, s=200, palette="coolwarm")
plt.xlabel("Speed-up (× Baseline Latency)")
plt.ylabel("Accuracy Retention (%)")
plt.title("G. Speed-up vs Accuracy Retention")
save_fig(fig, "G_speedup_vs_accuracy")

# ==============================================================
# H. Learning Efficiency (Accuracy/Time)
# ==============================================================
efficiency = [qat[m]["final_accuracy"] / (qat[m]["recovery_time_sec"]/60) for m in models]
fig = plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=efficiency, palette="crest")
plt.ylabel("Accuracy per Minute (Efficiency)")
plt.title("H. Learning Efficiency of QAT")
save_fig(fig, "H_learning_efficiency")

# ==============================================================
# Tables as PDF Pages
# ==============================================================
def plot_table(df, title):
    fig, ax = plt.subplots(figsize=(10, len(df)*0.5 + 1))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    plt.title(title)
    pdf.savefig(fig)
    plt.close(fig)

plot_table(summary_ptq, "Table 1 – PTQ Summary")
plot_table(summary_qat, "Table 2 – QAT Summary")

# ==============================================================
# Finalize Report
# ==============================================================
pdf.close()
print(f"\n✅ Multi-page PDF report saved → {pdf_path}")
print("✅ All individual figures saved under /workspace/task1/plots/")
