"""
Visualize fault_dataset.npz — IEEE 39-Bus Power Grid Fault Dataset
Usage: python visualize_npz.py [path_to_npz]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load dataset ───────────────────────────────────────────────────────────
npz_path = sys.argv[1] if len(sys.argv) > 1 else "data/fault_dataset.npz"
data = np.load(npz_path)
X      = data["X"]        # (N, 39, 5, 50)
y_loc  = data["y_loc"]    # (N,)  bus index, -1 = normal
y_type = data["y_type"]   # (N,)  0=SLG, 1=LL, 2=DLG, 3=3PH, 4=Normal

N, n_buses, n_features, T = X.shape
FEAT_NAMES  = ["Vm (pu)", "θ (rad)", "P (MW)", "Q (MVar)", "Δf (Hz)"]
FAULT_NAMES = ["SLG", "LL", "DLG", "3PH", "Normal"]

print(f"Dataset: {npz_path}")
print(f"  Samples: {N}  |  Buses: {n_buses}  |  Features: {n_features}  |  Timesteps: {T}")
print(f"  Fault samples : {(y_loc >= 0).sum()}")
print(f"  Normal samples: {(y_loc < 0).sum()}")
print()

# ── Figure 1: Dataset overview (class distributions) ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1a) Fault type distribution
type_counts = [(y_type == i).sum() for i in range(len(FAULT_NAMES))]
colors = ["#1565C0", "#6A1B9A", "#00695C", "#BF360C", "#2E7D32"]
bars = axes[0].bar(FAULT_NAMES, type_counts, color=colors)
axes[0].set_title("Fault Type Distribution", fontweight="bold")
axes[0].set_ylabel("Sample Count")
for i, v in enumerate(type_counts):
    axes[0].text(i, v + max(type_counts) * 0.02, str(v), ha="center", fontsize=9)

# 1b) Fault frequency per bus
bus_counts = np.bincount(y_loc[y_loc >= 0], minlength=n_buses)
axes[1].bar(range(n_buses), bus_counts, color="#1565C0", alpha=0.7)
axes[1].set_title("Fault Frequency per Bus", fontweight="bold")
axes[1].set_xlabel("Bus Index")
axes[1].set_ylabel("Count")

# 1c) Fault vs Normal pie chart
fault_total = (y_loc >= 0).sum()
normal_total = (y_loc < 0).sum()
axes[2].pie([fault_total, normal_total], labels=["Fault", "Normal"],
            autopct="%1.1f%%", colors=["#C62828", "#2E7D32"], startangle=90)
axes[2].set_title("Fault vs Normal Ratio", fontweight="bold")

plt.suptitle(f"Dataset Overview  |  {N:,} total samples", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/viz_01_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 2: Feature distributions (boxplots) ───────────────────────────
fig, axes = plt.subplots(1, n_features, figsize=(20, 4))
# Use mean over time for each sample/bus
X_mean = X.mean(axis=3)  # (N, 39, 5)

for f in range(n_features):
    fault_vals  = X_mean[y_loc >= 0, :, f].flatten()
    normal_vals = X_mean[y_loc < 0, :, f].flatten()
    # Subsample for speed
    rng = np.random.default_rng(42)
    n_sub = min(50000, len(fault_vals))
    axes[f].boxplot(
        [rng.choice(normal_vals, n_sub, replace=False),
         rng.choice(fault_vals, n_sub, replace=False)],
        labels=["Normal", "Fault"], widths=0.6
    )
    axes[f].set_title(FEAT_NAMES[f], fontweight="bold")

plt.suptitle("Feature Distributions: Normal vs Fault (mean over time)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("data/viz_02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 3: Sample time-series for each fault type ─────────────────────
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(len(FAULT_NAMES), n_features, hspace=0.4, wspace=0.3)

for ft_idx, ft_name in enumerate(FAULT_NAMES):
    mask = y_type == ft_idx
    if mask.sum() == 0:
        continue
    sample_idx = np.where(mask)[0][0]
    bus = y_loc[sample_idx] if y_loc[sample_idx] >= 0 else 0
    for f in range(n_features):
        ax = fig.add_subplot(gs[ft_idx, f])
        signal = X[sample_idx, bus, f, :]  # (T,)
        ax.plot(signal, linewidth=1.2, color=colors[ft_idx])
        if ft_idx == 0:
            ax.set_title(FEAT_NAMES[f], fontsize=10, fontweight="bold")
        if f == 0:
            ax.set_ylabel(ft_name, fontsize=10, fontweight="bold")
        if ft_idx == len(FAULT_NAMES) - 1:
            ax.set_xlabel("Timestep")
        ax.tick_params(labelsize=7)

plt.suptitle("Sample Time-Series per Fault Type (at faulted bus)", fontsize=14, fontweight="bold")
plt.savefig("data/viz_03_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 4: Heatmap — average feature values across buses (per type) ───
fig, axes = plt.subplots(1, len(FAULT_NAMES), figsize=(22, 6))

for ft_idx, ft_name in enumerate(FAULT_NAMES):
    mask = y_type == ft_idx
    if mask.sum() == 0:
        continue
    # Mean over samples and time → (n_buses, n_features)
    avg = X[mask].mean(axis=(0, 3))
    im = axes[ft_idx].imshow(avg.T, aspect="auto", cmap="RdBu_r")
    axes[ft_idx].set_title(ft_name, fontweight="bold")
    axes[ft_idx].set_xlabel("Bus Index")
    if ft_idx == 0:
        axes[ft_idx].set_yticks(range(n_features))
        axes[ft_idx].set_yticklabels(FEAT_NAMES, fontsize=8)
    else:
        axes[ft_idx].set_yticks([])
    plt.colorbar(im, ax=axes[ft_idx], fraction=0.046, pad=0.04)

plt.suptitle("Average Feature Heatmap per Fault Type (Bus × Feature)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/viz_04_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 5: Correlation matrix of features (fault samples) ─────────────
fault_mask = y_loc >= 0
X_fault_flat = X[fault_mask].mean(axis=3)  # (N_fault, 39, 5)
X_flat = X_fault_flat.reshape(-1, n_features)  # (N_fault*39, 5)

rng = np.random.default_rng(42)
sub_idx = rng.choice(len(X_flat), min(100000, len(X_flat)), replace=False)
corr = np.corrcoef(X_flat[sub_idx].T)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(n_features))
ax.set_xticklabels(FEAT_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n_features))
ax.set_yticklabels(FEAT_NAMES, fontsize=9)
for i in range(n_features):
    for j in range(n_features):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, fraction=0.046, pad=0.04)
ax.set_title("Feature Correlation (Fault Samples)", fontweight="bold")
plt.tight_layout()
plt.savefig("data/viz_05_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ All visualizations saved to data/ folder:")
print("   viz_01_overview.png")
print("   viz_02_feature_distributions.png")
print("   viz_03_timeseries.png")
print("   viz_04_heatmaps.png")
print("   viz_05_correlation.png")
