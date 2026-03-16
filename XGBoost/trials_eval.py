import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("optuna_trials.csv", header=0)
required_cols = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "rmse",
]
mask = ~df[required_cols].isna().any(axis=1)
df_clean = df.loc[mask].copy()
print(
    f"Best Trial RMSE: {df_clean['rmse'].min():.4f} at index {df_clean['rmse'].idxmin()} with values: {df_clean.loc[df_clean['rmse'].idxmin(), required_cols].to_dict()}"
)
param_cols = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
]

# Create performance categories
df_clean["performance"] = pd.cut(
    df_clean["rmse"],
    bins=[
        0,
        np.percentile(df_clean["rmse"], 10),
        np.percentile(df_clean["rmse"], 25),
        np.percentile(df_clean["rmse"], 75),
        np.inf,
    ],
    labels=["Best", "Good", "Average", "Poor"],
)

best_mask = df_clean["performance"] == "Best"
poor_mask = df_clean["performance"] == "Poor"

# Correlation heatmap comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# All trials
corr_all = df_clean[param_cols + ["rmse"]].corr()
sns.heatmap(
    corr_all,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    ax=axes[0],
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"label": "Correlation"},
)
axes[0].set_title("All Trials Correlation", fontweight="bold", fontsize=12)

# Best trials only
corr_best = df_clean.loc[best_mask, param_cols + ["rmse"]].corr()
sns.heatmap(
    corr_best,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    ax=axes[1],
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"label": "Correlation"},
)
axes[1].set_title("Best Trials (Top 10%) Correlation", fontweight="bold", fontsize=12)

# Poor trials only
corr_poor = df_clean.loc[poor_mask, param_cols + ["rmse"]].corr()
sns.heatmap(
    corr_poor,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    ax=axes[2],
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"label": "Correlation"},
)
axes[2].set_title("Poor Trials (Bottom 25%) Correlation", fontweight="bold", fontsize=12)

plt.suptitle("Correlation Comparison: Best vs Poor Trials", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("eval/correlation_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

for param in param_cols:
    corr_all_val = corr_all.loc[param, "rmse"]
    corr_best_val = corr_best.loc[param, "rmse"]
    corr_poor_val = corr_poor.loc[param, "rmse"]
    diff = abs(corr_best_val - corr_poor_val)
    print(f"{param:20s}: All={corr_all_val:+.3f}, Best={corr_best_val:+.3f}, Poor={corr_poor_val:+.3f}, |Δ|={diff:.3f}")

interactions = [
    ("n_estimators", "learning_rate"),
    ("max_depth", "learning_rate"),
    ("subsample", "colsample_bytree"),
    ("learning_rate", "subsample"),
]

for p1, p2 in interactions:
    corr_best_val = corr_best.loc[p1, p2]
    corr_poor_val = corr_poor.loc[p1, p2]
    diff = abs(corr_best_val - corr_poor_val)
    print(f"{p1} × {p2}:")
    print(
        f"  Best: {corr_best_val:+.3f}, Poor: {corr_poor_val:+.3f}, |Δ|={diff:.3f} {'***' if diff > 0.3 else '**' if diff > 0.2 else '*' if diff > 0.1 else ''}"
    )
