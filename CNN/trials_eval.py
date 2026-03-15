#!/usr/bin/env python3
"""
Script to analyze and visualize Optuna optimization trials for CNN model.
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_trials(csv_path="optuna_trials.csv"):
    """Load trials from CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} trials from {csv_path}")
    return df

def analyze_trials(df):
    """Print summary statistics of trials."""
    print("\n" + "=" * 50)
    print("TRIAL SUMMARY")
    print("=" * 50)

    # Filter out infinite RMSE values
    valid_df = df[df["rmse"] != float("inf")]

    if len(valid_df) == 0:
        print("No valid trials found!")
        return

    print(f"Total trials: {len(df)}")
    print(f"Valid trials: {len(valid_df)}")
    print(f"Failed trials: {len(df) - len(valid_df)}")
    print()

    print(f"Best RMSE: {valid_df['rmse'].min():.4f}")
    print(f"Worst RMSE: {valid_df['rmse'].max():.4f}")
    print(f"Mean RMSE: {valid_df['rmse'].mean():.4f}")
    print(f"Median RMSE: {valid_df['rmse'].median():.4f}")
    print(f"Std RMSE: {valid_df['rmse'].std():.4f}")
    print()

    best_idx = valid_df["rmse"].idxmin()
    print("Best trial parameters:")
    for col in valid_df.columns:
        if col != "rmse":
            print(f"  {col}: {valid_df.loc[best_idx, col]}")
    print(f"  RMSE: {valid_df.loc[best_idx, 'rmse']:.4f}")
    print("=" * 50)

def plot_trials(df, output_dir="eval"):
    """Create visualization plots for trials."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter out infinite RMSE values
    valid_df = df[df["rmse"] != float("inf")]

    if len(valid_df) == 0:
        print("No valid trials to plot!")
        return

    sns.set_style("whitegrid")

    # RMSE over trials.
    plt.figure(figsize=(10, 6))
    plt.plot(valid_df.index, valid_df["rmse"], marker="o", linestyle="-", alpha=0.6)
    plt.axhline(y=valid_df["rmse"].min(), color="r", linestyle="--", label="Best RMSE")
    plt.xlabel("Trial Number")
    plt.ylabel("RMSE")
    plt.title("RMSE over Optimization Trials")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_over_trials.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/rmse_over_trials.png")

    # Parameter distributions.
    param_cols = [col for col in valid_df.columns if col != "rmse"]
    n_params = len(param_cols)

    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    for idx, param in enumerate(param_cols):
        ax = axes[idx]
        if valid_df[param].dtype in ["int64", "float64"]:
            # Numerical parameter
            ax.hist(valid_df[param], bins=20, alpha=0.7, edgecolor="black")
            ax.set_xlabel(param)
            ax.set_ylabel("Frequency")
            ax.set_title(f"{param} Distribution")
        else:
            # Categorical parameter
            valid_df[param].value_counts().plot(kind="bar", ax=ax, alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel("Frequency")
            ax.set_title(f"{param} Distribution")
            ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_distributions.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/parameter_distributions.png")

    # Parameter vs RMSE scatter plots.
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]

    for idx, param in enumerate(param_cols):
        ax = axes[idx]
        if valid_df[param].dtype in ["int64", "float64"]:
            ax.scatter(valid_df[param], valid_df["rmse"], alpha=0.6)
            ax.set_xlabel(param)
            ax.set_ylabel("RMSE")
            ax.set_title(f"RMSE vs {param}")

            # Add trend line if appropriate
            if len(valid_df[param].unique()) > 2:
                z = np.polyfit(valid_df[param], valid_df["rmse"], 1)
                p = np.poly1d(z)
                ax.plot(
                    valid_df[param], p(valid_df[param]), "r--", alpha=0.8, linewidth=2
                )
        else:
            # Box plot for categorical
            valid_df.boxplot(column="rmse", by=param, ax=ax)
            ax.set_xlabel(param)
            ax.set_ylabel("RMSE")
            ax.set_title(f"RMSE by {param}")

    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_vs_rmse.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/parameter_vs_rmse.png")

    # Correlation heatmap (for numerical parameters only).
    numerical_cols = [
        col for col in valid_df.columns if valid_df[col].dtype in ["int64", "float64"]
    ]
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = valid_df[numerical_cols].corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Parameter Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
        plt.close()
        print(f"Saved: {output_dir}/correlation_heatmap.png")

def save_summary(df, output_dir="eval"):
    """Save summary statistics to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    valid_df = df[df["rmse"] != float("inf")]

    if len(valid_df) == 0:
        print("No valid trials to summarize!")
        return

    best_idx = valid_df["rmse"].idxmin()

    summary = {
        "total_trials": len(df),
        "valid_trials": len(valid_df),
        "failed_trials": len(df) - len(valid_df),
        "best_rmse": float(valid_df["rmse"].min()),
        "worst_rmse": float(valid_df["rmse"].max()),
        "mean_rmse": float(valid_df["rmse"].mean()),
        "median_rmse": float(valid_df["rmse"].median()),
        "std_rmse": float(valid_df["rmse"].std()),
        "best_parameters": {
            col: valid_df.loc[best_idx, col]
            for col in valid_df.columns
            if col != "rmse"
        },
    }

    # Convert NumPy scalars before writing JSON.
    for key, value in summary["best_parameters"].items():
        if hasattr(value, "item"):
            summary["best_parameters"][key] = value.item()

    output_path = os.path.join(output_dir, "summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(description="Analyze CNN optimization trials")
    parser.add_argument(
        "--csv", type=str, default="optuna_trials.csv", help="Path to trials CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval",
        help="Directory to save plots and summary",
    )

    args = parser.parse_args()

    # Load trials
    df = load_trials(args.csv)
    if df is None:
        exit(1)

    analyze_trials(df)

    plot_trials(df, args.output_dir)

    save_summary(df, args.output_dir)

    print("\nAnalysis complete!")
