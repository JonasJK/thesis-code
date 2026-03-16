#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d


def load_coefficients(file_path):
    """Load coefficients from CSV file."""
    df = pd.read_csv(
        file_path,
        header=None,
        names=[
            "Red",
            "Green",
            "Blue",
            "VARI",
            "ExG",
            "ExGR",
            "CIVE",
            "VEG",
            "Intercept",
        ],
        skiprows=55,
    )
    return df


def calculate_statistics(df):
    """Calculate mean, std, and confidence intervals for each coefficient."""
    stats = {}
    for col in df.columns:
        stats[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "count": len(df[col]),
        }
        sem = stats[col]["std"] / np.sqrt(stats[col]["count"])  # Standard error of mean
        stats[col]["ci_95"] = 1.96 * sem
    return stats


def create_coefficient_plots(df, stats):
    """Create box plots and time series plots for coefficient analysis."""

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Box Plots", "Time Series (Sample Order)"))

    coefficients = [
        "Red",
        "Green",
        "Blue",
        "VARI",
        "ExG",
        "ExGR",
        "CIVE",
        "VEG",
        "Intercept",
    ]
    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#98D8C8",
        "#F7DC6F",
        "#BB8FCE",
    ]

    for i, coef in enumerate(coefficients):
        fig.add_trace(
            go.Box(y=df[coef], name=coef, marker_color=colors[i], showlegend=True),
            row=1,
            col=1,
        )

    sample_indices = list(range(len(df)))
    for i, coef in enumerate(coefficients):
        # Add scatter points with transparency
        fig.add_trace(
            go.Scatter(
                x=sample_indices,
                y=df[coef],
                mode="markers",
                name=f"{coef} points",
                marker={"color": colors[i], "opacity": 0.3, "size": 4},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Add smooth interpolated line using LOWESS-like smoothing
        if len(sample_indices) > 3:  # Need at least 4 points for cubic interpolation
            # Create more points for smooth curve
            x_smooth = np.linspace(min(sample_indices), max(sample_indices), len(sample_indices) * 3)
            f = interp1d(
                sample_indices,
                df[coef],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            y_smooth = f(x_smooth)

            fig.add_trace(
                go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode="lines",
                    name=coef,
                    line={"color": colors[i], "width": 2},
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        else:
            # Fallback for small datasets
            fig.add_trace(
                go.Scatter(
                    x=sample_indices,
                    y=df[coef],
                    mode="lines",
                    name=coef,
                    line={"color": colors[i], "width": 2},
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.update_layout(
        title_text="Coefficient Analysis: Box Plots and Time Series",
        height=600,
        showlegend=True,
    )

    fig.update_xaxes(title_text="Features", row=1, col=1)
    fig.update_xaxes(title_text="Sample Index", row=1, col=2)

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)

    return fig


def print_statistics_summary(stats):
    """Print a summary of the coefficient statistics."""
    print("\n" + "=" * 60)
    print("COEFFICIENT ANALYSIS SUMMARY")
    print("=" * 60)

    for coef, stat in stats.items():
        print(f"\n{coef.upper()} Coefficient:")
        print(f"  Mean:              {stat['mean']:8.3f}")
        print(f"  Standard Deviation: {stat['std']:8.3f}")
        print(f"  95% Confidence Interval: ±{stat['ci_95']:6.3f}")
        print(f"  Range:             [{stat['min']:7.3f}, {stat['max']:7.3f}]")
        print(f"  Sample Count:      {stat['count']:8d}")


def main():
    """Main function to run the coefficient analysis."""

    csv_file = "coefficients.csv"

    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found in current directory.")
        print("Please make sure the coefficients.csv file is in the same directory as this script.")
        return

    try:
        print("Loading coefficient data...")
        df = load_coefficients(csv_file)
        print(f"Loaded {len(df)} samples with {len(df.columns)} coefficients.")

        print("Calculating statistics...")
        stats = calculate_statistics(df)

        print_statistics_summary(stats)

        print("\nGenerating plots...")

        dashboard_fig = create_coefficient_plots(df, stats)
        dashboard_fig.show()

        print("\nSaving plot...")
        dashboard_fig.write_html("coefficient_dashboard.html")
        print("Plot saved as 'coefficient_dashboard.html'")

        print("\nFirst 5 rows of data:")
        print(df.head())

        print("\nBasic statistics:")
        print(df.describe())

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
