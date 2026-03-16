import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler


def load_data(coefficients_path, metrics_path):
    coef_df = pd.read_csv(
        coefficients_path,
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

    metrics_df = pd.read_csv(metrics_path)

    return coef_df, metrics_df


def prepare_data(coef_df, metrics_df):
    min_rows = min(len(coef_df), len(metrics_df))
    coef_df = coef_df.iloc[:min_rows]
    metrics_df = metrics_df.iloc[:min_rows]
    combined_df = pd.concat([coef_df, metrics_df], axis=1)
    return combined_df


def create_unified_quality_metric(combined_df):
    scaler = MinMaxScaler()

    metrics_to_normalize = pd.DataFrame(
        {
            "RMSE": combined_df["RMSE"],
            "MAE": combined_df["MAE"],
            "OneMinusSSIM": (1 - combined_df["SSIM"]),
        }
    )

    normalized_metrics = pd.DataFrame(
        scaler.fit_transform(metrics_to_normalize),
        columns=["RMSE_norm", "MAE_norm", "OneMinusSSIM_norm"],
    )

    combined_df["Quality_Score"] = normalized_metrics.mean(axis=1)

    combined_df["RMSE_norm"] = normalized_metrics["RMSE_norm"]
    combined_df["MAE_norm"] = normalized_metrics["MAE_norm"]
    combined_df["OneMinusSSIM_norm"] = normalized_metrics["OneMinusSSIM_norm"]

    return combined_df


def create_interactive_dashboard(combined_df):
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "Quality Score vs Red Coefficient",
            "Quality Score vs Green Coefficient",
            "Quality Score vs Blue Coefficient",
            "Quality Score vs VARI Coefficient",
            "Quality Score vs ExG Coefficient",
            "Quality Score vs ExGR Coefficient",
            "Quality Score vs CIVE Coefficient",
            "Quality Score vs VEG Coefficient",
            "Quality Score vs Intercept",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    coef_cols = [
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

    colors = combined_df["Quality_Score"]

    for i, coef in enumerate(coef_cols):
        row = (i // 3) + 1
        col = (i % 3) + 1

        fig.add_trace(
            go.Scatter(
                x=combined_df[coef],
                y=combined_df["Quality_Score"],
                mode="markers",
                marker={
                    "color": colors,
                    "colorscale": "Viridis",
                    "size": 8,
                    "opacity": 0.7,
                    "colorbar": (
                        {
                            "title": "Quality Score",
                            "tickmode": "array",
                            "tickvals": [0, 0.5, 1],
                            "ticktext": ["Best", "Medium", "Worst"],
                        }
                        if i == 8
                        else None
                    ),
                    "showscale": i == 8,
                },
                name=f"{coef} vs Quality",
                hovertemplate=f"<b>{coef}</b>: %{{x:.2f}}<br>"
                + "<b>Quality Score</b>: %{y:.3f}<br>"
                + "<b>RMSE</b>: "
                + combined_df["RMSE"].round(2).astype(str)
                + "<br>"
                + "<b>MAE</b>: "
                + combined_df["MAE"].round(2).astype(str)
                + "<br>"
                + "<b>SSIM</b>: "
                + combined_df["SSIM"].round(3).astype(str)
                + "<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        z = np.polyfit(combined_df[coef], combined_df["Quality_Score"], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(combined_df[coef].min(), combined_df[coef].max(), 50)
        y_trend = p(x_trend)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                line={"color": "rgba(255, 0, 0, 0.8)", "width": 3},
                name=f"{coef} Trend",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=1000,
        width=1400,
        plot_bgcolor="rgba(15, 23, 42, 1)",
        paper_bgcolor="rgba(15, 23, 42, 1)",
        font={
            "family": "Segoe UI, Roboto, Helvetica Neue, Arial",
            "size": 12,
            "color": "#e2e8f0",
        },
    )

    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(100, 116, 139, 0.2)",
                zeroline=False,
                title_font={"size": 12, "color": "#cbd5e1"},
                tickfont={"color": "#cbd5e1"},
                row=i,
                col=j,
            )

            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(100, 116, 139, 0.2)",
                zeroline=False,
                title="Quality Score",
                title_font={"size": 12, "color": "#cbd5e1"},
                tickfont={"color": "#cbd5e1"},
                range=[0, 1],
                row=i,
                col=j,
            )
    return fig


def create_top_coefficients_summary(combined_df):
    top_performers = combined_df[combined_df["Quality_Score"] <= combined_df["Quality_Score"].quantile(0.25)]

    coef_cols = [
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

    summary_data = []
    for coef in coef_cols:
        mean_val = top_performers[coef].mean()
        std_val = top_performers[coef].std()
        min_val = top_performers[coef].min()
        max_val = top_performers[coef].max()

        summary_data.append(
            {
                "Coefficient": coef,
                "Mean": mean_val,
                "Std": std_val,
                "Min": min_val,
                "Max": max_val,
            }
        )

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def main():
    coefficients_file = "coefficients.csv"
    metrics_file = "metrics.csv"

    if not os.path.exists(coefficients_file):
        print(f"Error: File '{coefficients_file}' not found.")
        return

    if not os.path.exists(metrics_file):
        print(f"Error: File '{metrics_file}' not found.")
        return

    try:
        print("Loading data...")
        coef_df, metrics_df = load_data(coefficients_file, metrics_file)

        print(f"Loaded {len(coef_df)} rows of coefficients data")
        print(f"Loaded {len(metrics_df)} rows of metrics data")

        print("Preparing data...")
        combined_df = prepare_data(coef_df, metrics_df)

        combined_df = create_unified_quality_metric(combined_df)

        dashboard_fig = create_interactive_dashboard(combined_df)

        dashboard_fig.show()

        print("\n" + "=" * 80)
        print("OPTIMAL COEFFICIENT RANGES FOR TOP 25% PERFORMERS")
        print("=" * 80)
        print(f"{'Coefficient':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 80)

        summary_df = create_top_coefficients_summary(combined_df)
        for _, row in summary_df.iterrows():
            print(
                f"{row['Coefficient']:<12} {row['Mean']:<10.2f} {row['Std']:<10.2f} {row['Min']:<10.2f} {row['Max']:<10.2f}"
            )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
