import json
import os

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [make_json_safe(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    else:
        return obj


def partial_corr(df, x, y, covariates):
    from sklearn.linear_model import LinearRegression

    xi, yi = df[x].values.reshape(-1, 1), df[y].values.reshape(-1, 1)
    if len(covariates) == 0:
        rx, ry = xi - xi.mean(), yi - yi.mean()
    else:
        cov = df[covariates].values
        lr = LinearRegression()
        lr.fit(cov, xi)
        rx = xi - lr.predict(cov)
        lr.fit(cov, yi)
        ry = yi - lr.predict(cov)
    r, p = stats.pearsonr(rx.ravel(), ry.ravel())
    return r, p


def analyze(input_csv, output_dir, cv_folds=10, random_state=12345678, shap_sample=200):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv).dropna()
    X = df[["n_estimators", "max_depth", "max_features"]]
    y = df["rmse"]

    print(
        "Best Trial Found: "
        + str(
            df.loc[
                df["rmse"].idxmin(),
                ["n_estimators", "max_depth", "max_features", "rmse"],
            ].to_dict()
        )
    )

    corr = df.corr(method="pearson")["rmse"].drop("rmse")
    spearman = df.corr(method="spearman")["rmse"].drop("rmse")
    kendall = df.corr(method="kendall")["rmse"].drop("rmse")
    mi = mutual_info_regression(X, y, random_state=random_state)

    corr_summary = {
        f: {
            "pearson": corr[f],
            "spearman": spearman[f],
            "kendall": kendall[f],
            "mutual_info": mi[i],
        }
        for i, f in enumerate(X.columns)
    }
    partial = {f: partial_corr(df, f, "rmse", [c for c in X.columns if c != f])[0] for f in X.columns}

    models = {
        "Linear": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
        "Poly2": Pipeline(
            [
                ("p", PolynomialFeatures(2, include_bias=False)),
                ("s", StandardScaler()),
                ("m", LinearRegression()),
            ]
        ),
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=500, random_state=random_state),
        "SVR(RBF)": Pipeline([("s", StandardScaler()), ("m", SVR(kernel="rbf", C=1.0))]),
    }

    cv = KFold(n_splits=min(cv_folds, len(df)), shuffle=True, random_state=random_state)
    perf = []
    for name, model in models.items():
        scores_r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
        scores_rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error"))
        perf.append(
            {
                "model": name,
                "r2_mean": np.mean(scores_r2),
                "rmse_mean": np.mean(scores_rmse),
            }
        )
    perf_df = pd.DataFrame(perf)
    best_name = perf_df.loc[perf_df["rmse_mean"].idxmin(), "model"]
    best_model = models[best_name].fit(X, y)

    pi = permutation_importance(
        best_model,
        X,
        y,
        n_repeats=30,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    importance = dict(zip(X.columns, pi.importances_mean, strict=False))

    try:
        if "Forest" in best_name or "Boost" in best_name:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X)
        else:
            sample_idx = np.random.choice(len(X), min(shap_sample, len(X)), replace=False)
            explainer = shap.KernelExplainer(best_model.predict, X.iloc[sample_idx, :])
            shap_values = explainer.shap_values(X.iloc[sample_idx, :], nsamples=100)
        shap_abs = np.mean(np.abs(shap_values), axis=0)
        shap_importance = dict(zip(X.columns, shap_abs, strict=False))
    except Exception:
        shap_importance = {}

    y_pred = best_model.predict(X)
    best_idx = np.argmin(y_pred)
    best_params = df.iloc[best_idx][["n_estimators", "max_depth", "max_features"]].to_dict()
    best_pred_rmse = y_pred[best_idx]
    best_true_rmse = y.iloc[best_idx]

    summary = {
        "best_model": best_name,
        "model_performance": perf_df.to_dict(orient="records"),
        "correlations": corr_summary,
        "partial_correlations": partial,
        "feature_importance": importance,
        "shap_importance": shap_importance,
        "predicted_best_params": best_params,
        "predicted_best_rmse": float(best_pred_rmse),
        "true_rmse_at_predicted_best": float(best_true_rmse),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump((make_json_safe(summary)), f, indent=2)

    print("\n=== SUMMARY REPORT ===")
    print(f"Best model: {best_name}")
    print("Avg R² and RMSE across models:")
    print(perf_df.to_string(index=False))
    print(
        "\nStrongest simple correlation:",
        max(corr_summary.items(), key=lambda x: abs(x[1]["pearson"])),
    )
    print("Most important feature (Permutation):", max(importance, key=importance.get))
    if shap_importance:
        print(
            "Most important feature (SHAP):",
            max(shap_importance, key=shap_importance.get),
        )
    print(f"\nPredicted best parameter set (lowest predicted RMSE={best_pred_rmse:.3f}): {best_params}")
    print(f"True RMSE at that point: {best_true_rmse:.3f}")
    print(f"\nSummary saved to: {os.path.abspath(os.path.join(output_dir, 'summary.json'))}")


if __name__ == "__main__":
    analyze("optuna_trials.csv", "eval", cv_folds=10, random_state=12345678, shap_sample=200)
