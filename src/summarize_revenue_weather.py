from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load merged daily revenue + weather data."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def basic_summary(df: pd.DataFrame) -> None:
    """Print basic descriptive statistics."""
    print("\n=== BASIC STATS (Revenue, Temp, Rain) ===")
    print(df[["revenue", "temp", "rain"]].describe())


def weekday_summary(df: pd.DataFrame) -> pd.Series:
    """Compute average revenue by weekday."""
    print("\n=== AVERAGE REVENUE BY WEEKDAY ===")

    summary = (
        df.groupby("weekday_name")["revenue"]
        .mean()
        .sort_values(ascending=False)
    )

    print(summary)
    return summary


def rain_summary(df: pd.DataFrame) -> pd.Series:
    """Compare revenue on rainy vs non-rainy days."""
    print("\n=== RAIN VS NO RAIN ===")

    df = df.copy()
    df["is_rain"] = (df["rain"] > 0).astype(int)

    summary = df.groupby("is_rain")["revenue"].mean()
    summary.index = ["No Rain", "Rain"]

    print(summary)
    return summary


def save_summaries(
    weekday_summary: pd.Series,
    rain_summary: pd.Series,
    out_dir: str = "data/analytics",
) -> None:
    """Save summary tables to CSV."""
    weekday_summary.to_csv(f"{out_dir}/summary_revenue_by_weekday.csv")
    rain_summary.to_csv(f"{out_dir}/summary_revenue_rain_vs_no_rain.csv")
    print("\nSaved summary CSV files to data/analytics/")


def build_design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build regression matrix with weather + weekday controls."""
    df = df.copy()
    weekday_dummies = pd.get_dummies(df["weekday_name"], prefix="weekday", drop_first=True)
    weekday_dummies = weekday_dummies.astype(float)

    feature_frames = [
        pd.Series(1.0, index=df.index, name="intercept"),
        pd.to_numeric(df["temp"], errors="coerce").rename("temp"),
        pd.to_numeric(df["rain"], errors="coerce").rename("rain"),
        weekday_dummies,
    ]
    X_df = pd.concat(feature_frames, axis=1)
    return X_df.to_numpy(dtype=float), X_df.columns.tolist()


def linear_regression_summary(
    df: pd.DataFrame,
    out_dir: str = "data/analytics",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray] | None:
    """Fit OLS to estimate association between revenue and weather."""
    df = df.dropna(subset=["revenue", "temp", "rain", "weekday_name"]).copy()
    if df.empty:
        print("\n=== LINEAR REGRESSION (Weather -> Revenue) ===")
        print("No rows with complete revenue/weather data; skipping regression.")
        return None

    X, feature_names = build_design_matrix(df)
    y = df["revenue"].to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta

    residuals = y - y_hat
    sse = float(np.sum(residuals ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst else float("nan")
    rmse = float(np.sqrt(sse / len(y)))

    n = len(y)
    p = X.shape[1]
    dof = max(n - p, 0)
    if dof > 0:
        sigma2 = sse / dof
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * xtx_inv))
        t_stat = np.where(se > 0, beta / se, np.nan)
    else:
        se = np.full_like(beta, np.nan, dtype=float)
        t_stat = np.full_like(beta, np.nan, dtype=float)

    coef_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": beta,
            "std_err": se,
            "t_stat": t_stat,
        }
    )

    metrics = pd.DataFrame(
        [
            {
                "rows": n,
                "features": p,
                "r2": r2,
                "rmse": rmse,
                "mean_revenue": float(y.mean()),
            }
        ]
    )

    print("\n=== LINEAR REGRESSION (Weather -> Revenue) ===")
    print("NOTE: This analysis is NOT causal, likely low value, and highly likely discontinued.")
    print(metrics.to_string(index=False))
    print("\nCoefficients (weather + weekday controls):")
    print(coef_table.to_string(index=False))

    coef_table.to_csv(f"{out_dir}/regression_weather_coefficients_v1.csv", index=False)
    metrics.to_csv(f"{out_dir}/regression_weather_metrics_v1.csv", index=False)
    print("\nSaved regression outputs to data/analytics/")
    return df, coef_table, y_hat


def save_regression_plots(
    df: pd.DataFrame,
    coef_table: pd.DataFrame,
    y_hat: np.ndarray,
    out_dir: str = "data/analytics/figures",
    name_prefix: str = "",
    title_prefix: str = "",
    std_err_column: str = "std_err",
) -> None:
    """Save basic regression visualizations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plots. Install with: pip install matplotlib")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    plot_df = coef_table[coef_table["feature"] != "intercept"].copy()
    plot_df = plot_df.sort_values("coef")
    prefix = f"{name_prefix}_" if name_prefix else ""

    plt.figure(figsize=(8, 4.5))
    if std_err_column not in plot_df.columns:
        raise KeyError(f"Missing expected std error column: {std_err_column}")

    plt.errorbar(
        plot_df["coef"],
        plot_df["feature"],
        xerr=plot_df[std_err_column],
        fmt="o",
        color="#1f77b4",
        ecolor="#9ecae1",
        capsize=3,
    )
    plt.axvline(0, color="#555", linewidth=1)
    plt.title(f"{title_prefix}Regression Coefficients (Weather + Weekday Controls)")
    plt.xlabel("Coefficient (Revenue USD)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{prefix}regression_coefficients_v1.png", dpi=150)
    plt.close()

    y = df["revenue"].to_numpy(dtype=float)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_hat, y, alpha=0.75, color="#2ca02c")
    min_val = float(min(y_hat.min(), y.min()))
    max_val = float(max(y_hat.max(), y.max()))
    plt.plot([min_val, max_val], [min_val, max_val], color="#333", linewidth=1)
    plt.title(f"{title_prefix}Actual vs Predicted Revenue")
    plt.xlabel("Predicted Revenue (USD)")
    plt.ylabel("Actual Revenue (USD)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{prefix}regression_actual_vs_predicted_v1.png", dpi=150)
    plt.close()

    residuals = y - y_hat

    plt.figure(figsize=(6, 4.5))
    plt.scatter(y_hat, residuals, alpha=0.75, color="#d62728")
    plt.axhline(0, color="#333", linewidth=1)
    plt.title(f"{title_prefix}Residuals vs Fitted")
    plt.xlabel("Fitted Revenue (USD)")
    plt.ylabel("Residual (Actual - Fitted)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{prefix}regression_residuals_vs_fitted_v1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    plt.hist(residuals, bins=12, color="#ff7f0e", alpha=0.85, edgecolor="white")
    plt.title(f"{title_prefix}Residual Distribution")
    plt.xlabel("Residual (USD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{prefix}regression_residuals_hist_v1.png", dpi=150)
    plt.close()

    def scatter_with_trendline(x: np.ndarray, y_vals: np.ndarray, x_label: str, out_name: str) -> None:
        plt.figure(figsize=(6, 4.5))
        plt.scatter(x, y_vals, alpha=0.75, color="#1f77b4")
        if len(x) > 1 and np.isfinite(x).all() and np.isfinite(y_vals).all():
            slope, intercept = np.polyfit(x, y_vals, 1)
            x_line = np.linspace(float(x.min()), float(x.max()), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, color="#111", linewidth=1.5)
        plt.title(f"{title_prefix}Revenue vs {x_label}")
        plt.xlabel(x_label)
        plt.ylabel("Revenue (USD)")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / out_name, dpi=150)
        plt.close()

    scatter_with_trendline(
        df["temp"].to_numpy(dtype=float),
        y,
        "Temperature (F)",
        f"{prefix}scatter_revenue_vs_temp_v1.png",
    )
    scatter_with_trendline(
        df["rain"].to_numpy(dtype=float),
        y,
        "Rain",
        f"{prefix}scatter_revenue_vs_rain_v1.png",
    )

    print(f"\nSaved regression plots to {out_dir}/")


def monte_carlo_weather_effect(
    df: pd.DataFrame,
    n_iter: int = 2000,
    seed: int = 42,
    out_dir: str = "data/analytics",
) -> None:
    """Permutation-based Monte Carlo test for weather coefficients."""
    df = df.dropna(subset=["revenue", "temp", "rain", "weekday_name"]).copy()
    if df.empty:
        print("\n=== MONTE CARLO (Weather Relation) ===")
        print("No rows with complete revenue/weather data; skipping simulation.")
        return

    rng = np.random.default_rng(seed)

    X_obs, feature_names = build_design_matrix(df)
    y = df["revenue"].to_numpy(dtype=float)
    beta_obs, *_ = np.linalg.lstsq(X_obs, y, rcond=None)

    temp_idx = feature_names.index("temp")
    rain_idx = feature_names.index("rain")

    null_coefs = np.zeros((n_iter, 2), dtype=float)
    weather_values = df[["temp", "rain"]].to_numpy(dtype=float)

    for i in range(n_iter):
        perm_idx = rng.permutation(len(df))
        permuted = df.copy()
        permuted[["temp", "rain"]] = weather_values[perm_idx]
        X_perm, _ = build_design_matrix(permuted)
        beta_perm, *_ = np.linalg.lstsq(X_perm, y, rcond=None)
        null_coefs[i, 0] = beta_perm[temp_idx]
        null_coefs[i, 1] = beta_perm[rain_idx]

    obs_temp = beta_obs[temp_idx]
    obs_rain = beta_obs[rain_idx]
    p_temp = float((np.abs(null_coefs[:, 0]) >= abs(obs_temp)).mean())
    p_rain = float((np.abs(null_coefs[:, 1]) >= abs(obs_rain)).mean())

    summary = pd.DataFrame(
        [
            {
                "feature": "temp",
                "observed_coef": obs_temp,
                "null_mean": float(np.mean(null_coefs[:, 0])),
                "null_std": float(np.std(null_coefs[:, 0], ddof=1)),
                "empirical_p_two_sided": p_temp,
                "n_iter": n_iter,
            },
            {
                "feature": "rain",
                "observed_coef": obs_rain,
                "null_mean": float(np.mean(null_coefs[:, 1])),
                "null_std": float(np.std(null_coefs[:, 1], ddof=1)),
                "empirical_p_two_sided": p_rain,
                "n_iter": n_iter,
            },
        ]
    )

    null_df = pd.DataFrame(null_coefs, columns=["temp_coef", "rain_coef"])
    null_df.to_csv(f"{out_dir}/monte_carlo_weather_null_v1.csv", index=False)
    summary.to_csv(f"{out_dir}/monte_carlo_weather_summary_v1.csv", index=False)

    print("\n=== MONTE CARLO (Weather Relation) ===")
    print("Permutation test that breaks weather->revenue alignment, controlling weekday.")
    print(summary.to_string(index=False))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping Monte Carlo plots.")
        return

    out_fig_dir = Path(out_dir) / "figures"
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4.5))
    plt.hist(null_coefs[:, 0], bins=20, color="#6baed6", alpha=0.85, edgecolor="white")
    plt.axvline(obs_temp, color="#111", linewidth=1.5)
    plt.title("Monte Carlo Null: Temp Coef")
    plt.xlabel("Coefficient (Revenue USD / F)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_fig_dir / "monte_carlo_temp_coef_v1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    plt.hist(null_coefs[:, 1], bins=20, color="#9ecae1", alpha=0.85, edgecolor="white")
    plt.axvline(obs_rain, color="#111", linewidth=1.5)
    plt.title("Monte Carlo Null: Rain Coef")
    plt.xlabel("Coefficient (Revenue USD / Rain)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_fig_dir / "monte_carlo_rain_coef_v1.png", dpi=150)
    plt.close()

    print(f"\nSaved Monte Carlo outputs to {out_dir}/ and {out_fig_dir}/")


def fit_robust_regression(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    huber_k: float = 1.345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively reweighted least squares with Huber weights."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    for _ in range(max_iter):
        residuals = y - X @ beta
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        scale = 1.4826 * mad

        if scale <= 1e-12:
            break

        abs_r = np.abs(residuals)
        threshold = huber_k * scale
        weights = np.ones_like(residuals)
        mask = abs_r > threshold
        weights[mask] = threshold / abs_r[mask]

        Xw = X * np.sqrt(weights)[:, None]
        yw = y * np.sqrt(weights)
        beta_new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break

        beta = beta_new

    residuals = y - X @ beta
    return beta, weights, residuals


def robust_regression_summary(
    df: pd.DataFrame,
    out_dir: str = "data/analytics",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray] | None:
    """Fit robust regression (Huber) to reduce outlier influence."""
    df = df.dropna(subset=["revenue", "temp", "rain", "weekday_name"]).copy()
    if df.empty:
        print("\n=== ROBUST REGRESSION (Huber) ===")
        print("No rows with complete revenue/weather data; skipping regression.")
        return None

    X, feature_names = build_design_matrix(df)
    y = df["revenue"].to_numpy(dtype=float)

    beta, weights, residuals = fit_robust_regression(X, y)
    y_hat = X @ beta

    sse = float(np.sum(weights * (residuals ** 2)))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - y_hat) ** 2)) / sst if sst else float("nan")
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    n = len(y)
    p = X.shape[1]
    dof = max(n - p, 0)
    if dof > 0:
        sigma2 = sse / dof
        xtwx_inv = np.linalg.pinv(X.T @ (weights[:, None] * X))
        se = np.sqrt(np.diag(sigma2 * xtwx_inv))
        t_stat = np.where(se > 0, beta / se, np.nan)
    else:
        se = np.full_like(beta, np.nan, dtype=float)
        t_stat = np.full_like(beta, np.nan, dtype=float)

    coef_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": beta,
            "std_err_wls": se,
            "t_stat_wls": t_stat,
        }
    )

    metrics = pd.DataFrame(
        [
            {
                "rows": n,
                "features": p,
                "r2": r2,
                "rmse": rmse,
                "mean_revenue": float(y.mean()),
                "avg_weight": float(np.mean(weights)),
            }
        ]
    )

    print("\n=== ROBUST REGRESSION (Huber) ===")
    print("NOTE: Std errors/t-stats are WLS approximations, not fully robust.")
    print(metrics.to_string(index=False))
    print("\nCoefficients (robust, weather + weekday controls):")
    print(coef_table.to_string(index=False))

    coef_table.to_csv(f"{out_dir}/regression_weather_coefficients_robust_v1.csv", index=False)
    metrics.to_csv(f"{out_dir}/regression_weather_metrics_robust_v1.csv", index=False)
    print("\nSaved robust regression outputs to data/analytics/")
    return df, coef_table, y_hat


def main() -> None:
    data_path = "data/analytics/daily_revenue_weather_v1.csv"

    df = load_data(data_path)

    basic_summary(df)
    weekday_avg = weekday_summary(df)
    rain_avg = rain_summary(df)

    save_summaries(weekday_avg, rain_avg)
    regression = linear_regression_summary(df)
    if regression is not None:
        df_used, coef_table, y_hat = regression
        save_regression_plots(df_used, coef_table, y_hat)
    robust = robust_regression_summary(df)
    if robust is not None:
        df_used, coef_table, y_hat = robust
        save_regression_plots(
            df_used,
            coef_table,
            y_hat,
            name_prefix="robust",
            title_prefix="Robust ",
            std_err_column="std_err_wls",
        )
    monte_carlo_weather_effect(df)


if __name__ == "__main__":
    main()
