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

    plt.figure(figsize=(8, 4.5))
    plt.errorbar(
        plot_df["coef"],
        plot_df["feature"],
        xerr=plot_df["std_err"],
        fmt="o",
        color="#1f77b4",
        ecolor="#9ecae1",
        capsize=3,
    )
    plt.axvline(0, color="#555", linewidth=1)
    plt.title("Regression Coefficients (Weather + Weekday Controls)")
    plt.xlabel("Coefficient (Revenue USD)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "regression_coefficients_v1.png", dpi=150)
    plt.close()

    y = df["revenue"].to_numpy(dtype=float)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_hat, y, alpha=0.75, color="#2ca02c")
    min_val = float(min(y_hat.min(), y.min()))
    max_val = float(max(y_hat.max(), y.max()))
    plt.plot([min_val, max_val], [min_val, max_val], color="#333", linewidth=1)
    plt.title("Actual vs Predicted Revenue")
    plt.xlabel("Predicted Revenue (USD)")
    plt.ylabel("Actual Revenue (USD)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "regression_actual_vs_predicted_v1.png", dpi=150)
    plt.close()

    residuals = y - y_hat

    plt.figure(figsize=(6, 4.5))
    plt.scatter(y_hat, residuals, alpha=0.75, color="#d62728")
    plt.axhline(0, color="#333", linewidth=1)
    plt.title("Residuals vs Fitted")
    plt.xlabel("Fitted Revenue (USD)")
    plt.ylabel("Residual (Actual - Fitted)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "regression_residuals_vs_fitted_v1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    plt.hist(residuals, bins=12, color="#ff7f0e", alpha=0.85, edgecolor="white")
    plt.title("Residual Distribution")
    plt.xlabel("Residual (USD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "regression_residuals_hist_v1.png", dpi=150)
    plt.close()

    def scatter_with_trendline(x: np.ndarray, y_vals: np.ndarray, x_label: str, out_name: str) -> None:
        plt.figure(figsize=(6, 4.5))
        plt.scatter(x, y_vals, alpha=0.75, color="#1f77b4")
        if len(x) > 1 and np.isfinite(x).all() and np.isfinite(y_vals).all():
            slope, intercept = np.polyfit(x, y_vals, 1)
            x_line = np.linspace(float(x.min()), float(x.max()), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, color="#111", linewidth=1.5)
        plt.title(f"Revenue vs {x_label}")
        plt.xlabel(x_label)
        plt.ylabel("Revenue (USD)")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / out_name, dpi=150)
        plt.close()

    scatter_with_trendline(
        df["temp"].to_numpy(dtype=float),
        y,
        "Temperature (F)",
        "scatter_revenue_vs_temp_v1.png",
    )
    scatter_with_trendline(
        df["rain"].to_numpy(dtype=float),
        y,
        "Rain",
        "scatter_revenue_vs_rain_v1.png",
    )

    print(f"\nSaved regression plots to {out_dir}/")


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


if __name__ == "__main__":
    main()
