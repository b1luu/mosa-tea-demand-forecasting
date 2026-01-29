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
) -> None:
    """Fit OLS to estimate association between revenue and weather."""
    df = df.dropna(subset=["revenue", "temp", "rain", "weekday_name"]).copy()
    if df.empty:
        print("\n=== LINEAR REGRESSION (Weather -> Revenue) ===")
        print("No rows with complete revenue/weather data; skipping regression.")
        return

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
    print(metrics.to_string(index=False))
    print("\nCoefficients (weather + weekday controls):")
    print(coef_table.to_string(index=False))

    coef_table.to_csv(f"{out_dir}/regression_weather_coefficients_v1.csv", index=False)
    metrics.to_csv(f"{out_dir}/regression_weather_metrics_v1.csv", index=False)
    print("\nSaved regression outputs to data/analytics/")


def main() -> None:
    data_path = "data/analytics/daily_revenue_weather_v1.csv"

    df = load_data(data_path)

    basic_summary(df)
    weekday_avg = weekday_summary(df)
    rain_avg = rain_summary(df)

    save_summaries(weekday_avg, rain_avg)
    linear_regression_summary(df)


if __name__ == "__main__":
    main()
