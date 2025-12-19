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


def main() -> None:
    data_path = "data/analytics/daily_revenue_weather_v1.csv"

    df = load_data(data_path)

    basic_summary(df)
    weekday_avg = weekday_summary(df)
    rain_avg = rain_summary(df)

    save_summaries(weekday_avg, rain_avg)


if __name__ == "__main__":
    main()