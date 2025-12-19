"""
V1 Definition

Question (V1):
Does daily weather (temperature, rain) correlate with daily revenue at Mosa Tea?

Unit of analysis:
1 row = 1 calendar day

Revenue definition:
Gross Order Total (USD), summed per day
Not net of refunds
Includes tax
(documented, consistent, simple)
"""
from pathlib import Path

import pandas as pd

ORDERS_PATH = Path("data/clean/orders-2025-10-01-2025-10-31_anonymized.csv")
WEATHER_PATH = Path("data/external/daily_weather.csv")
RAW_WEATHER_PATH = Path("data/raw/rawweatherdata.csv")
OUTPUT_PATH = Path("data/analytics/daily_revenue_weather_v1.csv")

WEATHER_COLUMN_ALIASES = {
    "date": ["date", "day", "obs_date"],
    "temp": ["temp", "temperature", "temp_f", "tavg"],
    "rain": ["rain", "precip", "precipitation", "rainfall", "prcp"],
}


def normalize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_cols = {col.lower(): col for col in df.columns}
    renamed = {}
    for target, aliases in WEATHER_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                renamed[lower_cols[alias]] = target
                break
    df = df.rename(columns=renamed)
    missing = [col for col in ["date", "temp", "rain"] if col not in df.columns]
    if missing:
        raise ValueError(
            "Weather file missing required columns. "
            f"Expected columns like: {', '.join(['date', 'temp', 'rain'])}. "
            f"Missing: {', '.join(missing)}."
        )
    return df[["date", "temp", "rain"]]

def load_orders(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["Order Date", "Order Total"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Orders file missing required columns: {', '.join(missing)}")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Order Total"] = pd.to_numeric(df["Order Total"], errors="coerce")

    bad_dates = df["Order Date"].isna()
    bad_totals = df["Order Total"].isna()
    if bad_dates.any() or bad_totals.any():
        df = df.loc[~(bad_dates | bad_totals)].copy()

    return df


def load_weather(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Weather file not found at {path}. "
            "Provide a daily weather CSV with columns for date, temp, and rain."
        )
    weather = pd.read_csv(path)
    weather = normalize_weather_columns(weather)
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.date
    weather["temp"] = pd.to_numeric(weather["temp"], errors="coerce")
    weather["rain"] = pd.to_numeric(weather["rain"], errors="coerce")

    if weather["date"].isna().any():
        raise ValueError("Weather file has invalid dates in the date column.")
    if weather["date"].duplicated().any():
        raise ValueError("Weather file has duplicate dates; expected 1 row per day.")

    return weather


def prepare_weather_from_raw(raw_path: Path, out_path: Path) -> Path:
    raw = pd.read_csv(raw_path)
    raw.columns = [col.strip().upper() for col in raw.columns]

    if "DATE" not in raw.columns or "PRCP" not in raw.columns:
        raise ValueError("Raw weather data must include DATE and PRCP columns.")

    raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce").dt.date
    raw["PRCP"] = pd.to_numeric(raw["PRCP"], errors="coerce")

    if "TAVG" in raw.columns:
        raw["TAVG"] = pd.to_numeric(raw["TAVG"], errors="coerce")
    if "TMAX" in raw.columns:
        raw["TMAX"] = pd.to_numeric(raw["TMAX"], errors="coerce")
    if "TMIN" in raw.columns:
        raw["TMIN"] = pd.to_numeric(raw["TMIN"], errors="coerce")

    has_tavg = "TAVG" in raw.columns
    has_tmax = "TMAX" in raw.columns
    has_tmin = "TMIN" in raw.columns

    if has_tavg:
        temp = raw["TAVG"]
        if has_tmax and has_tmin:
            temp = temp.fillna((raw["TMAX"] + raw["TMIN"]) / 2)
    elif has_tmax and has_tmin:
        temp = (raw["TMAX"] + raw["TMIN"]) / 2
    else:
        raise ValueError("Raw weather data must include TAVG or both TMAX and TMIN.")

    weather = pd.DataFrame(
        {
            "date": raw["DATE"],
            "temp": temp,
            "rain": raw["PRCP"],
        }
    ).dropna(subset=["date"])

    # If multiple rows per date (e.g., multiple stations), average them.
    weather = weather.groupby("date", as_index=False).mean(numeric_only=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(out_path, index=False)
    print(f"Prepared weather file from raw data: {out_path}")
    return out_path


def report_duplicate_totals(orders: pd.DataFrame) -> None:
    orders = orders.copy()
    orders["date"] = orders["Order Date"].dt.date
    dup_counts = orders.groupby(["date", "Order Total"]).size()
    repeat_rows = dup_counts[dup_counts > 1].sum()
    total_rows = len(orders)
    pct_repeat = (repeat_rows / total_rows * 100) if total_rows else 0

    print("Duplicate Order Total diagnostic (signal only; not proof of duplication):")
    print(f"- rows with repeated Order Total on same date: {repeat_rows} / {total_rows} ({pct_repeat:.1f}%)")
    print("- top repeats (date, order total, count):")
    top_repeats = dup_counts[dup_counts > 1].sort_values(ascending=False).head(10)
    for (date, total), count in top_repeats.items():
        print(f"  {date} | {total} | {count}")


def main() -> None:
    orders = load_orders(ORDERS_PATH)
    report_duplicate_totals(orders)

    daily = (
        orders.assign(date=orders["Order Date"].dt.date)
        .groupby("date", as_index=False)
        .agg(revenue=("Order Total", "sum"))
        .sort_values("date")
    )
    dt = pd.to_datetime(daily["date"])
    daily["weekday"] = dt.dt.weekday
    daily["weekday_name"] = dt.dt.day_name()
    daily = daily[["date", "revenue", "weekday", "weekday_name"]]

    if not WEATHER_PATH.exists() and RAW_WEATHER_PATH.exists():
        prepare_weather_from_raw(RAW_WEATHER_PATH, WEATHER_PATH)

    # Try to merge weather if present; otherwise create empty columns
    if WEATHER_PATH.exists():
        weather = load_weather(WEATHER_PATH)
        merged = daily.merge(weather, on="date", how="left")
        if merged["temp"].isna().any() or merged["rain"].isna().any():
            n_missing = merged[merged["temp"].isna() | merged["rain"].isna()].shape[0]
            print(
                "WARNING: Missing weather for "
                f"{n_missing} day(s). Check date alignment/timezone."
            )
    else:
        print(
            f"WARNING: Weather file not found at {WEATHER_PATH} "
            "and no raw weather file was available. Writing revenue-only output."
        )
        merged = daily.copy()
        merged["temp"] = pd.NA
        merged["rain"] = pd.NA

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
