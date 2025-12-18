import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)


# -----------------------------
# Config
# -----------------------------

# Raw input (you can keep this absolute path for now)
raw_path = Path("/Users/brandon/Desktop/mosa-tea-demand-forecasting/data/raw/orders-2025-10-01-2025-10-31.csv")

# Outputs
clean_path = Path("data/clean/orders-2025-10-01-2025-10-31_anonymized.csv")
mapping_dir = Path("data/private")  # DO NOT COMMIT THIS
item_mapping_path = mapping_dir / "item_name_mapping.csv"
modifier_mapping_path = mapping_dir / "item_modifier_mapping.csv"

# PII columns to drop (only drop if present)
pii_columns = [
    "Order Name",
    "Recipient Name",
    "Recipient Email",
    "Recipient Phone",
    "Recipient Address",
    "Recipient Address 2",
    "Recipient Postal Code",
    "Recipient City",
    "Recipient Region",
    "Recipient Country",
    "Fulfillment Notes",
]

# Columns you want to anonymize but still keep identifiable via IDs
# (adjust these if your Square export uses slightly different names)
SENSITIVE_CATEGORICALS = {
    "Item Name": ("Item ID", "MTI"),          # Item IDs
    "Item Modifiers": ("Modifier ID", "MTM"), # Modifier IDs (optional but recommended)
}


def make_codebook(series: pd.Series, prefix: str) -> dict:
    """
    Create a deterministic mapping:
    unique value -> PREFIX0001, PREFIX0002, ...

    Deterministic means: given the same set of values, it produces the same mapping
    because we sort the unique values first.
    """
    values = series.dropna().astype(str).unique()
    values = sorted(values)  # deterministic ordering
    return {val: f"{prefix}{str(i+1).zfill(4)}" for i, val in enumerate(values)}


def save_mapping(mapping: dict, out_path: Path, left_name: str, right_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df = pd.DataFrame(list(mapping.items()), columns=[left_name, right_name])
    mapping_df.to_csv(out_path, index=False)


def main():
    # -----------------------------
    # Load raw
    # -----------------------------
    df = pd.read_csv(raw_path)

    print("Columns in raw file:")
    for col in df.columns:
        print(col)

    # -----------------------------
    # Remove PII
    # -----------------------------
    cols_to_drop = [col for col in pii_columns if col in df.columns]
    df_clean = df.drop(columns=cols_to_drop)

    print("\nDropped PII columns:")
    for col in cols_to_drop:
        print(col)

    # -----------------------------
    # Generate IDs + mapping files
    # -----------------------------
    # We will replace sensitive categorical columns with IDs,
    # and write mapping files privately.
    mapping_dir.mkdir(parents=True, exist_ok=True)

    for source_col, (id_col, prefix) in SENSITIVE_CATEGORICALS.items():
        if source_col not in df_clean.columns:
            print(f"\n[Skip] Column not found for anonymization: {source_col}")
            continue

        codebook = make_codebook(df_clean[source_col], prefix=prefix)

        # Create ID column
        df_clean[id_col] = df_clean[source_col].astype(str).map(codebook)

        # Save mapping privately
        if source_col == "Item Name":
            save_mapping(codebook, item_mapping_path, "Item Name", id_col)
            print(f"\nSaved item mapping to: {item_mapping_path}")
        elif source_col == "Item Modifiers":
            save_mapping(codebook, modifier_mapping_path, "Item Modifiers", id_col)
            print(f"\nSaved modifier mapping to: {modifier_mapping_path}")
        else:
            # Generic fallback if you add more later
            generic_map_path = mapping_dir / f"{source_col.lower().replace(' ', '_')}_mapping.csv"
            save_mapping(codebook, generic_map_path, source_col, id_col)
            print(f"\nSaved mapping to: {generic_map_path}")

        # Drop original column after mapping
        df_clean = df_clean.drop(columns=[source_col])

    print("\nColumns AFTER cleaning + anonymization:")
    for col in df_clean.columns:
        print(col)

    # -----------------------------
    # Write anonymized clean CSV
    # -----------------------------
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(clean_path, index=False)
    print(f"\nSaved anonymized CSV to: {clean_path}")


if __name__ == "__main__":
    main()
