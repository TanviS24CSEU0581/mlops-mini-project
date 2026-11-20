import pandas as pd

TRAIN = "data/raw/train.csv"
LATER = "data/raw/later.csv"
HOLDOUT = "data/raw/holdout_features.csv"

TARGET = "gb_per_day"
GROUP_COL = "circle"
DATE_COL = "date"
SKIP_TEXT = {"record_id", "date"}

SENTINELS = [-1, -9, -99, -999, -9999, 99, 999, 9999, 99999]


def load():
    return pd.read_csv(TRAIN), pd.read_csv(LATER), pd.read_csv(HOLDOUT)


def check_leaking_column(train, hold, target=TARGET):
    """Columns in train but not in holdout (excluding target), with correlation to target."""
    candidates = [c for c in train.columns if c not in hold.columns and c != target]
    result = {}
    for c in candidates:
        if pd.api.types.is_numeric_dtype(train[c]):
            result[c] = round(train[c].corr(train[target]), 4)
        else:
            result[c] = "non-numeric"
    return result


def check_unit_mixup(train, group_col=GROUP_COL):
    """Ratio of biggest to smallest per-group median, for every numeric column."""
    rows = []
    for c in train.select_dtypes("number").columns:
        med = train.groupby(group_col)[c].median().abs()
        med = med[med > 0]
        if len(med) < 2:
            continue
        rows.append((c, round(med.max() / med.min(), 2), med.idxmax(), med.idxmin()))
    out = pd.DataFrame(rows, columns=["column", "ratio", "biggest_group", "smallest_group"])
    return out.sort_values("ratio", ascending=False)


def check_sentinel_value(train):
    """Known sentinel numbers found in each numeric column, with counts."""
    rows = []
    for c in train.select_dtypes("number").columns:
        counts = train[c].value_counts()
        for v in SENTINELS:
            if v in counts.index:
                rows.append((c, v, int(counts[v])))
    return pd.DataFrame(rows, columns=["column", "value", "count"])


def check_duplicate_rows(train):
    """Number of extra copies."""
    return int(train.duplicated().sum())


def check_stuck_sensor(train, date_col=DATE_COL):
    """Longest run of identical values per numeric column, with first and last date."""
    t = train.copy()
    t[date_col] = pd.to_datetime(t[date_col])
    t = t.sort_values(date_col).reset_index(drop=True)
    rows = []
    for c in t.select_dtypes("number").columns:
        run_id = (t[c] != t[c].shift()).cumsum()
        sizes = t.groupby(run_id).size()
        longest = sizes.idxmax()
        seg = t[run_id == longest]
        rows.append((c, int(sizes.max()), seg[c].iloc[0],
                     seg[date_col].iloc[0].date(), seg[date_col].iloc[-1].date()))
    out = pd.DataFrame(rows, columns=["column", "run_length", "value", "first_date", "last_date"])
    return out.sort_values("run_length", ascending=False)


def check_target_typos(train, target=TARGET):
    """Values far above the typical target (about 10x too big)."""
    y = train[target].dropna().sort_values()
    med = y.median()
    big = y[y > 5 * med]
    return {"median": med, "n_suspicious": len(big), "top_values": y.tail(30).round(2).tolist()}


def check_spelling_variants(train, later):
    """Value counts for text columns, train vs later."""
    result = {}
    for c in train.select_dtypes(exclude="number").columns:
        if c in SKIP_TEXT:
            continue
        result[c] = {
            "train": train[c].value_counts(dropna=False).to_dict(),
            "later": later[c].value_counts(dropna=False).to_dict() if c in later.columns else None,
        }
    return result


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    train, later, hold = load()
    print("SHAPE:", train.shape)
    print(train.dtypes, "\n")

    print("=== 1. LEAKING COLUMN (in train, missing from holdout) ===")
    print(check_leaking_column(train, hold), "\n")
    print("=== 2. UNIT MIXUP (look for ratio near 10/12/60/100/1000) ===")
    print(check_unit_mixup(train), "\n")
    print("=== 3. SENTINEL VALUES ===")
    print(check_sentinel_value(train), "\n")
    print("=== 4. DUPLICATE ROWS ===")
    print(check_duplicate_rows(train), "\n")
    print("=== 5. STUCK SENSOR (look for a long run_length) ===")
    print(check_stuck_sensor(train), "\n")
    print("=== 6. TARGET TYPOS ===")
    print(check_target_typos(train), "\n")
    print("=== 7. SPELLING VARIANTS ===")
    for col, d in check_spelling_variants(train, later).items():
        print(col)
        print("  train:", d["train"])
        print("  later:", d["later"])