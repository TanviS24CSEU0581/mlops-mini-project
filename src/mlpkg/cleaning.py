import numpy as np
import pandas as pd

TIER_MAP = {"fla": "flagship", "mid": "mid", "bas": "basic"}
SENTINEL_MINUTES = 9999
MB_PER_GB = 1024


def normalise_tier(s: pd.Series) -> pd.Series:
    """'FLAGSHIP', ' flagship', 'fla.' -> 'flagship' (same for mid, basic)."""
    out = s.str.strip().str.lower().str[:3].map(TIER_MAP)
    if out.isna().any():
        raise ValueError("Unknown phone_tier spelling found")
    return out


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the data problems found in train.csv."""
    df = df.drop_duplicates().copy()                     
    df["phone_tier"] = normalise_tier(df["phone_tier"])   

    # unit_mixup: Semi-urban typed plan_gb_limit in MB
    mask = (df["circle"] == "Semi-urban") & (df["plan_gb_limit"] > 100)
    df.loc[mask, "plan_gb_limit"] = df.loc[mask, "plan_gb_limit"] / MB_PER_GB

    # sentinel_value: 9999 in video_minutes -> median of the real values
    vm = df["video_minutes"].astype(float).replace(SENTINEL_MINUTES, np.nan)
    df["video_minutes"] = vm.fillna(vm.median())

    return df.reset_index(drop=True)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """For later.csv and holdout: these files are already clean, only normalise the tier."""
    df = df.copy()
    df["phone_tier"] = normalise_tier(df["phone_tier"])
    return df