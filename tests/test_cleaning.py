import pandas as pd
from mlpkg.cleaning import clean_train, normalise_tier


def small_df():
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R2", "R3"],
        "date": ["2026-01-01"] * 4,
        "age": [20, 30, 30, 40],
        "screen_hours": [1.0, 2.0, 2.0, 3.0],
        "apps_installed": [10, 20, 20, 30],
        "plan_gb_limit": [3072.0, 3.0, 3.0, 4.0],
        "video_minutes": [9999, 50, 50, 70],
        "wifi_hours": [1.0, 2.0, 2.0, 3.0],
        "phone_tier": ["FLAGSHIP", " mid", " mid", "bas."],
        "circle": ["Semi-urban", "Urban", "Urban", "Rural"],
        "gb_per_day": [4.0, 5.0, 5.0, 6.0],
    })


def test_normalise_tier():
    s = pd.Series(["Flagship", " mid", "bas.", "fla.", "MID"])
    assert normalise_tier(s).tolist() == ["flagship", "mid", "basic", "flagship", "mid"]


def test_clean_train_fixes_problems():
    out = clean_train(small_df())
    assert len(out) == 3
    assert out["plan_gb_limit"].max() <= 5
    assert (out["video_minutes"] < 9999).all()
    assert set(out["phone_tier"]) == {"flagship", "mid", "basic"}