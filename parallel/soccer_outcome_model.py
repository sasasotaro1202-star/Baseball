"""1X2 outcome labeling with the regulation-time draw rule.
ET/PK winner is ignored -- label always derived from regulation-time score."""
import pandas as pd


def label_outcome(row):
    if row["home_score"] > row["away_score"]:
        return "H"
    if row["home_score"] < row["away_score"]:
        return "A"
    return "D"


def apply_regulation_time_labels(df):
    df = df.copy()
    if "extra_time_winner" in df.columns or "penalty_winner" in df.columns:
        df["_note"] = "extra_time/penalty winner present but ignored per output spec"
    df["outcome_label"] = df.apply(label_outcome, axis=1)
    return df


def three_way_probabilities_pct(prob_home, prob_draw, prob_away):
    return {
        "home_pct": round(prob_home * 100, 2),
        "draw_pct": round(prob_draw * 100, 2),
        "away_pct": round(prob_away * 100, 2),
    }
