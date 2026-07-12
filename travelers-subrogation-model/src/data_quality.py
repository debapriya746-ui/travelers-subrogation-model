"""
Data-quality diagnostics referenced throughout the competition deck.

These are exploratory / reporting functions (not part of the modeling
pipeline itself) that reproduce the "time travel glitches" analysis:

  * Liability-percentage vs. subrogation rate (slide 3)
  * License age vs. driver age at claim -- "negative driving experience" (slide 5)
  * Claim day-of-week mismatch between reported and computed values (slide 7)
  * Driver age anomalies: 8 to 241 years old (slide 6)
  * Vehicle made-year in the future relative to the claim (slide 4)

Each function returns a small summary dict (for the printed report / README)
and optionally saves a plot.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def _with_parsed_dates(df):
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["claim_date"]):
        df["claim_date"] = pd.to_datetime(df["claim_date"], format="%m/%d/%Y", errors="coerce")
    return df


def liability_vs_subrogation_rate(df, target_col="subrogation"):
    """Reproduces slide 3: subrogation rate by policyholder-liability quartile."""
    df = df.copy()
    df["liab_bin"] = pd.cut(
        df["liab_prct"], bins=[-0.01, 25, 50, 75, 100],
        labels=["0-25%", "26-50%", "51-75%", "76-100%"],
    )
    summary = df.groupby("liab_bin", observed=True)[target_col].agg(["count", "mean"])
    summary.columns = ["total_claims", "subrogation_rate"]
    print(summary)
    return summary


def license_age_issue(df, save_path=None):
    """Reproduces slide 5: license age exceeding driver age at claim time."""
    df2 = _with_parsed_dates(df)
    df2["driver_age_at_claim"] = df2["claim_date"].dt.year - df2["year_of_born"]
    df2["license_age_issue"] = df2["age_of_DL"] > df2["driver_age_at_claim"]
    df2["driving_experience"] = df2["driver_age_at_claim"] - df2["age_of_DL"]

    problem = df2[df2["license_age_issue"]]
    valid = df2[~df2["license_age_issue"]]

    summary = {
        "total_claims": len(df2),
        "valid_pct": round(len(valid) / len(df2) * 100, 2),
        "invalid_pct": round(len(problem) / len(df2) * 100, 2),
        "worst_case_years": float(abs(problem["driving_experience"].min())) if len(problem) else 0.0,
    }
    print(f"License age issue -- invalid: {summary['invalid_pct']}% of records "
          f"(worst case: {summary['worst_case_years']:.0f} years of negative experience)")

    if save_path:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(valid["driver_age_at_claim"], valid["age_of_DL"], alpha=0.3, s=20,
                   color="#22c55e", label=f"Valid (n={len(valid):,})")
        ax.scatter(problem["driver_age_at_claim"], problem["age_of_DL"], alpha=0.9, s=40,
                   color="#ef4444", edgecolors="darkred", label=f"Invalid (n={len(problem):,})", zorder=5)
        lim = max(df2["driver_age_at_claim"].max(), df2["age_of_DL"].max())
        ax.plot([0, lim], [0, lim], "g--", linewidth=2, label="Valid: license age <= driver age")
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 50)
        ax.set_xlabel("Driver age at claim")
        ax.set_ylabel("Age when license obtained")
        ax.set_title("License age vs. driver age at claim")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    return summary


def driver_age_anomalies(df, low=15, high=100):
    """Reproduces slide 6: implausible driver ages (children, centenarians+)."""
    df2 = _with_parsed_dates(df)
    driver_age = df2["claim_date"].dt.year - df2["year_of_born"]
    implausible = (driver_age < low) | (driver_age > high)

    summary = {
        "n_implausible": int(implausible.sum()),
        "pct_implausible": round(implausible.mean() * 100, 2),
        "min_age": float(driver_age.min()),
        "max_age": float(driver_age.max()),
    }
    print(f"Driver age anomalies: {summary['n_implausible']} records "
          f"({summary['pct_implausible']}%), range {summary['min_age']:.0f}-{summary['max_age']:.0f} years")
    return summary


def vehicle_year_glitch(df):
    """Reproduces slide 4: vehicle made-year in the future relative to the claim date."""
    df2 = _with_parsed_dates(df)
    future_vehicle = df2["vehicle_made_year"] > df2["claim_date"].dt.year
    summary = {
        "n_future": int(future_vehicle.sum()),
        "pct_future": round(future_vehicle.mean() * 100, 2),
    }
    print(f"Vehicle-made-year glitch: {summary['pct_future']}% of records show a made-year "
          f"in the future relative to the claim date")
    return summary


def day_of_week_mismatch(df, save_path=None):
    """Reproduces slide 7: reported vs. computed claim day-of-week mismatch."""
    df2 = _with_parsed_dates(df)
    df2["computed_dow"] = df2["claim_date"].dt.strftime("%a")

    day_abbrev_map = {
        "Monday": "Mon", "Tuesday": "Tue", "Wednesday": "Wed",
        "Thursday": "Thu", "Friday": "Fri", "Saturday": "Sat", "Sunday": "Sun",
    }
    reported = df2["claim_day_of_week"].map(day_abbrev_map).fillna(df2["claim_day_of_week"])
    match = (reported == df2["computed_dow"]).astype(int)

    summary = {
        "total": len(df2),
        "matches": int(match.sum()),
        "match_rate_pct": round(match.mean() * 100, 2),
    }
    print(f"Day-of-week match rate: {summary['match_rate_pct']}% "
          f"({summary['matches']}/{summary['total']})")

    if save_path:
        match_label = pd.Series(match).map({1: "Match", 0: "Non-Match"})
        confusion = pd.crosstab(reported, match_label)
        for col in ["Match", "Non-Match"]:
            if col not in confusion.columns:
                confusion[col] = 0
        confusion = confusion[["Match", "Non-Match"]]

        cmap = LinearSegmentedColormap.from_list("custom", ["#90EE90", "#FF0000"], N=100)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap=cmap, linewidths=1, linecolor="black")
        plt.xlabel("Match status")
        plt.ylabel("Reported day of week")
        plt.title("Claim day-of-week: reported vs. computed")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    return summary


def run_full_report(df, plot_dir=None):
    """Run every diagnostic and print a consolidated summary (used by scripts/run_eda.py)."""
    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    print("\n[1] Liability vs. subrogation rate")
    liability_vs_subrogation_rate(df)

    print("\n[2] Vehicle made-year glitch")
    vehicle_year_glitch(df)

    print("\n[3] License age issue")
    license_age_issue(df, save_path=(plot_dir / "license_age_issue.png") if plot_dir else None)

    print("\n[4] Driver age anomalies")
    driver_age_anomalies(df)

    print("\n[5] Claim day-of-week mismatch")
    day_of_week_mismatch(df, save_path=(plot_dir / "dow_confusion_matrix.png") if plot_dir else None)

    print("\n" + "=" * 70)
