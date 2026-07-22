# Smarter Subrogation: Predicting Recovery Opportunities

**2025 Travelers University Modeling Competition - "Data & Dreams"**
Team: Debapriya Chatterjee, Pooja Raj Lakshmi, Vishal Reddy - University of Connecticut

## The business problem

When TriGuard pays out a claim where its policyholder isn't at fault, it may
be able to recover ("subrogate") that payout from the other party. Chasing
every claim is expensive; chasing none leaves money on the table. This
project builds a classifier that predicts which claims are likely to be
worth pursuing, so investigators can prioritize the highest-value cases.

## Data quality findings ("time travel glitches")

Before modeling, exploratory analysis surfaced several systemic data issues,
each of which fed directly into feature engineering rather than being
silently dropped:

| Issue | Finding | Action taken |
|---|---|---|
| Vehicle made-year | ~92% of vehicles show a made-year in the future relative to the claim | Flagged (`invalid_vehicle_year`); `vehicle_age` kept as a feature because it retained predictive signal despite being technically wrong |
| License age | ~15% of drivers appear to have gotten a license before they were old enough (worst case: -20 years of "experience") | Flagged (`invalid_license_age`) as a potential fraud/data-integrity signal |
| Driver age | Ages ranging from 8 to 241 years old (650 records, 3.61%) | Capped to a plausible [15, 100] range |
| Claim date / day-of-week | Dataset described as 2020-2021 claims; actual dates are 2015-2016, and the reported day-of-week frequently doesn't match the date | Day of week recomputed directly from `claim_date` rather than trusting the provided field |

The single strongest predictor turned out to be the simplest: **policyholder
liability percentage**. Binning it into quartiles confirmed the hypothesis
that low-liability claims subrogate far more often (>36% subrogation rate
under 25% liability) than high-liability claims (~0% above 75%).

## Modeling approach

- **Model**: LightGBM, chosen over XGBoost and CatBoost for comparable-or-better
  F1, 5-10x faster hyperparameter search, ~50% lower memory footprint, and
  native handling of 20+ categorical features without one-hot encoding.
- **Hyperparameter search**: Optuna, 100 trials, 10-fold CV, tuning tree
  structure, regularization, and `scale_pos_weight` (rather than a fixed
  77/23 class-ratio weight) to handle class imbalance.
- **Calibration**: Cross-validated isotonic regression transforms raw
  LightGBM scores into probabilities that reflect true likelihood — needed
  because a raw ranking score is not something a business stakeholder can
  interpret as "percent chance of recovery."
- **Threshold selection**: Repeated Stratified K-Fold CV (5 splits x 5
  repeats = 25 iterations) across thresholds from 0.10 to 0.90, then the
  **1-SE rule**: instead of the raw F1-maximizing threshold, we select a
  more conservative threshold whose mean F1 is within one standard error of
  the best, trading a hair of peak performance for materially better
  stability across folds.
- **Variance reduction**: the final recommendation is an ensemble average
  across 5 random seeds to reduce prediction variability on individual claims.

### Results

| Metric | Value |
|---|---|
| Mean CV F1 | ~0.60 |
| Raw best threshold (max F1) | 0.29 |
| Selected threshold (1-SE rule) | **0.315** |
| Class balance | 77% non-subrogation / 23% subrogation |
| Threshold search grid | 0.10-0.90, step 0.01 |
| CV iterations for threshold selection | 25 (5 folds x 5 repeats) |

**Top predictors**: liability %, accident type, accident site, witness
presence, higher education indicator, vehicle mileage/age, bodyshop network
status, address/contact-info change indicators.

## What would make this model better

Variables not available in this dataset that would likely improve accuracy:
official fault determination / police reports, third-party insurance details
(carrier, coverage limits), geographic/jurisdictional recovery-rate patterns,
prior claim/subrogation history, weather and road conditions at the time of
the accident, attorney involvement, and witness statement quality.

## Known limitations

- The vehicle-made-year and license-age data-quality issues are systemic
  enough that they warrant a real data-pipeline investigation before this
  model goes fully into production, even though the corresponding engineered
  flags currently retain useful signal.
- Training data is from 2015-2016 claims (despite being described as
  2020-2021); performance may degrade if future claim patterns differ
  meaningfully from this window (distribution shift).
- No fault-determination data means the model is working with an incomplete
  picture of liability.

## Repository structure

```
.
├── config.py                # paths, feature lists, hyperparameter/config defaults
├── requirements.txt
├── data/                     # place Training_TriGuard.csv / Testing_TriGuard.csv here (gitignored)
├── src/
│   ├── features.py           # FeatureEngineer transformer (all data-quality fixes + engineered features)
│   ├── preprocessing.py      # ColumnTransformer builder (numeric passthrough + ordinal-encoded categoricals)
│   ├── pipeline.py           # MLPipeline: data prep, Optuna search, final model training
│   ├── calibration.py        # ModelCalibrator (isotonic/sigmoid probability calibration)
│   ├── threshold.py          # ThresholdOptimizer (repeated-CV 1-SE rule threshold selection)
│   ├── model_io.py           # save/load the full model package
│   └── data_quality.py       # EDA diagnostics reproducing the "time travel glitches" analysis
├── scripts/
│   ├── run_eda.py            # python scripts/run_eda.py
│   ├── train.py              # python scripts/train.py --trials 100
│   └── predict.py            # python scripts/predict.py --test-csv data/Testing_TriGuard.csv
├── notebooks/
│   └── model_development.ipynb   # narrative walkthrough using the src/ modules
└── outputs/                  # plots, saved model, submission CSV (gitignored, dirs kept)
```

## Setup

**Option A — conda (recommended, keeps this out of your `base` environment):**

```bash
conda env create -f environment.yml
conda activate subrogation-model
```

**Option B - venv + pip:**

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Either way, place the competition CSVs in `data/` (see `data/README.md`).

## Usage

```bash
# 1. Data quality report (reproduces the deck's "time travel glitches" analysis)
python scripts/run_eda.py

# 2. Train: Optuna search -> final model -> calibration -> 1-SE threshold -> save package
python scripts/train.py --trials 100 --cv-folds 10

# 3. Predict on the test set and write a submission file
python scripts/predict.py --test-csv data/Testing_TriGuard.csv
```

All three scripts read their defaults from `config.py`; every path and
hyperparameter can also be overridden via CLI flags (`--help` on any script
for the full list).

## Recommendations

1. Investigate and resolve the vehicle-made-year and claim-date data quality
   issues at the source.
2. Acquire fault-determination and third-party insurance data.
3. Monitor for distribution drift given the training data's 2015-2016 window.
4. Retrain quarterly as new claims data becomes available.
