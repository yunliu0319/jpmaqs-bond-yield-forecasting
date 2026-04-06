# Bond Yield Forecasting with JPMaQS Macro-Quantamental Indicators

Forecasting weekly changes in 10-year government bond yields across nine developed markets using macro-quantamental signals from the JPMaQS dataset. Built for the BFI / J.P. Morgan Macro-Quantamental Data Competition (University of Chicago, 2026).

---

## The Problem

Weekly bond yield changes are driven by multiple factors at different time scopes including economic shocks and monetary policy regimes, investors' liquidity and risk aversions concerns. 

This project builds a panel forecasting model that remains predictive across regime shifts by combining linear and non-linear learners, evaluated on a strict 52-week out-of-sample window.

---

## Approach

**Data**: 40 point-in-time JPMaQS indicators (published without look-ahead revision bias) spanning:
- Bond carry at 2Y and 5Y maturities (`DU02YCRY_NSA`, `DU05YCRY_NSA`)
- CPI inflation and 1Y inflation expectations (`CPIH_SA_P1M1ML12`, `INFE1Y_JA`)
- Equity excess returns and cross-sectional z-scores (`EQXR_NSA`, `EQXR_NSA_csz`)
- Credit-to-GDP (`PCREDITGDP_SA`)
- Three engineered features: carry slope, real yield spread, equity z-score

**Model**: Equal-weight `VotingRegressor` ensemble of:
- Ridge Regression (L2, interpretable coefficients)
- ElasticNet (L1+L2, sparse signal selection)
- HistGradientBoosting (non-linear, handles missing values natively)

**Validation**: Walk-forward `TimeSeriesSplit` (5 folds, 4-week gap) on a pooled 9-country panel — sorted by date, not country, to enforce strictly chronological folds.

**Target**: Weekly `DU10YXR_NSA` (10-year bond duration excess return, % per week) — proportional to yield changes, across USD, EUR, GBP, JPY, AUD, CAD, CHF, SEK, NOK.

---

## Results

| Metric | Value |
|---|---|
| CV RMSE (ensemble) | 0.927% ± 0.163 per week |
| OOS RMSE | 0.688% vs. 0.714% naive baseline |
| Skill over naive (RMSE) | 3.6% |
| OOS directional accuracy | **57.9%** vs. 50% naive (+7.9pp) |
| Training OOF directional accuracy | 54.5% |

OOS period: 52 weeks, March 2025 – March 2026 (468 country-week observations).

**Key finding**: ElasticNet ranked best in walk-forward CV (RMSE 0.882%) but worst OOS (1.6% skill), as its sparse coefficients overfit to the tightening regime prevalent in the training sample and lost predictive power in the subsequent cutting cycle. HistGradBoost's non-linear, full-feature structure generalized better across the regime shift and is the ensemble's primary source of OOS skill — validating the ensemble's regime-robustness design over pure CV-RMSE minimization.

---

## Selected Diagnostics

| | |
|---|---|
| ![CV RMSE by fold](cv_rmse.png) | ![Feature importance](feature_importance.png) |
| ![OOS vs actual](oos_vs_actual.png) | ![Predictions heatmap](predictions_heatmap.png) |

Top predictors by Ridge coefficient magnitude: equity excess return (0.122), 5Y bond carry (0.099), 2Y bond carry (0.070), carry slope (0.070), CPI inflation (0.060). All align with the economic channels motivating the feature selection.

---

## Repo Structure

```
jpmaqs.ipynb          # Main notebook: data download → features → model → OOS predictions
predictions.txt       # 52×9 matrix of weekly OOS predictions (tab-delimited)
rationale 2.pdf       # 3-page economic rationale document
cv_rmse.png           # CV RMSE by fold and model
feature_importance.png # Ridge coefficients + permutation importance
oos_vs_actual.png     # OOS predictions vs actuals per country
predictions_heatmap.png # Heatmap of OOS prediction matrix
residuals.png         # OOF residual diagnostics
```

---

## Setup

```bash
pip install macrosynergy jupyter
```

Credentials for the JPMaQS API (client ID and secret) are required and must be placed in `credentials.json` — not included in this repo. Access can be requested via the Macrosynergy Academic Program.

```json
{
  "DQ_CLIENT_ID": "...",
  "DQ_CLIENT_SECRET": "..."
}
```

Run the notebook top to bottom to reproduce all results. Downloaded data is cached locally in `cache/` to avoid redundant API calls.

---

## Economic Rationale

Full methodology and economic justification for predictor selection, model design, and result interpretation is documented in [`rationale 2.pdf`](rationale%202.pdf), covering:
1. Predictor selection grounded in the Fisher decomposition, bond carry theory, and risk appetite channels
2. Ensemble design rationale — regime robustness over CV-RMSE minimization
3. OOS results, naive baseline comparison, per-model breakdown, and model limitations
