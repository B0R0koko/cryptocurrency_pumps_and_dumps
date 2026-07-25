# Mitigating Class Imbalance in Pump-and-Dump Detection

A Comparative Analysis of Imbalance-Aware Algorithms on Binance P&D events.

<p align="center">
  <img src="props/pump_and_dump_wojak.png" alt="Pump and dump wojak illustration with BTC candlestick charts" width="720">
</p>

## What is a pump-and-dump?

A pump-and-dump (P&D) is a coordinated market-manipulation scheme. Organizers accumulate a cheap, thinly traded asset in advance, then a Telegram/Discord channel broadcasts a synchronized "buy now" signal to thousands of followers at a pre-announced timestamp. The sudden demand spike (the **pump**) inflates the price within seconds. Insiders sell into that retail bid, collapsing the price over the next few minutes (the **dump**) and leaving late buyers with losses. In crypto these schemes target low-cap Binance tokens because order books are shallow enough for a few thousand USDT of coordinated flow to move the mid-price by 5% or more in under a minute.

The image above shows a real test-set event from our dataset: a 4% jump at the announcement (left panel) followed by a 30% drawdown within 40 seconds (right panel).

The detection task in this repository is the **target-prediction** variant: given an announcement is about to happen, rank all ~290 BTC-quoted pairs on Binance by their probability of being the manipulation target, using only microstructure signals observable **15 minutes** *before* the announcement (order-flow imbalance, slippage, multi-day trade counts, lifetime prior-pump count). This is an extreme-imbalance cross-sectional classification problem (1 positive per ~290 negatives), which is the reason class-imbalance handling drives most of the modeling choices below.

## About this project

This repository contains the code, data manifests, and experiments backing the IEEE Access submission *"Mitigating Class Imbalance in Pump-and-Dump Detection"* (Access-2026-02451). It predicts which ticker in a given cross-section is the pump target, using microstructure features and ranking-aware learning on a panel with a negative/positive imbalance ratio of 154–291 (train → test).

## Highlights

- **Problem.** For each announced P&D event on Binance, pick the manipulated ticker out of the full cross-section of BTC-quoted pairs active at that time (train avg. 155, validation 239, test 290 candidates per event).
- **Dataset.** 410 labeled Binance BTC-pair pump events after filtering (Dec 2018 – Mar 2024), split into 227 train / 103 validation / 80 test positives (survivorship filter is applied to train only, so validation and test include events whose pump did not reach top-10 by 1-minute post-pump return).
- **Approach.** Cross-sectionally standardized microstructure features + CatBoost classifier with `TOPKAUC` early stopping, evaluated against Logistic Regression, Random Forest, CatBoost Ranker, and CatBoost + SMOTE baselines.
- **Execution-aware backtest.** Square-root market-impact model (Tóth / Donier–Bonart) fitted per asset from trade-level candles (5-min pre-pump entry, 5-sec sell-only post-pump exit), translated into size-dependent VWAP slippage.

<p align="center">
  <img src="props/wojak_quant_outplays_bogdanoff.png" alt="Wojak quant builds an ML model that flags an incoming pump-and-dump and sells before the dump, outplaying the Bogdanoff organizers" width="720">
</p>

## Main Results

Evaluation on 80 held-out P&D cross-sections (test period > 2021-05-01).

### TOPK accuracy (test set)

| Model                         | TOPK@1  | TOPK@2  | TOPK@5  | TOPK@10 | TOPK@20 | TOPK@30 |
|-------------------------------|:------:|:------:|:------:|:------:|:------:|:------:|
| Logistic Regression + Tuned   | 0.113  | 0.150  | 0.375  | 0.525  | 0.663  | **0.750** |
| Random Forest + Tuned         | 0.075  | 0.175  | **0.413** | **0.550** | **0.675** | 0.738 |
| CatBoost Classifier + Tuned   | 0.125  | **0.238** | 0.388  | 0.475  | 0.638  | 0.738 |
| CatBoost + SMOTE + Tuned      | 0.063  | 0.125  | 0.238  | 0.350  | 0.438  | 0.525 |
| CatBoost Ranker + Tuned       | **0.138** | 0.163  | 0.325  | 0.425  | 0.563  | 0.663 |
| **CatBoost + TOPKAUC ES**     | 0.075  | 0.200  | 0.363  | 0.463  | 0.613  | 0.738 |

### TOPKAUC with 95% bootstrap CIs

TOPKAUC is integrated and normalized over the low-K% region K% ∈ (0, 20%], so it isolates the selective, economically relevant regime. Intervals are cross-section-level bootstrap (1000 iterations).

| Model                          | TOPKAUC   | 95% CI             |
|--------------------------------|:----------:|:------------------:|
| CatBoost + TOPKAUC ES          | 0.659      | [0.582, 0.740]     |
| **Random Forest + Tuned**      | **0.683**  | **[0.600, 0.762]** |

Logistic Regression + Tuned attains the second-highest TOPKAUC point estimate (0.680), followed by CatBoost Classifier + Tuned (0.674) and CatBoost + TOPKAUC ES (0.659); the paired ES vs. Random Forest + Tuned bootstrap test does NOT reject H0: ES > RF+Tuned at the 5% level (see below).

### Portfolio performance (CatBoost + TOPKAUC ES, 25 bps round-trip, no reinvestment, mid-price entry/exit)

| K  | Avg. trade return | Annualized return | Annualized vol. | Sharpe |
|:--:|:-----------------:|:-----------------:|:---------------:|:------:|
| 1  | 0.0135 | 0.6812 | 0.8220 | 0.83 |
| 2  | 0.0230 | 1.1593 | 0.5879 | 1.97 |
| 5  | 0.0131 | 0.6584 | 0.2801 | **2.35** |
| 10 | 0.0067 | 0.3356 | 0.1546 | 2.17 |
| 20 | 0.0048 | 0.2408 | 0.1045 | 2.30 |
| 30 | 0.0025 | 0.1270 | 0.0771 | 1.65 |

The K=5 portfolio maximises Sharpe (2.35), with K=20 close behind (2.30). Under the fitted square-root impact model (15-min entry, 1-min exit, K=5) cumulative ROE declines monotonically from 0.72 at 100 USDT per trade to −0.42 at 10,000 USDT, crossing zero between 2,000 and 5,000 USDT — the edge survives only at retail order sizes, and larger notionals would need TWAP/VWAP splitting to recover it. A BTC buy-and-hold baseline over the same event windows delivers an annualized return of −0.425 (Sharpe −0.665).

### Key findings

1. **Cross-sectional standardization matters.** Every learned model beats the random baseline at every K; the signal is in the engineered microstructure features, not just the model.
2. **CatBoost + TOPKAUC ES has the strongest validation TOPKAUC** (used to select it for the portfolio backtest) and on test set ES (0.659) is statistically indistinguishable from Random Forest + Tuned (0.683), the top point-estimate. ES also has the second-highest PR-AUC (0.075), the imbalance-aware classification metric.
3. **SMOTE fails here.** Synthetic oversampling in this high-dimensional, cross-sectionally standardized panel *degrades* performance at every TOPK threshold (tuned SMOTE reaches TOPKAUC 0.490 vs 0.683 for Random Forest + Tuned), a cautionary tale against applying imbalance tricks uncritically. Root causes: bounded sign-meaningful features, cross-sectional reference violated by nearest-neighbor interpolation across events, and high-dimensional sparsity with only 227 training positives.
4. **Ranker underperforms on the aggregate metric.** The tuned CatBoost Ranker (YetiRank) ties for the best TOPK@1 point estimate (0.138) but reaches TOPKAUC of only 0.592, materially below the class-weighted classifiers. In a 1-vs-~290 cross-section, most pairwise gradient mass is negative-vs-negative noise; class-weighted log-loss concentrates gradient on the actually relevant decision boundary.
5. **Paired-bootstrap significance.** Paired bootstrap of TOPKAUC ES vs. Random Forest + Tuned gives an observed TOPKAUC difference of −0.024 with 95% CI [−0.069, 0.018] and a one-sided p-value of 0.867 (alternative: ES > RF+Tuned). The CI includes zero, so the two models are statistically indistinguishable on aggregate TOPKAUC at the 5% level.
6. **Robustness.** Retraining ES on random 70% subsets of the training set yields σ(TOPKAUC) = 0.020 (80% in [0.635, 0.685]). The early test subperiod (May 2021 – Jun 2022, 76 events) scores 0.661, close to the full-test 0.659; the late subperiod (Jul 2022 – Mar 2024) contains only 4 events and is excluded via a `min_pumps=10` guard. All results are exactly reproducible from the released code (fixed seeds and deterministic feature ordering).
7. **Feature-window audit.** The z-score normaliser look-ahead was fixed, the feature cutoff was tightened from T − 1 hour to T − 15 minutes to match the portfolio's entry time, and the hourly normaliser bars were re-anchored to that cutoff. Top predictors are prior-pump history, long-horizon trade counts (2D, 14D, 7D), multi-day slippage imbalance, and intra-day standardized returns.

The full manuscript is in [`paper/access.pdf`](paper/access.pdf) with a latexdiff-highlighted revision version in [`paper/access_highlighted.pdf`](paper/access_highlighted.pdf).

---

## Reproducibility Guide

### 1. Environment

```bash
# Python 3.13 with Poetry
poetry install
```

All subsequent commands assume `poetry run` prefixes (or an activated `poetry shell`).

### 2. Dataset layout

Datasets are expected under `/data/pumps/data/` (configurable in `core/paths.py`):

```
/data/pumps/data/
├── raw/
│   └── binance/spot/trades/       # daily .zip from data.binance.vision
├── transformed/
│   └── binance/spot/trades/       # HIVE-partitioned parquet
├── features/                      # per-pump feature parquets
└── studies.db                     # Optuna SQLite
```

Make sure this directory exists and is writable before running the pipeline.

### 3. Get the P&D event labels

Already checked in:

- `resources/pumps.json` — 175 Telegram-curated events (Dec 2018 – Apr 2024) plus 1111 events from La Morgia et al. 2020/2021, filtered by our inclusion criteria (ticker identification, ±5 min announcement regularity, 5%/3× price-volume verification). Candidates that failed any criterion were dropped, leaving 410 valid Binance BTC-pair events.

No action needed for this step; the JSON file is versioned.

### 4. Download raw trade data from Binance

Binance publishes complete tick-level history at [data.binance.vision](https://data.binance.vision) under the `data/spot/daily/trades/<PAIR>/` prefix (daily aggregated-trades `.zip` files). We use the archive (not the REST API) because it is immutable, complete, and carries buy/sell flags needed for microstructure features.

Run the scraper to populate `raw/binance/spot/trades/`:

```bash
poetry run python -m market_data.parsers.binance.BinanceSpotTradesParser
```

This iterates all BTC-quoted pairs that appear in `resources/pumps.json` (+/- a buffer window for feature offsets) over the date range covered by the event list. Expect several hundred GB of raw zip files and a multi-hour run on a decent connection. Adjust `Bounds.for_days(...)` in `run_main()` if you want a smaller re-run.

### 5. Convert raw zips to HIVE-partitioned parquet

```bash
poetry run python -m preprocessing.run
```

This walks `raw/binance/spot/trades/` and writes `transformed/binance/spot/trades/<pair>/<date>.parquet`. The HIVE layout enables cheap per-day / per-pair scans with Polars.

### 6. Build features

```bash
poetry run python -m features.FeatureWriter
```

For each pump event in `resources/pumps.json`, this materializes the cross-section (all tickers active within the relevant window) and computes microstructure features (asset returns, flow imbalance, slippage, aggressor imbalance, number of trades, etc.) at multiple offsets from 5 min to 14 days before the announcement. Output: one parquet per event under `/data/pumps/data/features/`.

CPU-parallel via `run_parallel(cpu_count=...)`.

### 7. Train models and run the full comparison

The training notebook `notebooks/research_notebook.ipynb` orchestrates the full experiment:

```bash
poetry run jupyter lab
# open notebooks/research_notebook.ipynb, run all
```

Under the hood the notebook uses the pipelines in `backtest/pipelines/`:

- `LogisticRegression` — class-weighted baseline
- `RandomForest` — tuned baseline
- `CatboostClassifier` — tuned CatBoost classifier
- `CatboostClassifierSMOTE` — CatBoost with SMOTE oversampling
- `CatboostClassifierTOPKAUC` — CatBoost with `TOPKAUC` early stopping (our best model)
- `CatboostRanker` — learning-to-rank baseline

Each pipeline handles: data split (train < 2020-09-01, val 2020-09-01 to 2021-05-01, test > 2021-05-01), cross-sectional standardization, Optuna hyperparameter tuning (10 trials per pipeline, 1000 for CatBoost + TOPKAUC Early Stopping), training, and scoring. Results and plots land in `notebooks/analysis_outputs/` and `notebooks/images/`.

### 8. Portfolio simulation and price-impact backtest

`notebooks/visualisations.ipynb` builds the top-K portfolio under the fitted square-root impact model and produces the plots in Section IV of the paper. The impact model is fitted per asset from trade-level candles; VWAP slippage for size Q is `I_vwap(Q) = (2/3) * β * sqrt(Q)`.

### 9. Compile the paper

```bash
just paper          # builds paper/access.pdf
```

Or the highlighted revision version:

```bash
cd paper && pdflatex -interaction=scrollmode access_highlighted.tex
```

---

## Code Map

| Module           | What it owns                                                                 |
|------------------|------------------------------------------------------------------------------|
| `core/`          | Shared types (`PumpEvent`, `CurrencyPair`, `FeatureType`), paths, time utils |
| `market_data/`   | Scrapy-based Binance archive scraper                                         |
| `preprocessing/` | Raw `.zip` → HIVE-partitioned Parquet                                        |
| `features/`      | `PumpsFeatureWriter`: per-event cross-section feature materialization        |
| `backtest/pipelines/` | ML model implementations (all extend `BasePipeline`)                   |
| `backtest/portfolio/` | Execution simulation, price-impact model, VWAP slippage                |
| `backtest/utils/` | Dataset building, evaluation metrics, robustness testing                    |
| `notebooks/`     | Experiment orchestration + paper figures                                     |
| `paper/`         | IEEE Access LaTeX sources                                                    |
| `resources/`     | Event labels (`pumps.json`)                                                  |

## Development Commands

```bash
just format-all     # black (120-char lines)
just pylint         # lint
just mypy           # type checks
poetry run pytest -q
```

---

## License

Released under the [MIT License](LICENSE). Free to use, modify, and distribute, including for commercial purposes.
