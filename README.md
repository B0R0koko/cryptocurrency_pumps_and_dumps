# Mitigating Class Imbalance in Pump-and-Dump Detection

A Comparative Analysis of Imbalance-Aware Algorithms on Binance P&D events.

<p align="center">
  <img src="props/pump_and_dump_wojak.png" alt="Pump and dump wojak illustration with BTC candlestick charts" width="720">
</p>

## What is a pump-and-dump?

A pump-and-dump (P&D) is a coordinated market-manipulation scheme. Organizers accumulate a cheap, thinly traded asset in advance, then a Telegram/Discord channel broadcasts a synchronized "buy now" signal to thousands of followers at a pre-announced timestamp. The sudden demand spike (the **pump**) inflates the price within seconds. Insiders sell into that retail bid, collapsing the price over the next few minutes (the **dump**) and leaving late buyers with losses. In crypto these schemes target low-cap Binance tokens because order books are shallow enough for a few thousand USDT of coordinated flow to move the mid-price by 5% or more in under a minute.

The image above shows a real test-set event from our dataset: a 4% jump at the announcement (left panel) followed by a 30% drawdown within 40 seconds (right panel).

The detection task in this repository is the **target-prediction** variant: given an announcement is about to happen, rank all eligible BTC-quoted pairs on Binance by their probability of being the manipulation target, using only microstructure signals observable **15 minutes** *before* the announcement (order-flow imbalance, slippage, multi-day trade counts, lifetime prior-pump count). The test cross-section averages about 288 assets, with one positive target, so class-imbalance handling drives most of the modeling choices below.

## About this project

This repository contains the code, data manifests, and experiments backing the IEEE Access submission *"Mitigating Class Imbalance in Pump-and-Dump Detection"* (Access-2026-02451). It predicts which ticker in a given cross-section is the pump target, using microstructure features and ranking-aware learning on a panel with a negative/positive imbalance ratio of about 154–287 (train → test).

## Highlights

- **Problem.** For each announced P&D event on Binance, pick the manipulated ticker from BTC-quoted pairs with at least one trade in the exact past-only eligibility window (train avg. 155, validation 237, test 288 candidates per event).
- **Dataset.** The manifest contains 498 unique Binance BTC-pair announcements (Jan 2018 – Aug 2022). Required pre-decision history is available for 473 events, split into 290 train / 103 validation / 80 test positives. No event is filtered using a post-announcement outcome.
- **Approach.** Cross-sectionally standardized microstructure features + CatBoost classifier with `Top@k%AUC` early stopping, evaluated against logistic regression, the random forest algorithm, the CatBoost ranker, and CatBoost + SMOTE baselines.
- **Execution-aware backtest.** Square-root market-impact model (Tóth / Donier–Bonart) fitted per asset from trade-level candles (5-min pre-pump entry, 5-sec sell-only post-pump exit), translated into size-dependent VWAP slippage.

<p align="center">
  <img src="props/wojak_quant_outplays_bogdanoff.png" alt="Wojak quant builds an ML model that flags an incoming pump-and-dump and sells before the dump, outplaying the Bogdanoff organizers" width="720">
</p>

## Main Results

Evaluation on 80 held-out P&D cross-sections (test period >= 2021-05-01).

Notation is consistent throughout: `Top@k` and `Top@k%` are fixed-count and percentage hit rates; `Top@kAUC` and `Top@k%AUC` are the corresponding curve areas. This project reports and optimizes `Top@k%AUC`.

Final model selection is performed before test evaluation. Across all 11 tuned and untuned candidates, CatBoost + Top@k%AUC early stopping has the highest validation Top@k%AUC (0.667), ahead of untuned CatBoost (0.615) and tuned logistic regression (0.608). That frozen validation winner is the only model used for the test portfolio; test comparisons of the other models are descriptive.

### Top@k accuracy (test set)

| Model                         | Top@1  | Top@2  | Top@5  | Top@10 | Top@20 | Top@30 |
|-------------------------------|:------:|:------:|:------:|:------:|:------:|:------:|
| Logistic regression + Tuned   | 0.113  | 0.175  | **0.400** | **0.575** | **0.713** | 0.750 |
| Random forest + Tuned         | 0.113  | 0.175  | 0.375  | 0.513  | 0.663  | **0.763** |
| CatBoost classifier + Tuned   | 0.088  | 0.188  | 0.375  | 0.513  | 0.650  | 0.713 |
| CatBoost + SMOTE + Tuned      | 0.075  | 0.113  | 0.213  | 0.250  | 0.338  | 0.413 |
| CatBoost ranker + Tuned       | 0.100  | 0.163  | 0.300  | 0.425  | 0.538  | 0.575 |
| **CatBoost + Top@k%AUC ES**   | **0.150** | **0.225** | 0.388 | 0.563 | 0.663 | 0.750 |

### Top@k%AUC with 95% bootstrap CIs

Top@k%AUC is integrated and normalized over the low-k% region k% ∈ (0, 20%], so it isolates the selective, economically relevant regime. Intervals are cross-section-level bootstrap (1000 iterations).

| Model                       | Top@k%AUC  | 95% CI             |
|-----------------------------|:----------:|:------------------:|
| **CatBoost + Top@k%AUC ES** | **0.674** | **[0.592, 0.755]** |
| CatBoost classifier         | 0.627     | [0.546, 0.706]     |

Tuned logistic regression has the highest test Top@k%AUC point estimate (0.695), followed by CatBoost + Top@k%AUC ES (0.674), tuned random forest (0.668), tuned CatBoost (0.664), and untuned logistic regression (0.660). The paired comparator is selected only on validation data, which selects untuned CatBoost; the pre-specified one-sided paired test favors ES at the 5% level.

### Portfolio performance (CatBoost + Top@k%AUC ES, 25 bps round-trip, no reinvestment, trade-price entry/exit)

| k  | Avg. event return | Annualized return | Annualized vol. | Sharpe |
|:--:|:-----------------:|:-----------------:|:---------------:|:------:|
| 1  | 0.0119 | 0.7382 | 0.7382 | 1.00 |
| 2  | 0.0115 | 0.7121 | 0.4131 | 1.72 |
| 5  | 0.0115 | 0.7158 | 0.2815 | **2.54** |
| 10 | 0.0068 | 0.4232 | 0.2063 | 2.05 |
| 20 | 0.0040 | 0.2456 | 0.1172 | 2.10 |
| 30 | 0.0026 | 0.1608 | 0.0801 | 2.01 |

The pre-specified k=5 portfolio maximises Sharpe (2.54). Under the fitted square-root impact model (15-min decision, first subsequent pre-announcement fill, 1-min exit), cumulative ROE declines monotonically from 0.801 at 100 USDT per trade to −0.294 at 10,000 USDT, crossing zero between 5,000 and 10,000 USDT. A BTC buy-and-hold baseline over the same event calendar delivers an annualized return of −0.444 (Sharpe −0.977).

### Key findings

1. **Cross-sectional standardization matters.** Every learned model beats the random baseline at every k; the signal is in the engineered microstructure features, not just the model.
2. **CatBoost + Top@k%AUC ES is strongest at the lowest fixed-count thresholds.** It leads Top@1 (0.150) and Top@2 (0.225), and it leads all three reported classification metrics: PR-AUC 0.084, F1 0.150, and balanced accuracy 0.574. The untuned CatBoost classifier leads Top@1% at 0.325, while tuned logistic regression leads the broader aggregate Top@k%AUC at 0.695.
3. **SMOTE fails here.** Synthetic oversampling in this high-dimensional, cross-sectionally standardized panel degrades performance: tuned SMOTE reaches Top@k%AUC 0.415 versus 0.695 for tuned logistic regression. Root causes include bounded sign-meaningful features, cross-event interpolation that breaks the cross-sectional reference, and high-dimensional sparsity with only 290 training positives.
4. **Ranker underperforms on the aggregate metric.** The tuned CatBoost ranker (YetiRank) reaches Top@k%AUC 0.555, materially below the leading classifiers. In a 1-vs-~287 test cross-section, much of the ordinal supervision concerns negative-vs-negative ordering, whereas class-weighted log-loss focuses on separating the single target.
5. **Paired-bootstrap significance.** The non-ES comparator is selected on validation only (untuned CatBoost). Paired bootstrap gives ES minus comparator Top@k%AUC of 0.0473 with two-sided 95% CI [−0.0005, 0.0938] and a one-sided p-value of 0.0235. The pre-specified one-sided test favors ES at the 5% level, while the two-sided interval narrowly includes zero.
6. **Robustness.** Retraining ES on random 70% subsets of the training set yields σ(Top@k%AUC) = 0.013 and a central 80% interval [0.662, 0.693]. The early test subperiod (May 2021 – May 2022, 76 events) scores 0.676; the late Jun–Aug 2022 subperiod contains only four events and is excluded by a `min_pumps=10` guard.
7. **Feature-window audit.** All regressors end strictly at T − 15 minutes; universe membership uses the exact past-only day; sub-hour z-scores use distinct exact windows; post-announcement outcomes never filter the sample; and validation/test values never supply training imputation priors. Top predictors include prior-pump history, multi-day trade counts, standardized quote volume, and standardized returns.

The final manuscript is in `paper/src/access.tex`, with the compiled paper at `paper/src/access.pdf`. The latexdiff comparison against the original submission is in `paper/src/access_highlighted.tex` and `paper/src/access_highlighted.pdf`.

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

- `resources/pumps.json` — 498 unique Binance BTC-pair announcement labels from Jan 2018 through Aug 2022. Each row provides a target pair, exchange, and UTC timestamp. Of these, 473 can be materialized from the local market archive; the 25 unavailable events are recorded in `resources/dropped_pumps.json`.

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

This walks `raw/binance/spot/trades/` and writes HIVE partitions under `transformed/binance/spot/trades/date=YYYY-MM-DD/symbol=<PAIR>/`. The layout enables cheap per-day/per-pair scans with Polars.

### 6. Build features

```bash
poetry run python -m features.FeatureWriter
```

For each pump event in `resources/pumps.json`, this materializes the cross-section (all tickers active in the exact day ending at `T - 15 min`) and computes microstructure features (asset returns, flow imbalance, slippage, aggressor imbalance, number of trades, etc.) at multiple offsets from 5 min to 14 days before the decision cutoff. Output: one parquet per usable event under `/data/pumps/data/features/pumps/`.

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
- `CatboostClassifierTOPKAUC` — CatBoost with `Top@k%AUC` early stopping
- `CatboostRanker` — learning-to-rank baseline

Each pipeline handles: data split (train < 2020-09-01, validation in [2020-09-01, 2021-05-01), test >= 2021-05-01), cross-sectional standardization, Optuna hyperparameter tuning (10 trials per pipeline, 1000 for CatBoost + Top@k%AUC early stopping), training, and scoring. The notebook first reports validation Top@k, Top@k%, Top@k%AUC, and classification tables/figures, selects the final model by validation Top@k%AUC, and only then opens the test split for final metrics and portfolio backtesting. Results and plots land in `notebooks/analysis_outputs/` and `notebooks/images/`.

### 8. Portfolio simulation and price-impact backtest

`notebooks/research_notebook.ipynb` runs the Top@k portfolio and fitted square-root impact analysis; `notebooks/visualisations.ipynb` regenerates the introductory event plot. The impact model is fitted per asset from trade-level candles ending at the transaction being priced; VWAP slippage for size Q is `I_vwap(Q) = (2/3) * β * sqrt(Q)`.

### 9. Compile the paper

```bash
just paper              # builds paper/src/access.pdf
just paper-highlighted  # builds the current-vs-original highlighted PDF
```

The final LaTeX source and PDF live in `paper/src/`, figures live in `paper/src/images/`, and the IEEE Access class and Type 1 fonts live in `paper/styles/`. All result tables are inlined in `access.tex`.

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
