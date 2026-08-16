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

Final model selection is performed before test evaluation. Across all 11 tuned and untuned candidates, CatBoost + Top@k%AUC early stopping has the highest validation Top@k%AUC (0.667), ahead of tuned CatBoost (0.655) and tuned random forest (0.628). That frozen validation winner is the only model used for the test portfolio; test comparisons of the other models are descriptive.

### Top@k accuracy (test set)

| Model                         | Top@1  | Top@2  | Top@5  | Top@10 | Top@20 | Top@30 |
|-------------------------------|:------:|:------:|:------:|:------:|:------:|:------:|
| Logistic regression + Tuned   | 0.100  | 0.200  | **0.425** | **0.575** | **0.700** | **0.763** |
| Random forest + Tuned         | 0.113  | 0.175  | 0.350  | 0.500  | 0.650  | 0.713 |
| CatBoost classifier + Tuned   | 0.138  | **0.275** | 0.400  | 0.525  | 0.650  | 0.725 |
| CatBoost + SMOTE + Tuned      | 0.113  | 0.163  | 0.300  | 0.425  | 0.538  | 0.600 |
| CatBoost ranker + Tuned       | 0.063  | 0.163  | 0.275  | 0.463  | 0.613  | 0.700 |
| **CatBoost + Top@k%AUC ES**   | **0.150** | 0.225 | 0.388 | 0.563 | 0.663 | 0.750 |

### Top@k%AUC with 95% bootstrap CIs

Top@k%AUC is integrated and normalized over the low-k% region k% ∈ (0, 20%], so it isolates the selective, economically relevant regime. Intervals are cross-section-level bootstrap (1000 iterations).

| Model                       | Top@k%AUC  | 95% CI             |
|-----------------------------|:----------:|:------------------:|
| **CatBoost + Top@k%AUC ES** | **0.674** | **[0.592, 0.755]** |
| CatBoost classifier + Tuned | 0.662     | [0.575, 0.747]     |

Tuned logistic regression has the highest test Top@k%AUC point estimate (0.697), followed by CatBoost + Top@k%AUC ES (0.674), tuned CatBoost (0.662), untuned logistic regression (0.660), and tuned random forest (0.651). The paired comparator is selected only on validation data, which selects tuned CatBoost. Their 0.0116 test-set difference is not statistically significant under the pre-specified paired bootstrap test.

### Portfolio performance (CatBoost + Top@k%AUC ES, 25 bps round-trip, no reinvestment, trade-price entry/exit)

| k  | Avg. event return | Annualized return | Annualized vol. | Sharpe |
|:--:|:-----------------:|:-----------------:|:---------------:|:------:|
| 1  | 0.0123 | 0.7621 | 0.7423 | 1.03 |
| 2  | 0.0118 | 0.7338 | 0.4161 | 1.76 |
| 5  | 0.0120 | 0.7416 | 0.2896 | **2.56** |
| 10 | 0.0072 | 0.4457 | 0.2130 | 2.09 |
| 20 | 0.0043 | 0.2674 | 0.1251 | 2.14 |
| 30 | 0.0029 | 0.1828 | 0.0894 | 2.04 |

The pre-specified k=5 portfolio maximises Sharpe (2.56). Under the fitted square-root impact model (15-min decision, first subsequent pre-announcement fill, full inventory liquidation at the 1-min exit), cumulative ROE declines monotonically from 0.840 at 100 USDT per trade to −0.201 at 10,000 USDT, crossing zero between 5,000 and 10,000 USDT. A BTC buy-and-hold baseline over the same event calendar delivers an annualized return of −0.444 (Sharpe −0.977).

### Key findings

1. **Cross-sectional standardization matters.** Every learned model beats the random baseline at every k; the signal is in the engineered microstructure features, not just the model.
2. **CatBoost + Top@k%AUC ES is strongest at the most selective fixed-count threshold.** It leads Top@1 (0.150), F1 (0.150), and balanced accuracy (0.574). Tuned CatBoost leads PR-AUC (0.087), Top@2 (0.275), and Top@1% (0.338), while tuned logistic regression leads the broader aggregate Top@k%AUC at 0.697.
3. **SMOTE remains weaker.** Even after 100 tuning trials, tuned SMOTE reaches Top@k%AUC 0.574 versus 0.697 for tuned logistic regression. Likely contributors include bounded sign-meaningful features, cross-event interpolation that breaks the cross-sectional reference, and high-dimensional sparsity with only 290 training positives.
4. **The tuned ranker is competitive but not leading.** The CatBoost ranker (YetiRank), trained on highest-first five-minute return ranks, reaches Top@k%AUC 0.629 after 100 trials. It closes much of the gap to the classifiers but remains below the validation-selected CatBoost + ES model (0.674) and tuned logistic regression (0.697).
5. **Paired-bootstrap uncertainty.** The non-ES comparator selected on validation is tuned CatBoost. Paired bootstrap gives ES minus comparator Top@k%AUC of 0.0116 with two-sided 95% CI [−0.0120, 0.0369] and a one-sided p-value of 0.174; the models are not statistically distinguishable at the 5% level.
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

Each pipeline handles: data split (train < 2020-09-01, validation in [2020-09-01, 2021-05-01), test >= 2021-05-01), cross-sectional standardization, 100-trial Optuna hyperparameter tuning, training, and scoring. The notebook first reports validation Top@k, Top@k%, Top@k%AUC, and classification tables/figures, selects the final model by validation Top@k%AUC, and only then opens the test split for final metrics and portfolio backtesting. Results and plots land in `notebooks/analysis_outputs/` and `notebooks/images/`.

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
