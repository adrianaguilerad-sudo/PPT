# Portfolio — Project Context for Claude Code

## Project Overview

**Investment Portfolio Optimization & Management Tool**

An automated quantitative investment system for monthly portfolio rebalancing. Manages a diversified portfolio across European indices and ETFs, executed via Trade Republic with EUR 1,000/month contributions. Uses multi-factor ranking, shrinkage estimators, Heston SDE validation, and Optuna ensemble calibration.

---

## Directory Structure

```
Portfolio/
├── Scripts/
│   ├── MainScripts/                  # Production pipeline & orchestration
│   │   ├── Libraries/                # Core library modules (config + 4 engines)
│   │   │   ├── config.py             # Centralized config: tickers, weights, thresholds
│   │   │   ├── MarketSelection_lib.py      # MarketAnalyzer class, FF factors, ranking
│   │   │   ├── FinanceFunctions_lib.py     # Data download, James-Stein, Ledoit-Wolf
│   │   │   ├── InvestingStrategy_lib.py    # Backtesting engine, Heston filter
│   │   │   └── MarkowitzPortfolio_lib.py   # Portfolio class, Sharpe optimization
│   │   ├── Monthly_Execution_Master.py   # PRODUCTION ENTRY POINT
│   │   ├── FullMonthlyReport.py          # Simplified monthly report (static params)
│   │   ├── BackTestPortfolio.py          # Standalone backtest runner
│   │   ├── CalibratePortfolio.py         # Optuna calibration (with turnover penalty)
│   │   └── AnalyzeBackTestResults.py     # Post-analysis & visualization
│   ├── OtherScripts/                 # Utility scripts
│   │   ├── AutomatedUpdatePortfolio.py   # Live rebalancing execution
│   │   ├── InitializePortfolio.py        # One-time portfolio initialization
│   │   └── BacktestMonthlyRebalance.py   # Standalone monthly backtest
│   └── SingleFunctions/              # Experimental/modular standalone versions
├── CacheData/
│   ├── CacheDataFrames/              # Monthly backtest snapshots (.pkl, ~13 files)
│   ├── Local_Market_Cache.pkl        # Market fundamentals, prices, FX, FF factors
│   └── Ticker_Mapping_Cache.json     # Raw ticker → EUR equivalent mapping
├── Results/
│   └── BackTests/                    # Excel backtest outputs (.xlsx)
│       └── WeightOptimizations/      # Optuna HTML visualizations
├── .venv/                            # Python virtual environment
└── CLAUDE.md                         # This file
```

---

## Entry Points

| Script | Purpose | Duration |
|---|---|---|
| `Monthly_Execution_Master.py` | Full production pipeline (baseline → ensemble calibration → optimized run → live execution) | ~30–60 min |
| `FullMonthlyReport.py` | Monthly report with static config params | ~5–10 min |
| `BackTestPortfolio.py` | Backtest only | ~2–5 min |
| `CalibratePortfolio.py` | Optuna calibration with turnover penalty | ~20–40 min |
| `AnalyzeBackTestResults.py` | Post-analysis of Excel results | ~1 min |

**Activate environment (PowerShell):**
```powershell
.venv\Scripts\activate
```

---

## Core Libraries

### `config.py`
Single source of truth for all parameters:
- `TICKERS`: Full asset universe (ETFs + indices)
- `PROTECTED_TICKERS`, `VALUE_REFUGE_TICKERS`, `KNOWN_LOSERS`: Hold-forever categories
- `MONTHLY_CONTRIBUTION_EUR = 1000.0`, `TOP_CANDIDATES_COUNT = 4`
- `HOLDING_ZONE_PERCENTILE = 75`: Buffer to suppress unnecessary asset rotation
- Z-score ranking weights: Valuation 39% / Quality 33% / Market 28%
- Health thresholds: `ROA > 0%`, `Interest_Coverage > 1.5x`, must pass 5 of 7 checks

### `MarketSelection_lib.py` — `MarketAnalyzer` class
- Downloads prices (concurrent), computes Fama-French 5-factor + momentum
- Ranks candidates via Windsor Z-scores across valuation, quality, and market metrics
- RAM-caches prices and fundamentals to minimize API calls

### `FinanceFunctions_lib.py`
- `download_historical_prices_batch()`: Concurrent yfinance batch downloads
- `get_james_stein_expected_returns()`: Shrinkage estimator for expected returns
- `get_ledoit_wolf_covariance()`: Shrinkage covariance matrix
- `fetch_global_fama_french_daily()`: Downloads FF factors from Ken French library
- Handles GBP pence → GBP conversion, EUR FX normalization

### `InvestingStrategy_lib.py`
- `run_historical_backtest()`: Core 12-month simulation with monthly rebalancing
- `apply_heston_sde_filter()`: Validates stochastic volatility assumptions before accepting candidates
- `load_or_update_ticker_mapping()`: Disk-caches EUR ticker equivalents
- `GLOBAL_MERGED_DF_CACHE`: In-process RAM cache used across Optuna trials

### `MarkowitzPortfolio_lib.py` — `Portfolio` class
- `add_money()`, `add_money_to_asset_at_price()`
- `calculate_rebalance_contribution()`: Distributes monthly EUR contribution by target weight
- `optimize_maximize_sharpe_ratio()`: Tangency portfolio via scipy optimizer
- `get_current_portfolio_state()`: Live NAV using real-time prices

---

## Data Pipeline

```
yfinance / yahooquery  →  MarketAnalyzer (prices, fundamentals, FF, FX)
         ↓
Candidate Ranking (multi-factor Z-scores)
         ↓
Heston SDE Filter
         ↓
Markowitz Optimization (James-Stein + Ledoit-Wolf)
         ↓
12-Month Backtest Simulation
         ↓
Optuna Ensemble (5 models × 500 trials, TPE sampler)
         ↓
Averaged Parameters → Monthly Execution → Excel Report
```

---

## Dependencies

No `requirements.txt` exists — install manually into `.venv`. Key packages:

| Package | Role |
|---|---|
| `yfinance`, `yahooquery` | Market data |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | `LedoitWolf` covariance estimator |
| `scipy` | Optimization (Sharpe maximization) |
| `statsmodels` | Regression for factor models |
| `optuna` | Bayesian hyperparameter optimization (TPE) |
| `openpyxl` | Excel I/O for backtest reports |
| `matplotlib`, `plotly` | Visualization |
| `requests` | HTTP (FF factor downloads) |

---

## Current Portfolio Holdings

As of last update in `FullMonthlyReport.py`:

| Ticker | Exchange | Theme | Units |
|---|---|---|---|
| IDR.MC | BME | Infrastructure/EM | 418 |
| EGLN.L | LSE | Gold ETC | 410 |
| BTC-EUR | Crypto | Value Refuge | 833.5 |
| JNJ.DE | XETRA | Healthcare | 236.7 |
| VVSM.DE | XETRA | Semiconductors | 660 |
| VVMX.DE | XETRA | Rare Earths | 304 |
| URNU.L | LSE | Uranium | 29 |

---

## Key Design Decisions

- **Holding zone (75th percentile)**: Prevents unnecessary rotation due to minor ranking changes; minimizes transaction costs
- **Ensemble calibration (5 models)**: Reduces overfitting from single Optuna run
- **Shrinkage estimators**: James-Stein (returns) + Ledoit-Wolf (covariance) for robustness with small samples
- **Heston SDE validation**: Filters candidates whose realized volatility path is inconsistent with stochastic model assumptions
- **EUR normalization**: All non-EUR tickers mapped to EUR equivalents; GBP pence handled explicitly
- **Protected/refuge tickers**: Never sold regardless of ranking — held for strategic diversification

---

## Caching Strategy

| Cache | Location | Contents | Invalidation |
|---|---|---|---|
| Market fundamentals | `Local_Market_Cache.pkl` | Prices, fundamentals, FF factors, FX | Manual deletion or TTL logic inside library |
| Ticker mappings | `Ticker_Mapping_Cache.json` | EUR equivalent per raw ticker | Updated on new ticker first seen |
| Backtest data | `CacheDataFrames/*.pkl` | Monthly merged DataFrames for Optuna | One file per calendar month |
| RAM cache | In-process global dict | Merged backtest DF per Optuna process | Process exit |

---

## Testing

No unit test framework. Correctness is validated empirically:
- Backtesting via `run_historical_backtest()` with real historical data
- Optuna multi-model calibration as cross-validation proxy
- Manual inspection of Excel outputs in `Results/BackTests/`

---

## Current Status (as of 2026-05-03)

- **Active development**: Recent commits debugging backtesting heuristics
- **Backtest pipeline**: Functional; multiple daily runs being generated
- **Calibration**: Optuna ensemble working; best params saved to JSON
- **Known issues**: Heuristic logic in backtester was recently debugged (see last 3 commits)
- **No formal tests**: All validation is empirical/manual
- **Documentation**: `Investment_Strategy_Tool_Architecture_and_Technical_Document.pdf` in root
