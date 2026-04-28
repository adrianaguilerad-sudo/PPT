import os
# CRITICAL: Disable internal multithreading for math libraries BEFORE importing them.
# This prevents 256 math threads from crashing the CPU scheduler when n_jobs=-1.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import optuna
import numpy as np
from InvestingStrategy_lib import (
    run_historical_backtest, PROTECTED_TICKERS, NOT_AVAIABLE_TICKERS, 
    load_or_update_ticker_mapping, FUNDAMENTAL_WINDOW_YEARS
)
from MarketSelection_lib import get_index_tickers, MarketAnalyzer
import logging
import sys
import optuna.visualization as vis
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
# CalibratePortfolio.py (Relevant section in objective function)
from config import DEFAULT_WEIGHTS_CONFIG, DEFAULT_HEALTH_CONFIG

optuna.logging.get_logger("optuna").setLevel(logging.WARNING)

TURNOVER_PENALTY_WEIGHT = 0.2 

# --- WORKER SINGLETONS ---
# These globals stay in the RAM of each of the 16 independent processes.
# They ensure the heavy .pkl file is unpickled only ONCE per CPU core.
WORKER_ANALYZER = None
WORKER_MAPPING = None

import json
import os

def load_optimized_parameters(json_file_path="best_params.json"):
    """
    Loads optimized hyperparameters from Optuna's JSON output and reconstructs
    the structured dictionaries required by the investment strategy.
    """
    if not os.path.exists(json_file_path):
        print(f"[ERROR] Optimization file '{json_file_path}' not found.")
        return None, None, None, None

    with open(json_file_path, 'r') as f:
        best_params = json.load(f)

    # 1. Extract General Portfolio Settings
    top_candidates = best_params.get("top_candidates", 5)
    holding_zone = best_params.get("holding_zone", 75)

    # 2. Reconstruct Weights Dictionary (Filter keys with prefix 'w_')
    custom_weights_config = {
        key.replace("w_", ""): value 
        for key, value in best_params.items() 
        if key.startswith("w_")
    }
    
    # Re-inject static weights that were excluded from optimization
    custom_weights_config["PER"] = 0.0
    custom_weights_config["Price_Book"] = 0.0

    # 3. Reconstruct Health Parameters Dictionary (Filter keys with prefix 'h_')
    health_params_config = {
        "roa_min": best_params.get("h_roa_min", -0.05),
        "ebit_min": best_params.get("h_ebit_min", 0.01),
        "interest_coverage_min": best_params.get("h_int_cov_min", 3.0),
        "pass_threshold_financial": best_params.get("h_pass_fin", 5),
        "pass_threshold_standard": best_params.get("h_pass_std", 6)
    }

    print("[INFO] Successfully loaded and structured optimized parameters from JSON.")
    return top_candidates, holding_zone, custom_weights_config, health_params_config

def objective(trial, target_indices):
    global WORKER_ANALYZER, WORKER_MAPPING
    
    # Lazy initialization: Only load from disk the very first time this core runs a trial
    if WORKER_ANALYZER is None or WORKER_MAPPING is None:
        WORKER_MAPPING = load_or_update_ticker_mapping(target_indices)
        
        filtered_indices_dict = {}
        for index_name, tickers in WORKER_MAPPING.items():
            filtered_indices_dict[index_name] = [t for t in tickers if t not in NOT_AVAIABLE_TICKERS]
            
        today = datetime.now()
        start_simulation_date = today - relativedelta(months=12)
        historical_data_start = start_simulation_date - relativedelta(years=FUNDAMENTAL_WINDOW_YEARS)
        
        WORKER_ANALYZER = MarketAnalyzer(
            start_date=historical_data_start,
            end_date=today,
            indices_dict=filtered_indices_dict,
            market_proxy="VWCE.DE",
            ff_universe="developed"
        )

    # 1. Hyperparameter Search Space
    # 1. Hyperparameter Search Space (Reduced to prevent overfitting)
    top_candidates_opt = trial.suggest_int("top_candidates", 3, 6)
    holding_zone_opt = trial.suggest_int("holding_zone", 60, 85, step=5)
    
    # Define the core robust factors that actually deserve optimization
    optimizable_keys = [
        "EV_UFCF",           # Core unlevered cash flow valuation
        "EV_EBITDA_CAPEX",   # Capital intensity adjusted valuation
        "ROA",               # Core profitability
        "Alpha_6F",          # Risk-adjusted factor outperformance
        "Momentum_12M_1M"    # Trend and market inertia
    ]

    custom_weights_config = {}
    for key in DEFAULT_WEIGHTS_CONFIG.keys():
        if key in optimizable_keys:
            # Optimize only the core factors
            custom_weights_config[key] = trial.suggest_float(f"w_{key}", 0.0, 0.4, step=0.05)
        else:
            # Fix noisy, redundant, or strictly static parameters to 0.0
            custom_weights_config[key] = 0.0
    
    # Ensure standard static weights remain 0.0
    custom_weights_config['PER'] = 0.0
    custom_weights_config['Price_Book'] = 0.0
    
    # Fix Health Parameters (Do not optimize these, act as strict risk filters)
    custom_health_config = DEFAULT_HEALTH_CONFIG

    try:
        # Pass the preloaded RAM instances to bypass disk I/O completely
        mean_monthly_net_roi, total_trades_executed = run_historical_backtest(
            indices_dict=target_indices,
            months_to_simulate=12,
            monthly_contribution=1000.0,
            top_candidates=top_candidates_opt,
            holding_zone=holding_zone_opt,
            custom_weights=custom_weights_config,
            health_params=custom_health_config,
            export_excel=False, # Critical for optimization speed
            not_available_tickers=NOT_AVAIABLE_TICKERS,
            preloaded_analyzer=WORKER_ANALYZER, 
            preloaded_mapping=WORKER_MAPPING
        )
        
        loss = -mean_monthly_net_roi 
        
        sys.stdout.write(f"[TRIAL {trial.number}] Net ROI: {mean_monthly_net_roi:7.2f}% | Trades: {total_trades_executed:2d} | Loss: {loss:7.4f}\n")
        sys.stdout.flush()
        
        return loss

    except Exception as e:
        sys.stdout.write(f"[WARNING] Trial {trial.number} failed: {e}\n")
        sys.stdout.flush()
        return 999.0
    
if __name__ == "__main__":
    print("[INFO] Preparing architecture for max CPU utilization...")
    
    target_indices = {
        "S&P 500": get_index_tickers("SP500"),
        "EUROSTOXX 600": get_index_tickers("EUROSTOXX600"),
        "IBEX 35": get_index_tickers("IBEX35"),
        "PROTECTED_ASSETS": PROTECTED_TICKERS
    }

    # Force the main process to download and create the .pkl cache ONCE sequentially.
    # This ensures the parallel workers only have to read it.
    print("[INFO] Verifying/Building master cache...")
    mapped_indices = load_or_update_ticker_mapping(target_indices)
    filtered_indices = {k: [t for t in v if t not in NOT_AVAIABLE_TICKERS] for k, v in mapped_indices.items()}
    
    today = datetime.now()
    start_simulation = today - relativedelta(months=12)
    hist_start = start_simulation - relativedelta(years=FUNDAMENTAL_WINDOW_YEARS)
    
    _ = MarketAnalyzer(
        start_date=hist_start,
        end_date=today,
        indices_dict=filtered_indices,
        market_proxy="VWCE.DE",
        ff_universe="developed"
    )

    print("[INFO] Launching Optuna TPE Optimizer (In-Memory, Max Cores)...")
    
    # Pure RAM study, no SQLite locks
    study = optuna.create_study(
        study_name="Portfolio_Hyperparameter_Optimization", 
        direction="minimize",
        sampler=optuna.samplers.TPESampler()
    )
    
    # Blast the CPU!
    study.optimize(
        lambda trial: objective(trial, target_indices), 
        n_trials=500, 
        n_jobs=1
    )

    print("\n" + "="*50)
    print(" OPTIMIZATION COMPLETE ")
    print("="*50)
    
    best_params = study.best_params
    best_loss = study.best_value
    
    print(f"Best Loss Value: {best_loss:.4f}")
    print("Best Hyperparameters Discovered:")
    for key, value in best_params.items():
        print(f"  - {key}: {value}")

    fig = vis.plot_optimization_history(study)
    fig.show()

    fig2 = vis.plot_param_importances(study)
    fig2.show()

    today_str = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join("WeightOptimizations", today_str)
    os.makedirs(output_dir, exist_ok=True)

    fig.write_html(os.path.join(output_dir, "optimization_history.html"))
    fig2.write_html(os.path.join(output_dir, "param_importances.html"))

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
        
    print(f"\n[INFO] Optimization successfully saved at: {output_dir}")