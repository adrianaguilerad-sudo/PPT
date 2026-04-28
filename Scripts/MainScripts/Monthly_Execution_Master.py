import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Libraries'))
import subprocess
import optuna
import numpy as np
import json
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
import optuna.visualization as vis
from InvestingStrategy_lib import get_monthly_recommendations

# CRITICAL: Disable internal multithreading for math libraries BEFORE importing them.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Adjust imports according to your folder structure
from InvestingStrategy_lib import (
    run_historical_backtest, PROTECTED_TICKERS, NOT_AVAIABLE_TICKERS, 
    load_or_update_ticker_mapping, FUNDAMENTAL_WINDOW_YEARS
)
from MarketSelection_lib import get_index_tickers, MarketAnalyzer
from config import DEFAULT_WEIGHTS_CONFIG, DEFAULT_HEALTH_CONFIG

# Suppress Optuna spam
optuna.logging.get_logger("optuna").setLevel(logging.WARNING)

def create_objective(target_indices, preloaded_analyzer, preloaded_mapping):
    """
    Closure to generate the Optuna objective function with preloaded RAM objects.
    """
    def objective(trial):
        # Reduced and robust search space
        top_candidates_opt = trial.suggest_int("top_candidates", 3, 6)
        holding_zone_opt = trial.suggest_int("holding_zone", 60, 85, step=5)
        
        optimizable_keys = [
            "EV_UFCF", 
            "EV_EBITDA_CAPEX", 
            "ROA", 
            "Alpha_6F", 
            "Momentum_12M_1M"
        ]
        
        custom_weights_config = {}
        for key in DEFAULT_WEIGHTS_CONFIG.keys():
            if key in optimizable_keys:
                custom_weights_config[key] = trial.suggest_float(f"w_{key}", 0.0, 0.4, step=0.05)
            else:
                custom_weights_config[key] = 0.0
        
        custom_weights_config['PER'] = 0.0
        custom_weights_config['Price_Book'] = 0.0
        
        # Strict health filters (Non-optimizable)
        custom_health_config = DEFAULT_HEALTH_CONFIG
        try:
            mean_monthly_net_roi, _ = run_historical_backtest(
                indices_dict=target_indices,
                months_to_simulate=12, 
                monthly_contribution=1000.0,
                top_candidates=top_candidates_opt,
                holding_zone=holding_zone_opt,
                custom_weights=custom_weights_config,
                health_params=custom_health_config,
                export_excel=False, 
                not_available_tickers=NOT_AVAIABLE_TICKERS,
                preloaded_analyzer=preloaded_analyzer, 
                preloaded_mapping=preloaded_mapping
            )
            return -mean_monthly_net_roi 
        except Exception as e:
            return 999.0 
    return objective

if __name__ == "__main__":
    print("==================================================")
    print(" MONTHLY PORTFOLIO EXECUTION AND ENSEMBLE PIPELINE ")
    print("==================================================\n")
    
    # ---------------------------------------------------------
    # 0. INITIALIZATION & CACHING
    # ---------------------------------------------------------
    print("[INFO] Preloading market data and mappings into RAM...")
    target_indices = {
        "S&P 500": get_index_tickers("SP500"),
        "EUROSTOXX 600": get_index_tickers("EUROSTOXX600"),
        "IBEX 35": get_index_tickers("IBEX35"),
        "PROTECTED_ASSETS": PROTECTED_TICKERS
    }

    mapped_indices = load_or_update_ticker_mapping(target_indices)
    filtered_indices = {k: [t for t in v if t not in NOT_AVAIABLE_TICKERS] for k, v in mapped_indices.items()}
    
    today = datetime.now()
    start_simulation = today - relativedelta(months=12)
    hist_start = start_simulation - relativedelta(years=FUNDAMENTAL_WINDOW_YEARS)
    
    master_analyzer = MarketAnalyzer(
        start_date=hist_start,
        end_date=today,
        indices_dict=filtered_indices,
        market_proxy="VWCE.DE",
        ff_universe="developed"
    )

    # ---------------------------------------------------------
    # 1. BASELINE BACKTEST (DEFAULT PARAMS)
    # ---------------------------------------------------------
    print("\n[STEP 1] Executing Baseline Backtest (Default Config)...")
    baseline_roi, baseline_trades = run_historical_backtest(
        indices_dict=target_indices,
        months_to_simulate=12,
        monthly_contribution=1000.0,
        top_candidates=DEFAULT_WEIGHTS_CONFIG.get("top_candidates", 5),
        holding_zone=DEFAULT_WEIGHTS_CONFIG.get("holding_zone", 75),
        custom_weights=DEFAULT_WEIGHTS_CONFIG,
        health_params=DEFAULT_HEALTH_CONFIG,
        export_excel=True, 
        not_available_tickers=NOT_AVAIABLE_TICKERS,
        preloaded_analyzer=master_analyzer, 
        preloaded_mapping=mapped_indices
    )
    print(f"  -> Baseline Net ROI: {baseline_roi:.2f}% | Trades: {baseline_trades}")

    # ---------------------------------------------------------
    # 2. CALIBRATION & ENSEMBLING (3 MODELS)
    # ---------------------------------------------------------
    print("\n[STEP 2] Launching Ensemble Calibration (Independent Models)...")
    n_models = 5
    n_trials_per_model = 500
    all_best_params = []

    objective_func = create_objective(target_indices, master_analyzer, mapped_indices)
    output_dir = os.path.join("Results", "WeightOptimizations", today.strftime('%Y-%m-%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_models):
        print(f"  -> Training Model {i+1}/{n_models}...")
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective_func, n_trials=n_trials_per_model, n_jobs=1)
        
        all_best_params.append(study.best_params)
        print(f"     Model {i+1} Best ROI: {-study.best_value:.2f}%")
        # Generate and save Optuna charts for each specific model
        fig_hist = vis.plot_optimization_history(study)
        fig_imp = vis.plot_param_importances(study)

        fig_hist.write_html(os.path.join(output_dir, f"optimization_history_model_{i+1}.html"))
        fig_imp.write_html(os.path.join(output_dir, f"param_importances_model_{i+1}.html"))

    print("\n[INFO] Averaging parameters across models...")
    ensembled_params = {}
    
    for key in all_best_params[0].keys():
        values = [p[key] for p in all_best_params]
        if "top_candidates" in key or "holding_zone" in key:
            ensembled_params[key] = int(round(np.mean(values)))
        else:
            ensembled_params[key] = round(np.mean(values), 4)
            
    print("  -> Final Ensembled Parameters:")
    for k, v in ensembled_params.items():
        print(f"     {k}: {v}")

    # Reconstruct final dictionaries for the backtest
    final_custom_weights = {}
    for key in DEFAULT_WEIGHTS_CONFIG.keys():
        if f"w_{key}" in ensembled_params:
            final_custom_weights[key] = ensembled_params[f"w_{key}"]
        else:
            final_custom_weights[key] = 0.0
            
    final_health_config = DEFAULT_HEALTH_CONFIG 

    # ---------------------------------------------------------
    # 3. OPTIMIZED BACKTEST (ENSEMBLE PARAMS)
    # ---------------------------------------------------------
    print("\n[STEP 3] Executing Optimized Backtest (Ensemble Config)...")
    optimized_roi, optimized_trades = run_historical_backtest(
        indices_dict=target_indices,
        months_to_simulate=12,
        monthly_contribution=1000.0,
        top_candidates=ensembled_params["top_candidates"],
        holding_zone=ensembled_params["holding_zone"],
        custom_weights=final_custom_weights,
        health_params=final_health_config,
        export_excel=True, 
        not_available_tickers=NOT_AVAIABLE_TICKERS,
        preloaded_analyzer=master_analyzer, 
        preloaded_mapping=mapped_indices
    )
    print(f"  -> Optimized Net ROI: {optimized_roi:.2f}% | Trades: {optimized_trades}")

    # ---------------------------------------------------------
    # 4. SAVE ENSEMBLE PARAMS FOR PRODUCTION
    # ---------------------------------------------------------
    
    # Save the averaged params so FullMonthlyReport can read them
    # Make sure to format them as the JSON loader expects
    json_output = ensembled_params.copy()

        
    # Extraer y guardar las distribuciones de cada parámetro en todos los modelos
    for key in all_best_params[0].keys():
        json_output[f"{key}_distribution"] = [p[key] for p in all_best_params]


    json_output["h_roa_min"] = final_health_config['roa_min']
    json_output["h_ebit_min"] = final_health_config['ebit_min']
    json_output["h_int_cov_min"] = final_health_config['interest_coverage_min']
    json_output["h_pass_fin"] = final_health_config['pass_threshold_financial']
    json_output["h_pass_std"] = final_health_config['pass_threshold_standard']
    
    json_path = os.path.join(output_dir, "best_params.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=4)
        
    print(f"\n[INFO] Ensembled parameters saved at: {json_path}")

# ---------------------------------------------------------
    # 5. EXECUTE FULL MONTHLY REPORT (INTEGRATED)
    # ---------------------------------------------------------
    print("\n[STEP 4] Executing Full Monthly Market Analysis and Orders...")
    
    # MANUALLY UPDATE THIS DICTIONARY EVERY MONTH
    # Format -> 'Ticker': Quantity_Owned
    CURRENT_HOLDINGS = {
        'IDR.MC': 450,
        'EGLN.L': 426,
        'BTC-EUR': 812,
        'JNJ.DE': 244,
    }

    try:
        # Execute recommendations using the dynamically ensembled parameters
        get_monthly_recommendations(
            indices_dict=target_indices,
            current_holdings=CURRENT_HOLDINGS,
            monthly_contribution=1000.0,  # Adjust if necessary or import from config
            top_candidates=ensembled_params["top_candidates"],
            holding_zone=ensembled_params["holding_zone"],
            custom_weights=final_custom_weights,
            health_params=final_health_config,
            not_available_tickers=NOT_AVAIABLE_TICKERS
        )
        print("\n[INFO] Pipeline completed successfully. Review your action reports.")
    except Exception as e:
        print(f"\n[ERROR] Integrated monthly recommendation execution failed: {e}")