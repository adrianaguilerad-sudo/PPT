# Standard library imports
from dateutil.relativedelta import relativedelta
import pandas as pd
from FinanceFunctions_lib import get_james_stein_expected_returns, get_ledoit_wolf_covariance, get_james_stein_jorion_expected_returns
# Custom project imports
from MarketSelection_lib import load_market_data
import numpy as np
from datetime import datetime
import json
import os

# Import custom libraries
from MarketSelection_lib import get_index_tickers, MarketAnalyzer
from MarkowitzPortfolio_lib import Portfolio, initialize_portfolio
from FinanceFunctions_lib import get_european_ticker, get_prices_at_date

# InvestingStrategy_lib.py (Top of the file)
import config  # Import centralized config
import pickle
import uuid
# Global RAM cache to prevent disk I/O bottlenecks during Optuna optimization
GLOBAL_MERGED_DF_CACHE = {}


# Use constants from config instead of local definitions
FUNDAMENTAL_WINDOW_YEARS = config.FUNDAMENTAL_WINDOW_YEARS
MARKOWITZ_WINDOW_YEARS = config.MARKOWITZ_WINDOW_YEARS
NOT_AVAIABLE_TICKERS = config.NOT_AVAILABLE_TICKERS 
PROTECTED_TICKERS = config.PROTECTED_TICKERS

# No change needed in the functions themselves as they already use these 
# global variables or receive them as parameters.

def load_or_update_ticker_mapping(indices_dict, mapping_file=os.path.join("CacheData", "Ticker_Mapping_Cache.json")):
    """
    Loads existing ticker translations from a local JSON file. 
    If a new ticker is found, it fetches the EUR equivalent via API and updates the JSON.
    """
    print(f"[INFO] Loading or updating ticker mapping cache from '{mapping_file}'...")
    
    mapping_cache = {}
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                mapping_cache = json.load(f)
            print(f"[INFO] Successfully loaded {len(mapping_cache)} cached mappings.")
        except Exception as e:
            print(f"[WARNING] Failed to read mapping cache ({e}). Building a new one.")
            
    mapped_indices_dict = {}
    cache_updated = False
    
    for index_name, ticker_list in indices_dict.items():
        print(f"[INFO] Processing mappings for index: {index_name}")
        mapped_list = []
        
        for t in ticker_list:
            # If the ticker already has a suffix, assume it's mapped
            if '.' in t:
                mapped_ticker = t
            else:
                # Check if the raw ticker exists in our JSON dictionary
                if t in mapping_cache:
                    mapped_ticker = mapping_cache[t]
                else:
                    # New ticker detected: Fetch from API and store it
                    print(f"[INFO] New ticker detected: '{t}'. Fetching EUR equivalent...")
                    mapped_ticker = get_european_ticker(t)
                    mapping_cache[t] = mapped_ticker
                    cache_updated = True
            
            # Prevent duplicates in the final index list
            if mapped_ticker not in mapped_list:
                mapped_list.append(mapped_ticker)
                
        mapped_indices_dict[index_name] = mapped_list
        
    # If we added new tickers, overwrite the JSON file to save progress
    if cache_updated:
        try:
            with open(mapping_file, 'w') as f:
                json.dump(mapping_cache, f, indent=4)
            print(f"[INFO] Ticker mapping cache was updated and saved to '{mapping_file}'.")
        except Exception as e:
            print(f"[ERROR] Could not save updated mapping cache: {e}")
            
    return mapped_indices_dict

def apply_heston_sde_filter(mapped_candidates, transaction_date, sim_prices_df, fallback_tags, max_approved=5,years_lookback=2):
    """
    Applies Heston Stochastic Differential Equation filtering to a list of candidate tickers.
    Validates the Feller condition and a maximum volatility threshold.
    """
    
    from SDE_lib import HestonModelWindowSigma
    
    approved_equities = []
    lookback_sde = transaction_date - relativedelta(years=years_lookback)
    
    for ticker in mapped_candidates:
        model = HestonModelWindowSigma(ticker)
        print(f"Processing Heston filter for {ticker}")
        
        try:
            # Inject pre-downloaded data to save API calls, slicing to the exact timeframe
            if ticker not in sim_prices_df.columns:
                eur_data = load_market_data([ticker], start_date=lookback_sde, end_date=transaction_date)
                if eur_data.empty or ticker not in eur_data.columns:
                    continue
                model.prices = eur_data[ticker].dropna().values
            else:
                model.prices = sim_prices_df.loc[lookback_sde:transaction_date, ticker].dropna().values
                
            # Calibrate the model if we have enough data points
            if len(model.prices) > 252:
                model.calibrate(verbose=False)
                feller_val = 2 * model.kappa * model.theta_v
                
                # Verify Feller condition and acceptable volatility
                if feller_val > model.xi**2 and model.theta_v < 0.15:
                    approved_equities.append(ticker)
            
            # Stop processing once we reach the required number of assets
            if len(approved_equities) >= max_approved:
                break
                
        except Exception as e:
            print(f"[ERROR] Heston filter failed for {ticker}: {e}")
            continue
            
    # Apply fallback logic if no assets pass the strict mathematical criteria
    if not approved_equities:
        raise ValueError("[WARNING] No equities passed SDE filter. Carrying over previous portfolio.")

    else:
        print(f"SDE Approved tickers: {approved_equities}")
        
    return approved_equities


def build_monthly_records(
    my_portfolio,
    adjustments,
    current_prices,
    liquidation_records,
    updated_fiat_invested,
    current_portfolio_value,
    monthly_contribution,
    roi_percentage,
    top_candidates_df=None # New parameter added to handle candidates
):
    """
    Builds the tracking records for the current month to be exported to Excel.
    Combines liquidations, active asset statuses, and portfolio summaries.
    Optionally returns the top candidates dataframe for separate exporting.
    """
    month_records = []
    
    # Add liquidation records first
    month_records.extend(liquidation_records)

    # Add active portfolio allocations
    for i, tag in enumerate(my_portfolio.tags):
        weight = my_portfolio.weights[i]
        action_amount = adjustments.get(tag, 0.0)
        price_used = current_prices.get(tag, 0.0)
        
        shares_owned = my_portfolio.investments[tag].get('total_shares', 0.0)
        market_val = shares_owned * price_used
        
        month_records.append({
            'Ticker': tag,
            'Target_Weight': f"{weight:.2%}",
            'Price': price_used,
            'Shares_Owned': shares_owned,
            'Current_Market_Value': market_val,
            'Recommended_Action': action_amount,
            'Action_Taken': action_amount 
        })
        
    # Append summary rows
    month_records.append({'Ticker': 'TOTAL INVESTED', 'Current_Market_Value': updated_fiat_invested})
    month_records.append({'Ticker': 'TOTAL VALUE', 'Current_Market_Value': current_portfolio_value + monthly_contribution})
    month_records.append({'Ticker': 'ROI (%)', 'Current_Market_Value': roi_percentage})
    
    # Return both the main records and the passed candidates dataframe
    return month_records, top_candidates_df


def ensure_all_prices_available(final_universe_check, current_prices, transaction_date, transaction_date_str):
    """
    Checks if all required tickers are present in the current prices dictionary.
    If missing, it fetches the most recent price within a 10-day lookback window.
    """
    # Identify which tags are missing beforehand
    missing_tags = [tag for tag in final_universe_check if tag not in current_prices]
    
    if not missing_tags:
        print("[INFO] No new tickers to fetch")
        return current_prices

    for tag in missing_tags:
        print(f"[INFO] Fetching missing historical price for translated ticker: {tag}")
        try:
            # Look back 10 days to ensure we catch the last active trading day before transaction_date
            temp_start = transaction_date - relativedelta(days=10)
            temp_data = load_market_data([tag], start_date=temp_start, end_date=transaction_date)
            
            if not temp_data.empty and tag in temp_data.columns:
                current_prices[tag] = temp_data[tag].dropna().iloc[-1]
            else:
                print(f"[WARNING] yfinance returned empty data for {tag} at {transaction_date_str}.")
        except Exception as e:
            print(f"[ERROR] Could not fetch missing price for {tag}: {e}")
            
    return current_prices


def full_asset_rotation_strategy(
    my_portfolio, 
    approved_equities, 
    current_prices, 
    transaction_date_str, 
    monthly_contribution, 
    transaction_date,
    years_lookback=1
):
    """
    Executes the asset rotation, updates the portfolio universe, computes shrinkaged 
    expected returns and covariance matrices, optimizes the portfolio weights,
    and calculates the required monetary adjustments.
    """
    # 1. Update universe and liquidate unapproved assets using the class method
    extra_cash, liquidation_records = my_portfolio.update_universe_full_asset_rotation(
        new_assets=approved_equities, 
        current_prices=current_prices, 
        transaction_date=transaction_date_str
    )
    
    # 2. Calculate effective contribution
    effective_contribution = monthly_contribution + extra_cash
    print(f"Base Contribution: ${monthly_contribution:.2f} | Cash from Liquidations: ${extra_cash:.2f} | Total to Deploy: ${effective_contribution:.2f}")

    # 3. Calculate shrinkage metrics using a 1-year lookback
    lookback_opt = transaction_date - relativedelta(years=years_lookback)
    
    expected_returns = get_james_stein_jorion_expected_returns(my_portfolio.tags, lookback_opt, transaction_date, plotbool=False)
    cov_matrix = get_ledoit_wolf_covariance(my_portfolio.tags, lookback_opt, transaction_date, plotbool=False)
    
    # 4. Safely align expected returns using the class method
    my_portfolio.update_expected_returns(expected_returns)
    
    # 5. Apply covariance matrix, optimize, and calculate adjustments
    optimization_success = True
    adjustments = {}
    
    if len(cov_matrix) > 0:
        my_portfolio.set_covariance_matrix(cov_matrix)
        my_portfolio.optimize_maximize_sharpe_ratio(risk_free_rate=0.02)
        
        # 6. Calculate the specific adjustments needed with the new optimized weights
        adjustments = my_portfolio.calculate_rebalance_contribution(effective_contribution, current_prices)
    else:
        print("[ERROR] Covariance matrix failed. Aborting optimization.")
        optimization_success = False
        
    return my_portfolio, effective_contribution, liquidation_records, optimization_success, adjustments


def fundamental_floor_strategy(
    my_portfolio, 
    approved_equities, 
    healthy_universe,
    ranked_df,
    current_prices, 
    transaction_date_str, 
    monthly_contribution, 
    transaction_date,
    years_lookback=1,
    percentile_exit_threshold=None,
    sim_prices_df=None  # <--- NUEVO PARÁMETRO
):
    # 1. Liquidate only those that have failed fundamentally
    extra_cash, liquidation_records = my_portfolio.update_universe_fundamental_floor(
        new_candidates=approved_equities,
        healthy_universe=healthy_universe,
        ranked_df=ranked_df,
        current_prices=current_prices,
        transaction_date=transaction_date_str,
        percentile_exit_threshold=percentile_exit_threshold
    )
    
    effective_contribution = monthly_contribution + extra_cash

    # 2. Configure metrics (James-Stein / Ledoit-Wolf) using Pre-loaded RAM Data
    lookback_opt = transaction_date - relativedelta(years=years_lookback)
    
    # Inyectamos el sim_prices_df aquí:
    expected_returns = get_james_stein_jorion_expected_returns(
        my_portfolio.tags, lookback_opt, transaction_date, plotbool=False, prices_df=sim_prices_df
    )
    cov_matrix = get_ledoit_wolf_covariance(
        my_portfolio.tags, lookback_opt, transaction_date, plotbool=False, prices_df=sim_prices_df
    )
    
    my_portfolio.update_expected_returns(expected_returns)
    my_portfolio.set_covariance_matrix(cov_matrix)

    # 3. NO-SELL Optimization
    print(">>> Executing Sharpe Optimization with No-Sell constraint...")
    my_portfolio.optimize_sharpe_no_sell_bounded(
        new_contribution=effective_contribution, 
        current_prices=current_prices, 
        risk_free_rate=0.02
    )
    
    # 4. Calculate final adjustments
    adjustments = my_portfolio.calculate_rebalance_contribution(effective_contribution, current_prices)
    
    return my_portfolio, effective_contribution, liquidation_records, True, adjustments

def calculate_net_roi_with_taxes(gross_portfolio_value, total_trades, total_realized_gains):
    """
    Calculates the true net value of the portfolio after applying 
    Trade Republic fees (1 EUR/trade) and Spanish capital gains taxes.
    """
    broker_fee_per_trade = 1.0 
    total_broker_fees = total_trades * broker_fee_per_trade

    tax_paid = 0.0
    if total_realized_gains > 0:
        if total_realized_gains <= 6000:
            tax_paid = total_realized_gains * 0.19
        elif total_realized_gains <= 50000:
            tax_paid = (6000 * 0.19) + ((total_realized_gains - 6000) * 0.21)
        else:
            tax_paid = (6000 * 0.19) + (44000 * 0.21) + ((total_realized_gains - 50000) * 0.23)
            
    net_value = gross_portfolio_value - total_broker_fees - tax_paid
    return net_value


def save_execution_outputs(my_portfolio, global_ranked_df, current_prices, adjustments):
    """
    Creates a timestamped directory and saves the current portfolio state 
    (including monthly adjustments), the global ranked candidates, 
    and a JSON snapshot of the portfolio object.
    """
    today_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    root_folder = os.path.join("Results", "MyInvestments")
    folder_path = os.path.join(root_folder, today_str)
    
    # 1. Create directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"[INFO] Created directory: {folder_path}")

    # 2. Build and save Portfolio State to Excel
    portfolio_state = []
    for i, tag in enumerate(my_portfolio.tags):
        shares = my_portfolio.investments[tag].get('total_shares', 0.0)
        price = current_prices.get(tag, 0.0)
        
        # Safely handle weights in case optimization failed or weights are not set
        weight = 0.0
        if hasattr(my_portfolio, 'weights') and my_portfolio.weights is not None and len(my_portfolio.weights) > i:
            weight = my_portfolio.weights[i]
            
        # Get the adjustment made for this specific ticker (default to 0.0)
        action_amount = adjustments.get(tag, 0.0)
            
        portfolio_state.append({
            'Ticker': tag,
            'Shares': shares,
            'Price': price,
            'Current_Market_Value': shares * price,
            'Target_Weight': weight,
            'Adjustment_EUR': action_amount  # <--- NUEVA COLUMNA AQUÍ
        })
        
    portfolio_df = pd.DataFrame(portfolio_state)
    portfolio_filename = os.path.join(folder_path, f"Portfolio_State_{today_str}.xlsx")
    print(f"[INFO] Saving portfolio state to {portfolio_filename}...")
    portfolio_df.to_excel(portfolio_filename, index=False)

    # 3. Save Global Ranked DataFrame to Excel
    ranked_filename = os.path.join(folder_path, f"Global_Ranked_{today_str}.xlsx")
    print(f"[INFO] Saving global ranked candidates to {ranked_filename}...")
    global_ranked_df.to_excel(ranked_filename, index=False)

    # 4. Save Portfolio Snapshot to JSON
    # Convert numpy arrays to lists for JSON serialization
    portfolio_snapshot = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'assets': my_portfolio.tags,
        'weights': my_portfolio.weights.tolist() if hasattr(my_portfolio.weights, 'tolist') else my_portfolio.weights,
        'detailed_investments': my_portfolio.investments,
        'current_prices_at_execution': {k: float(v) for k, v in current_prices.items() if k in my_portfolio.tags}
    }
    
    snapshot_path = os.path.join(folder_path, f"Portfolio_Snapshot_{today_str}.json")
    with open(snapshot_path, 'w') as f:
        json.dump(portfolio_snapshot, f, indent=4)
        
    print(f"[INFO] Portfolio snapshot saved to {snapshot_path}")
    return folder_path

def get_monthly_recommendations(indices_dict,current_holdings, monthly_contribution=1000.0, 
                                top_candidates=5, holding_zone=80, 
                                custom_weights=None, health_params=None,
                                not_available_tickers=None):
    """
    Executes the investment strategy for the current date to generate 
    final trade recommendations, excluding any tickers flagged as not available.
    """
    print("==========================================")
    print(" STARTING REAL-TIME STRATEGY EXECUTION")
    print(f" DATE: {datetime.now().strftime('%Y-%m-%d')}")
    print("==========================================\n")

    # Fallback to global variable if no list is explicitly provided
    if not_available_tickers is None:
        try:
            not_available_tickers = NOT_AVAIABLE_TICKERS
        except NameError:
            not_available_tickers = []

    today = datetime.now()
    historical_data_start = today - relativedelta(years=FUNDAMENTAL_WINDOW_YEARS)

    # 1. Ticker Mapping and Filtering
    mapped_indices_dict = load_or_update_ticker_mapping(indices_dict)
    
    # --- NEW LOGIC: Filter out not available tickers ---
    print(f"[INFO] Filtering out {len(not_available_tickers)} not available tickers from the analysis...")
    filtered_indices_dict = {}
    for index_name, tickers in mapped_indices_dict.items():
        filtered_tickers = []
        for t in tickers:
            if t in not_available_tickers:
                print(f"[INFO] Ticker {t} is marked as not available and will be excluded.")
            else:
                filtered_tickers.append(t)
        filtered_indices_dict[index_name] = filtered_tickers

    filtered_indices_dict = filtered_indices_dict
    # 2. Initialize Market Analyzer using the FILTERED dictionary
    analyzer = MarketAnalyzer(
        start_date=historical_data_start,
        end_date=today,
        indices_dict=filtered_indices_dict, # <--- SE USA EL DICCIONARIO FILTRADO
        market_proxy="VWCE.DE",
        ff_universe="developed" 
    )

    # 3. Load Current Portfolio State
    my_portfolio = Portfolio()
    my_portfolio.Non_sellable_tickers = PROTECTED_TICKERS
    # 4. Market Ranking for Today
    print("\n>>> Phase 1: Ranking Universe...")
    lookback_prices_date = today - relativedelta(years=FUNDAMENTAL_WINDOW_YEARS)
    
    sim_prices_df = analyzer.prices_db.loc[lookback_prices_date:today]
    current_prices = get_prices_at_date(sim_prices_df, today)
    
    print("\n>>> Phase 2: Loading Current Holdings and converting to shares...")
    for ticker, eur_value in current_holdings.items():
        my_portfolio.add_financial_asset(ticker, 0.0) # Init tracking
        
        # Retrieve the price for conversion
        price = current_prices.get(ticker, 0.0)
        
        if price > 0:
            # Convert EUR value to total shares
            calculated_shares = eur_value / price
            my_portfolio.investments[ticker]['total_shares'] = calculated_shares
            print(f"[INFO] Loaded {ticker}: ${eur_value:.2f} | Price: ${price:.2f} | Shares: {calculated_shares:.6f}")
        else:
            print(f"[WARNING] Skipping {ticker}: Current price not found in database.")

    raw_market = analyzer.get_market_performance_coefficients(lookback_prices_date, today)

    multipliers_df = analyzer.get_multipliers(
        target_date=today, 
        current_prices=current_prices, 
        lookback_years=FUNDAMENTAL_WINDOW_YEARS,
        health_params=health_params
    )

    healthy_tickers = multipliers_df['Ticker'].tolist() if multipliers_df is not None and not multipliers_df.empty else []
    
    financial_health_df = analyzer.get_financials(target_date=today, lookback_years=FUNDAMENTAL_WINDOW_YEARS)
    
    if raw_market is None or multipliers_df is None or financial_health_df is None:
        raise ValueError("[ERROR] Strategy failed due to data extraction issues.")

    merged_df = pd.merge(raw_market, multipliers_df, on='Ticker')
    merged_df = pd.merge(merged_df, financial_health_df, on=['Ticker', 'Sector'])
    
    global_ranked_df = analyzer.rank_and_score(merged_df, custom_weights=custom_weights)
    top_candidates_df = global_ranked_df.head(top_candidates)
    
    approved_equities = top_candidates_df['Ticker'].tolist()
    final_universe_check = list(dict.fromkeys(my_portfolio.tags + approved_equities +PROTECTED_TICKERS))
    
    current_prices = ensure_all_prices_available(
        final_universe_check=final_universe_check,
        current_prices=current_prices,
        transaction_date=today,
        transaction_date_str=today.strftime('%Y-%m-%d')
    )

    # 5. Optimization & Rotation Strategy
    print(">>> Phase 3: Shrinkage Optimization & Asset Rotation")
    my_portfolio, effective_contribution, liquidation_records, opt_success, adjustments = fundamental_floor_strategy(
                my_portfolio=my_portfolio,
                approved_equities=approved_equities,
                healthy_universe=healthy_tickers,
                ranked_df=global_ranked_df,
                current_prices=current_prices,
                transaction_date_str=today.strftime('%Y-%m-%d'),
                monthly_contribution=monthly_contribution,
                transaction_date=today,
                years_lookback=MARKOWITZ_WINDOW_YEARS,
                percentile_exit_threshold=holding_zone,
                sim_prices_df=sim_prices_df  # <--- INYECTADO AQUÍ
            )

    # --- PROCESS NEW ALLOCATIONS ---
    my_portfolio.operate_adjustments(
        adjustments=adjustments,
        current_prices=current_prices,
        transaction_date=today.strftime('%Y-%m-%d')
    )

    # 6. Final Trade Execution Output
    print(f"\n{'='*50}")
    print(" FINAL TRADE RECOMMENDATIONS")
    print(f"{'='*50}")
    
    print(f"\n[CASH FLOW] New Contribution: ${monthly_contribution:.2f}")
    if liquidation_records:
        print("\n[SELL ORDERS] Assets failing fundamental floor or ranking threshold:")
        for record in liquidation_records:
            print(f" - SELL {record['Ticker']}: Sell all shares at approx ${record['Price']:.2f}")
    else:
        print("\n[SELL ORDERS] None. All current assets remain healthy.")

    print("\n[BUY/ADJUSTMENT ORDERS] Target rebalancing:")
    for ticker, amount in adjustments.items():
        if amount > 1.0: # Ignore dust
            price = current_prices.get(ticker, 0.0)
            shares_to_buy = amount / price if price > 0 else 0
            print(f" - BUY {ticker}: Invest ${amount:.2f} (~{shares_to_buy:.4f} shares at ${price:.2f})")

    # --- EXPORTING DATA ---
    print(f"\n>>> Phase 4: Exporting execution data...")
    final_path = save_execution_outputs(my_portfolio, global_ranked_df, current_prices, adjustments)

    print(f"\n{'='*50}")
    print(" END OF EXECUTION")
    print(f"{'='*50}")
    print(f"[INFO] All files successfully saved in: {final_path}")



def run_historical_backtest(indices_dict, months_to_simulate=12, monthly_contribution=1000.0, 
                            top_candidates=5, holding_zone=90, 
                            custom_weights=None, health_params=None,
                            export_excel=True, not_available_tickers=None,
                            preloaded_analyzer=None, preloaded_mapping=None):
    """
    Unified historical backtest engine.
    Supports both standard backtesting (with Excel exports) and Optuna hyperparameter 
    optimization (using preloaded RAM objects to bypass disk I/O).
    Optimized to track FIFO tax lots, broker fees, and precise Net ROI.
    """
    print("==========================================")
    print(" STARTING TAX-AWARE SLIDING WINDOW BACKTEST")
    print("==========================================\n")

    if not_available_tickers is None:
        try:
            not_available_tickers = NOT_AVAIABLE_TICKERS
        except NameError:
            not_available_tickers = []

    today = datetime.now()
    start_simulation_date = today - relativedelta(months=months_to_simulate)
    historical_data_start = start_simulation_date - relativedelta(years=FUNDAMENTAL_WINDOW_YEARS)

    # Use preloaded mapping if provided (Optuna mode), else load from disk
    if preloaded_mapping is not None:
        mapped_indices_dict = preloaded_mapping
        print("[INFO] Using preloaded ticker mapping from RAM.")
    else:
        mapped_indices_dict = load_or_update_ticker_mapping(indices_dict)

    print("[INFO] Filtering out blocked tickers from the investment universe...")
    filtered_indices_dict = {}
    for index_name, tickers in mapped_indices_dict.items():
        filtered_tickers = []
        for t in tickers:
            if t in not_available_tickers:
                pass # Ticker is blocked
            else:
                filtered_tickers.append(t)
        filtered_indices_dict[index_name] = filtered_tickers

    # Use preloaded analyzer if provided (Optuna mode), else instantiate
    if preloaded_analyzer is not None:
        analyzer = preloaded_analyzer
        print("[INFO] Using preloaded MarketAnalyzer from RAM.")
    else:
        analyzer = MarketAnalyzer(
            start_date=historical_data_start,
            end_date=today,
            indices_dict=filtered_indices_dict, 
            market_proxy="VWCE.DE",
            ff_universe="developed" 
        )

    my_portfolio = Portfolio()
    total_fiat_invested = 0.0
    excel_sheets = {} 
    candidates_sheets = {}
    my_portfolio.Non_sellable_tickers = PROTECTED_TICKERS 
        
    for ticker in PROTECTED_TICKERS:
        if ticker not in my_portfolio.tags:
            my_portfolio.add_financial_asset(ticker, 0.0)
            print(f"[INFO] Initialized protected asset in tracking: {ticker}")
            
    # Fiscal and trade trackers
    tax_lots = {} 
    total_trades_executed = 0
    total_realized_gains = 0.0

    for month_offset in range(months_to_simulate, -1, -1):
        sim_date = today - relativedelta(months=month_offset)
        sim_date_str = sim_date.strftime('%Y-%m-%d')
        lookback_prices = sim_date - relativedelta(years=FUNDAMENTAL_WINDOW_YEARS)
        
        print(f"\n{'='*50}")
        print(f" SIMULATING MONTH: {sim_date_str}")
        print(f"{'='*50}")
        
        sim_prices_df = analyzer.prices_db.loc[lookback_prices:sim_date]
        current_prices = get_prices_at_date(sim_prices_df, sim_date)
        
        # ==========================================
        # CACHE LOGIC: RAM & DISK (CONCURRENT SAFE)
        # ==========================================
        cache_dir = os.path.join("CacheData", "CacheDataFrames")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"backtest_data_{sim_date_str}.pkl")

        # 1. Check if data is already in RAM (Fastest - hits on subsequent trials)
        if sim_date_str in GLOBAL_MERGED_DF_CACHE:
            print(f">>> Phase 1: Ranking Universe for {sim_date_str} (Loaded from RAM cache)...")
            cache_data = GLOBAL_MERGED_DF_CACHE[sim_date_str]
            merged_df = cache_data['merged_df'].copy()
            healthy_tickers = cache_data['healthy_tickers']
            
        else:
            # 2. Check if data is in Disk (Calculated by another CPU core or previous run)
            if os.path.exists(cache_file):
                print(f">>> Phase 1: Ranking Universe for {sim_date_str} (Loaded from Disk cache)...")
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    GLOBAL_MERGED_DF_CACHE[sim_date_str] = cache_data
                    merged_df = cache_data['merged_df'].copy()
                    healthy_tickers = cache_data['healthy_tickers']
                except (EOFError, pickle.UnpicklingError) as e:
                    print(f"[WARNING] Corrupted cache file for {sim_date_str}, recalculating... ({e})")
                    # Force recalculation if file is corrupted
                    merged_df = None 
            else:
                merged_df = None

            # 3. Calculate from scratch and cache it (Only runs if RAM and Disk missed)
            if merged_df is None:
                print(f">>> Phase 1: Ranking Universe for {sim_date_str} (Calculating & Caching)...")
                
                raw_market = analyzer.get_market_performance_coefficients(lookback_prices, sim_date)
                
                multipliers_df = analyzer.get_multipliers(
                    target_date=sim_date, 
                    current_prices=current_prices, 
                    lookback_years=FUNDAMENTAL_WINDOW_YEARS,
                    health_params=health_params
                )
                
                healthy_tickers = multipliers_df['Ticker'].tolist() if multipliers_df is not None and not multipliers_df.empty else []
                financial_health_df = analyzer.get_financials(target_date=sim_date, lookback_years=FUNDAMENTAL_WINDOW_YEARS)
                
                if raw_market is None or multipliers_df is None or financial_health_df is None:
                    raise ValueError("[WARNING] Stopping backtest due to data extraction failure.")
                    
                merged_df = pd.merge(raw_market, multipliers_df, on='Ticker')
                merged_df = pd.merge(merged_df, financial_health_df, on=['Ticker', 'Sector'])
                
                # Create cache payload
                cache_data = {
                    'merged_df': merged_df,
                    'healthy_tickers': healthy_tickers
                }
                
                # Save to RAM
                GLOBAL_MERGED_DF_CACHE[sim_date_str] = cache_data

                # Save to Disk ATOMICALLY to prevent multiprocessing corruption
                temp_file = f"{cache_file}.{uuid.uuid4().hex}.tmp"
                try:
                    with open(temp_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                    # os.replace is an atomic operation at the OS level. 
                    # If two cores finish at the same time, one just quietly overwrites the other cleanly.
                    os.replace(temp_file, cache_file) 
                except Exception as e:
                    print(f"[ERROR] Could not save disk cache for {sim_date_str}: {e}")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

        # Apply weights dynamically (This must always run to test new Optuna combinations)
        global_ranked_df = analyzer.rank_and_score(merged_df, custom_weights=custom_weights)
        top_candidates_df = global_ranked_df.head(top_candidates)
        
        approved_equities = top_candidates_df['Ticker'].tolist()
        final_universe_check = list(dict.fromkeys(my_portfolio.tags + approved_equities))
        
        current_prices = ensure_all_prices_available(
            final_universe_check=final_universe_check,
            current_prices=current_prices,
            transaction_date=sim_date,
            transaction_date_str=sim_date_str
        )

        print(">>> Phase 2: Evaluating Portfolio ROI...")
        pre_rotation_shares = {tag: my_portfolio.investments[tag].get('total_shares', 0.0) for tag in my_portfolio.tags}
        
        current_portfolio_value = 0.0
        if my_portfolio.tags:
            _, current_portfolio_value = my_portfolio.get_current_portfolio_state(current_prices)

        print(">>> Phase 3: Shrinkage Optimization & Asset Rotation")
        my_portfolio, effective_contribution, liquidation_records, opt_success, adjustments = fundamental_floor_strategy(
                    my_portfolio=my_portfolio,
                    approved_equities=approved_equities,
                    healthy_universe=healthy_tickers,
                    ranked_df=global_ranked_df,
                    current_prices=current_prices,
                    transaction_date_str=sim_date_str,
                    monthly_contribution=monthly_contribution,
                    transaction_date=sim_date,
                    years_lookback=MARKOWITZ_WINDOW_YEARS,
                    percentile_exit_threshold=holding_zone,
                    sim_prices_df=sim_prices_df  # <--- INYECTADO AQUÍ
                )
        
        if not opt_success:
            break
            
        # Process liquidations (FIFO Tax Lots)
        for tag, initial_shares in pre_rotation_shares.items():
            current_shares = my_portfolio.investments[tag].get('total_shares', 0.0) if tag in my_portfolio.tags else 0.0
            shares_sold = initial_shares - current_shares
            
            if shares_sold > 0.0001:
                total_trades_executed += 1
                sell_price = current_prices.get(tag, 0.0)
                shares_to_process = shares_sold
                
                if tag in tax_lots:
                    while shares_to_process > 0.0001 and tax_lots[tag]:
                        lot = tax_lots[tag][0]
                        if lot['shares'] <= shares_to_process:
                            profit = (sell_price - lot['price']) * lot['shares']
                            total_realized_gains += profit
                            shares_to_process -= lot['shares']
                            tax_lots[tag].pop(0)
                        else:
                            profit = (sell_price - lot['price']) * shares_to_process
                            total_realized_gains += profit
                            lot['shares'] -= shares_to_process
                            shares_to_process = 0
        
        # Process new allocations
        my_portfolio.operate_adjustments(
            adjustments=adjustments,
            current_prices=current_prices,
            transaction_date=sim_date_str
        )
        
        for tag, amount_invested in adjustments.items():
            if amount_invested > 1.0:
                total_trades_executed += 1
                buy_price = current_prices.get(tag, 0.0)
                if buy_price > 0:
                    shares_bought = amount_invested / buy_price
                    if tag not in tax_lots:
                        tax_lots[tag] = []
                    tax_lots[tag].append({'shares': shares_bought, 'price': buy_price})

        total_fiat_invested += monthly_contribution
        
        net_portfolio_value = calculate_net_roi_with_taxes(
            gross_portfolio_value=current_portfolio_value + monthly_contribution, 
            total_trades=total_trades_executed, 
            total_realized_gains=total_realized_gains
        )
        
        net_roi_percentage = 0.0
        if total_fiat_invested > 0:
            net_roi_percentage = ((net_portfolio_value / total_fiat_invested) - 1) * 100
            
        print(f"[FISCAL] Trades: {total_trades_executed} | Realized Gains: ${total_realized_gains:.2f}")
        print(f"[FISCAL] Net Value: ${net_portfolio_value:.2f} | Net ROI: {net_roi_percentage:.2f}%")

        print(f">>> Phase 4: Recording Rebalance (+${effective_contribution:.2f})...")    
            
        month_records, month_candidates = build_monthly_records(
            my_portfolio=my_portfolio,
            adjustments=adjustments,
            current_prices=current_prices,
            liquidation_records=liquidation_records,
            updated_fiat_invested=total_fiat_invested,
            current_portfolio_value=current_portfolio_value,
            monthly_contribution=monthly_contribution,
            roi_percentage=net_roi_percentage, 
            top_candidates_df=global_ranked_df
        )
        
        excel_sheets[sim_date_str] = pd.DataFrame(month_records)
        if month_candidates is not None:
            candidates_sheets[sim_date_str] = month_candidates
            
    # Excel exporting logic (Only triggers if export_excel=True)
    if export_excel: 
        output_filename = os.path.join("Results", "BackTests", f"Backtest_Results_{today.strftime('%Y%m%d_%H%M%S')}.xlsx")
        candidates_filename = os.path.join("Results", "BackTests", f"TopCandidates_Results_{today.strftime('%Y%m%d_%H%M%S')}.xlsx")

        os.makedirs(os.path.join("Results", "BackTests"), exist_ok=True)
        
        print(f"\n[INFO] Saving detailed report to {output_filename}...")
        with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
            for sheet_name, df in excel_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
        print(f"[INFO] Saving top candidates report to {candidates_filename}...")
        with pd.ExcelWriter(candidates_filename, engine='xlsxwriter') as writer:
            for sheet_name, df in candidates_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("--- BACKTEST COMPLETE ---")

    all_net_rois = []
    for df in excel_sheets.values():
        if not df.empty and 'Ticker' in df.columns:
            roi_row = df[df['Ticker'] == 'ROI (%)']
            if not roi_row.empty:
                all_net_rois.append(roi_row['Current_Market_Value'].iloc[0])
    
    mean_monthly_net_roi = np.mean(all_net_rois) if all_net_rois else 0.0
    
    return mean_monthly_net_roi, total_trades_executed