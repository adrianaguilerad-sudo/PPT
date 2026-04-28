import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MainScripts', 'Libraries'))
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
# Import the custom libraries
from MarkowitzPortfolio_lib import Portfolio, initialize_portfolio
from FinanceFunctions_lib import get_james_stein_expected_returns, get_ledoit_wolf_covariance, get_current_prices

def run_monthly_report():
    print("==========================================")
    print(" MONTHLY PORTFOLIO REBALANCE REPORT")
    print("==========================================\n")

    # ==========================================
    # 1. Define Timeline and Fetch Advanced Metrics
    # ==========================================

    new_monthly_contribution = 1000.00
    
    tickers = ['SXR8.DE', 'IDR.MC', 'EGLN.L', 'BTC-EUR','JNJ.F','OPC.DE']

    current_money = {
        'SXR8.DE': 0.0,
        'IDR.MC': 339.90,
        'EGLN.L': 282.37,
        'BTC-EUR': 181.24,
        'JNJ.F':0,
        'OPC.DE':0,
    }

    today = datetime.now()
    
    my_portfolio = initialize_portfolio(tickers, current_money)
    # We use a 1-year lookback to give the Ledoit-Wolf estimator enough data 
    # to build a stable covariance matrix, while the EWMA span handles recent trends.
    lookback_start = today - relativedelta(years=1)

    print(">>> STEP 1: Updating Market Metrics")
    expected_returns = get_james_stein_expected_returns(tickers, lookback_start, today, plotbool=True)
    cov_matrix = get_ledoit_wolf_covariance(tickers, lookback_start, today, plotbool=True)

    # ==========================================
    # 2. Load Portfolio & Update State
    # ==========================================
    
    print("\n>>> STEP 2: Loading Current Portfolio State")

    if "my_portfolio" not in locals() and "my_portfolio" not in globals():
        
        portfolio_filename = "Current_Diverse_Portfolio.pkl"
        
        if os.path.exists(portfolio_filename):
            print(f"Loading existing portfolio object from '{portfolio_filename}'...")
            with open(portfolio_filename, "rb") as file:
                my_portfolio = pickle.load(file)
        else:
            print(f"Error: '{portfolio_filename}' not found in the current folder.")
            print("Please ensure you have saved the Portfolio object before running the monthly report.")
            return
    else:
        print("Manual Initialization")


    print("Updating the loaded portfolio with this month's fresh market metrics...")
        
    # Overwrite the old expected returns with the new EWMA calculations
    # We iterate through the portfolio's tags to ensure the order stays perfectly aligned
    my_portfolio.expected_returns = [expected_returns[tag] for tag in my_portfolio.tags]
    
    # Overwrite the old covariance matrix with the new Ledoit-Wolf matrix
    my_portfolio.set_covariance_matrix(cov_matrix)
    
    # Print a quick summary of the loaded investments
    current_total = sum(my_portfolio.investments[tag]['total_invested'] for tag in my_portfolio.tags)
    print(f"Successfully loaded portfolio. Current tracked value: ${current_total:.2f}")


    # ==========================================
    # 3. Optimize for the New Target Weights
    # ==========================================
    print("\n>>> STEP 3: Recomputing Optimal Weights")
    # Using 3.63% (0.0363) as the risk-free rate proxy for the Tangency Portfolio
    risk_free_rate = 0.02 
    my_portfolio.optimize_maximize_sharpe_ratio(risk_free_rate=risk_free_rate)
    print("\n>>> New Expected Return")
    print(f"Return   -> Expected: {my_portfolio.get_portfolio_return():>7.2%}")

    print("\n--- New Target Weights (Maximum Sharpe Ratio) ---")
    for i, tag in enumerate(my_portfolio.tags):
        print(f"{tag}: {my_portfolio.weights[i]:.2%}")

    # ==========================================
    # 4. Calculate the Rebalance Contribution
    # ==========================================
    print("\n>>> STEP 4: Executing Rebalance Strategy")
    
    # The new money you want to inject into the portfolio this month
    # Fetch live prices and evaluate portfolio state
    current_prices = get_current_prices(my_portfolio.tags)
    asset_states, current_market_value = my_portfolio.get_current_portfolio_state(current_prices)
    
    rebalance_plan = my_portfolio.calculate_rebalance_contribution(new_monthly_contribution, current_prices)

    print("\n--- Recommended Action Plan ---")
    for tag, amount in rebalance_plan.items():
        if amount > 0:
            print(f"ACTION -> BUY {tag}: ${amount:.2f}")
        else:
            print(f"ACTION -> SELL {tag}: ${abs(amount):.2f}")
    

    # ==========================================
    # End. Save the Object to a File
    # ==========================================
    
    filename = "my_portfolio_provisional.pkl"

    print(f"\nSaving portfolio object to '{filename}'...")

    # Open the file in 'wb' (Write Binary) mode and dump the object
    with open(filename, "wb") as file:
        pickle.dump(my_portfolio, file)

    print("Success! Your portfolio is now saved and ready for the monthly report.")

if __name__ == "__main__":
    run_monthly_report()