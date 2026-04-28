import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Import the custom class and the mathematical estimators
from MarkowitzPortfolio_lib import Portfolio
from FinanceFunctions_lib import get_james_stein_expected_returns, get_ledoit_wolf_covariance, get_current_prices

def run_live_rebalance():
    print("==========================================")
    print(" LIVE PORTFOLIO REBALANCE EXECUTION")
    print("==========================================\n")

    portfolio_filename = "my_portfolio_provisional.pkl"
    
    # ==========================================
    # 1. Load the Saved Portfolio
    # ==========================================
    if os.path.exists(portfolio_filename):
        print(f">>> Loading existing portfolio object from '{portfolio_filename}'...")
        with open(portfolio_filename, "rb") as file:
            my_portfolio = pickle.load(file)
    else:
        print(f"Error: '{portfolio_filename}' not found. Cannot proceed.")
        return
        
    # ==========================================
    # 2. Get Current State & Market Values
    # ==========================================
    print("\n>>> Fetching Live Market Data for Current Holdings...")
    current_prices = get_current_prices(my_portfolio.tags)
    asset_states, current_market_value = my_portfolio.get_current_portfolio_state(current_prices)
    
    total_historically_invested = sum(my_portfolio.investments[tag]['total_invested'] for tag in my_portfolio.tags)
    profit = current_market_value - total_historically_invested
    roi = (profit / total_historically_invested) * 100 if total_historically_invested > 0 else 0

    print("\n" + "="*50)
    print(" PORTFOLIO SNAPSHOT")
    print("="*50)
    print(f"Total Historically Invested: ${total_historically_invested:.2f}")
    print(f"Current Market Value:        ${current_market_value:.2f}")
    print(f"Net Profit:                  ${profit:.2f} ({roi:.2f}%)")
    print("="*50 + "\n")

    # ==========================================
    # 3. Update Expectations & Optimize New Weights
    # ==========================================
    print(">>> Recalculating Expected Returns and Covariance...")
    tickers = my_portfolio.tags
    today = datetime.now()
    lookback_start = today - relativedelta(years=1)
    
    # Fetch new estimators using the trailing 1-year window
    # Note: You can swap James-Stein with EWMA here if you prefer
    new_expected_returns = get_james_stein_expected_returns(tickers, lookback_start, today, plotbool=True)
    new_cov_matrix = get_ledoit_wolf_covariance(tickers, lookback_start, today, plotbool=True)
    
    # Overwrite internal portfolio metrics
    my_portfolio.expected_returns = [new_expected_returns[tag] for tag in my_portfolio.tags]
    my_portfolio.set_covariance_matrix(new_cov_matrix)
    
    # Run the optimizer to find the new Tangency Portfolio
    print("\n>>> Optimizing for the Maximum Sharpe Ratio...")
    my_portfolio.optimize_maximize_sharpe_ratio(risk_free_rate=0.02)
    
    print("\n--- New Target Weights ---")
    for i, tag in enumerate(my_portfolio.tags):
        print(f"{tag}: {my_portfolio.weights[i]:.2%}")
    print("-" * 26 + "\n")

    # ==========================================
    # 4. Plot Total Invested vs Current Value
    # ==========================================
    print(">>> Generating Portfolio Performance Chart...")
    labels = ['Historically Invested', 'Current Market Value']
    values = [total_historically_invested, current_market_value]
    colors = ['#3498db', '#2ecc71' if profit >= 0 else '#e74c3c']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors, width=0.5)
    plt.title('Portfolio Performance Overview', fontsize=14, pad=15)
    plt.ylabel('Amount in USD ($)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), 
                 f'${yval:,.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Pause execution to show the plot to the user
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 5. Calculate Rebalancing
    # ==========================================
    new_contribution = 1000.0
    print(f"\n>>> Calculating Rebalance Strategy with a ${new_contribution:.2f} contribution...")
    
    # Create a simple dictionary of current live prices
    current_prices = {tag: asset_states[tag]['current_price'] for tag in my_portfolio.tags}
    
    # Pass the prices directly into the new method
    adjustments = my_portfolio.calculate_rebalance_contribution(new_contribution, current_prices)
    
    # ==========================================
    # 6. Apply the Rebalancing Trades
    # ==========================================
    print("\n>>> Executing Live Trades...")
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    for tag, amount in adjustments.items():
        # Ignore micro-adjustments smaller than 1 cent
        if abs(amount) > 0.01:
            current_price = asset_states[tag]['current_price']
            my_portfolio.add_money_to_asset_at_price(tag, amount, current_price, transaction_date=today_str)
            
    # ==========================================
    # 7. Save the Updated Portfolio
    # ==========================================
    print(f"\n>>> Saving updated portfolio back to '{portfolio_filename}'...")
    with open(portfolio_filename, "wb") as file:
        pickle.dump(my_portfolio, file)
        
    print("==========================================")
    print(" SUCCESS: Live rebalance complete and saved.")
    print("==========================================")

if __name__ == "__main__":
    run_live_rebalance()