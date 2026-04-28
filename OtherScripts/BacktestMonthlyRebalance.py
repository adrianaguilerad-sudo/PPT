import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf

# Import the custom libraries
from MarkowitzPortfolio_lib import Portfolio
from FinanceFunctions_lib import get_ledoit_wolf_covariance, get_james_stein_expected_returns

def run_historical_money_simulation():
    tickers = ['SXR8.DE', 'IDR.MC', 'EGLN.L', 'BTC-EUR','JNJ.F']
    today = datetime.now()
    
    # We start exactly 12 months ago
    test_start = today - relativedelta(years=1)
    
    print("==========================================")
    print(" 12-MONTH FINANCIAL BACKTEST SIMULATION")
    print("==========================================\n")
    
    # Pre-fetch price data to get the exact closing prices on rebalance dates
    # We add a 10-day buffer to the start to ensure we catch the first trading day's price
    print("Pre-fetching historical price data for the simulation...")
    data = yf.download(tickers, start=test_start - relativedelta(days=10), end=today + relativedelta(days=1))
    
    if 'Adj Close' in data.columns.get_level_values(0):
        price_data = data['Adj Close'].dropna()
    else:
        price_data = data['Close'].dropna()

    my_portfolio = Portfolio()
    for t in tickers:
        my_portfolio.add_financial_asset(t, 0.0)
        
    monthly_contribution = 1000.0
    total_contributed = 0.0
    
    # Loop through 12 months
    for m in range(12):
        current_date = test_start + relativedelta(months=m)
        train_start = current_date - relativedelta(years=1)
        
        # Find the closest available trading price for this exact date
        date_str = current_date.strftime('%Y-%m-%d')
        current_prices = price_data.asof(date_str)
        
        print(f"\n{'='*50}")
        print(f" REBALANCE MONTH {m+1}: {date_str}")
        print(f"{'='*50}")
        
        # 1. Update Estimators & Optimize Weights (Silenced plots for cleaner console)
        #exp_ret = get_ewma_expected_returns(tickers, train_start, current_date, span=90, plotbool=False)
        exp_ret = get_james_stein_expected_returns(tickers, train_start, current_date, plotbool=True)

        cov_mat = get_ledoit_wolf_covariance(tickers, train_start, current_date, plotbool=False)
        
        my_portfolio.expected_returns = [exp_ret[t] for t in my_portfolio.tags]

        my_portfolio.set_covariance_matrix(cov_mat)
        
        my_portfolio.optimize_maximize_sharpe_ratio(risk_free_rate=0.02)
        
        print("\n--- Target Weights for this Month ---")
        for i, tag in enumerate(my_portfolio.tags):
            print(f"{tag}: {my_portfolio.weights[i]:.2%}")
            
        # 2. Sync 'total_invested' to current market value 
        # (Required so calculate_rebalance_contribution works with floating prices)
        current_portfolio_value = 0.0
        print("\n--- Current Holdings Before Rebalance ---")
        for tag in my_portfolio.tags:
            shares = my_portfolio.investments[tag]['total_shares']
            price = current_prices[tag]
            market_value = shares * price
            
            # Sync internal state to reality before calculating adjustments
            current_portfolio_value += market_value
            
            if shares > 0:
                print(f"{tag}: {shares:.4f} shares @ \u20ac{price:.2f} = \u20ac{market_value:.2f}")
        
        print(f"Total Portfolio Value: \u20ac{current_portfolio_value:.2f}")
        
        # 3. Add Contribution and Execute Trades
        print(f"\nAdding \u20ac{monthly_contribution:.2f} contribution...")
        total_contributed += monthly_contribution
        
        # Calculate how to distribute the new money to reach the target weights
        adjustments = my_portfolio.calculate_rebalance_contribution(monthly_contribution, current_prices)
        
        print("\n--- Executing Trades ---")
        for tag, amount in adjustments.items():
            price = current_prices[tag]
            # We skip micro-trades to avoid floating point clutter (e.g. trading $0.0001)
            if abs(amount) > 0.01: 
                my_portfolio.add_money_to_asset_at_price(tag, amount, price, transaction_date=date_str)
                
        # 4. Calculate New Portfolio Value Post-Rebalance
        new_portfolio_value = 0.0
        for tag in my_portfolio.tags:
            shares = my_portfolio.investments[tag]['total_shares']
            new_portfolio_value += shares * current_prices[tag]
            
        print(f"\n>>> END OF MONTH {m+1} PORTFOLIO VALUE: \u20ac{new_portfolio_value:.2f} <<<")

    # ==========================================
    # FINAL YEAR-END REVIEW
    # ==========================================
    # Get today's final prices
    final_prices = price_data.iloc[-1]
    final_portfolio_value = 0.0
    
    for tag in my_portfolio.tags:
        shares = my_portfolio.investments[tag]['total_shares']
        final_portfolio_value += shares * final_prices[tag]
        
    total_profit = final_portfolio_value - total_contributed
    roi = (total_profit / total_contributed) * 100
    
    print("\n" + "="*50)
    print(" 1-YEAR SIMULATION RESULTS (THE MOMENT OF TRUTH)")
    print("="*50)
    print(f"Total Cash Injected:   \u20ac{total_contributed:.2f}")
    print(f"Final Portfolio Value: \u20ac{final_portfolio_value:.2f}")
    print(f"Total Net Profit:      \u20ac{total_profit:.2f}")
    print(f"Return on Investment:  {roi:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_historical_money_simulation()