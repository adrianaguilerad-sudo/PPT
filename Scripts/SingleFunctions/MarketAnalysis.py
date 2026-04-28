import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MainScripts', 'Libraries'))
import pandas as pd
from MarketSelection_lib import get_index_tickers, load_market_data, rank_potential_market_winners, rank_potential_multiplier_winners
import warnings


# Suppress yfinance warnings for cleaner console output
#warnings.filterwarnings("ignore")

def main():
    # 1. Define parameters
    RISK_FREE_RATE = 0.02
    
    # 2. Execute pipeline
    print("--- STARTING QUANTITATIVE PIPELINE ---")
    
    # Fetch tickers
    tickers = get_index_tickers()
    
    # Load raw price data (This is the clean Pandas DataFrame)
    prices_df = load_market_data(tickers)
    
    # Rank potential winners using Alpha and Momentum directly on raw prices
    market_winners = rank_potential_market_winners(prices_df, rf_rate=RISK_FREE_RATE)

    # Rank potential winners using Alpha and Momentum directly on raw prices
    multipliers_winners = rank_potential_multiplier_winners(tickers)
    
    # Safety check: ensure the ranking function didn't return None due to bad inputs
    if market_winners is None:
        print("CRITICAL ERROR: Pipeline halted because ranking failed. Check input data types.")
        return
    if multipliers_winners is None:
        print("CRITICAL ERROR: Pipeline halted because ranking failed. Check input data types.")
        return
    
    # 3. Display results
    print("\n--- FINAL MARKET WINNER COMPANIES (Top 50) ---")
    print(market_winners.head(50).to_string(index=False))
    print("---------------------------------------")
    print("\n--- FINAL MULTIPLIERS WINNER COMPANIES (Top 50) ---")
    print(multipliers_winners.head(50).to_string(index=False))
    print("---------------------------------------")

if __name__ == "__main__":
    main()