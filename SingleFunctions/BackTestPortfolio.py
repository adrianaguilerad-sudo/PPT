# BackTestPortfolio.py
import config
from MarketSelection_lib import get_index_tickers
from InvestingStrategy_lib import run_historical_backtest

if __name__ == "__main__":
    target_indices = {
        "S&P 500": get_index_tickers("SP500"),
        "EUROSTOXX 600": get_index_tickers("EUROSTOXX600"),
        "IBEX 35": get_index_tickers("IBEX35"),
        "PROTECTED_ASSETS": config.PROTECTED_TICKERS
    }
    
    mean_monthly_net_roi, total_trades_executed = run_historical_backtest(
        indices_dict=target_indices,
        months_to_simulate=12,
        monthly_contribution=config.MONTHLY_CONTRIBUTION_EUR,
        top_candidates=config.TOP_CANDIDATES_COUNT,
        holding_zone=config.HOLDING_ZONE_PERCENTILE,
        custom_weights=config.DEFAULT_WEIGHTS_CONFIG,
        health_params=config.DEFAULT_HEALTH_CONFIG,
        not_available_tickers=config.NOT_AVAILABLE_TICKERS
    )

    print(f"\n[SUMMARY] Mean Monthly ROI: {mean_monthly_net_roi:.2f}% | Total Trades: {total_trades_executed}")

    print("\n" + "="*50)
    print(f" Mean Monthly Net ROI:  {mean_monthly_net_roi:.2f}%")
    print(f" Total Trades Executed: {total_trades_executed}")
    print("="*50 + "\n")