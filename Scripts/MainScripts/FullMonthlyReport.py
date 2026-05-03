# FullMonthlyReport.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Libraries'))
import config
from MarketSelection_lib import get_index_tickers
from InvestingStrategy_lib import get_monthly_recommendations

# Current portfolio quantities (This is the only thing that changes month to month)
CURRENT_HOLDINGS = {
    'IDR.MC': 418,
    'EGLN.L': 410,
    'BTC-EUR': 833.5,
    'JNJ.DE': 236.7,
    'VVSM.DE': 660,
    'VVMX.DE': 304,
    'URNU.L': 29
}

if __name__ == "__main__":
    # Scrape index tickers once
    target_indices = {
        "S&P 500": get_index_tickers("SP500"),
        "EUROSTOXX 600": get_index_tickers("EUROSTOXX600"),
        "IBEX 35": get_index_tickers("IBEX35"),
        "PROTECTED_ASSETS": config.PROTECTED_TICKERS
    }

    # Run recommendations using centralized config
    get_monthly_recommendations(
        indices_dict=target_indices,
        current_holdings=CURRENT_HOLDINGS,
        monthly_contribution=config.MONTHLY_CONTRIBUTION_EUR,
        top_candidates=config.TOP_CANDIDATES_COUNT,
        holding_zone=config.HOLDING_ZONE_PERCENTILE,
        custom_weights=config.DEFAULT_WEIGHTS_CONFIG,
        health_params=config.DEFAULT_HEALTH_CONFIG,
        not_available_tickers=config.NOT_AVAILABLE_TICKERS
    )