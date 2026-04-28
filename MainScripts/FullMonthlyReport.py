# FullMonthlyReport.py
import config
from MarketSelection_lib import get_index_tickers
from InvestingStrategy_lib import get_monthly_recommendations

# Current portfolio quantities (This is the only thing that changes month to month)
CURRENT_HOLDINGS = {
    'IDR.MC': 423,
    'EGLN.L': 420,
    'BTC-EUR': 831,
    'JNJ.DE': 236
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