# config.py
# Centralized configuration for the investment strategy

# --- TIME WINDOWS ---
FUNDAMENTAL_WINDOW_YEARS = 3
MARKOWITZ_WINDOW_YEARS = 1

# --- ASSET UNIVERSE SETTINGS ---
# Tickers that the optimizer/strategy should NEVER sell (e.g., long-term core holdings)

ETFS = [
    'ICGA.DE',  # Theme: China (iShares MSCI China UCITS ETF USD Acc)
    'VVSM.DE',  # Theme: Semiconductors (VanEck Semiconductor UCITS ETF Acc)
    'XAIX.DE',  # Theme: Artificial Intelligence (Xtrackers AI & Big Data UCITS ETF Acc)
    'URNU.DE',  # Theme: Uranium (Global X Uranium UCITS ETF Acc)
    'HTMW.DE',  # Theme: Green Hydrogen (L&G Hydrogen Economy UCITS ETF Acc)
    'ICLN',  # Theme: Green Energy (iShares Global Clean Energy UCITS ETF Acc)
    'VVMX.DE',   # Theme: Rare Earths (VanEck Rare Earth and Strategic Metals UCITS ETF Acc)
    'SXR8.DE',    # Theme:  SP500
    'WQTM.DE',   #Quantum Computing
    'DFNS.PA'   # Defense
]

#WQTM.DE Theme: Quantum Computing

VALUE_REFUGES=['BTC-EUR', 'EGLN.L']

LOSER_COMPANIES = ['IDR.MC','JNJ.DE']

PROTECTED_TICKERS = ETFS + VALUE_REFUGES #Sustituir por ETFS +VALUE_REFUGES

CURRENT_PROTECTED_TICKERS = PROTECTED_TICKERS + LOSER_COMPANIES
# Tickers that are known to be problematic or delisted
NOT_AVAILABLE_TICKERS = []
MONTHLY_CONTRIBUTION_EUR = 1000.0  # Keep your standard value
# --- PORTFOLIO SETTINGS ---
# Top candidates allowed; 5 to 10 assets is the structural baseline.
TOP_CANDIDATES_COUNT = 4 
# Relaxed threshold to mimic a buy-and-hold approach and minimize tax drag.
HOLDING_ZONE_PERCENTILE = 75 

# --- RANKING STRATEGY (Z-Score Weights) ---
# Weights are balanced across Valuation, Quality, and Market Factors (Sum = 1.0)
DEFAULT_WEIGHTS_CONFIG = {
    # Fundamental Valuation Multipliers 0.4167 39%
    'EV_UFCF': 0.2167,           # Strict pure cash valuation multiple
    'FCF_Yield': 0.10,         # Direct cash return capacity
    'EV_EBIT': 0.0,            
    'Earnings_Yield': 0.0,
    'EV_EBITDA_CAPEX': 0.10,   # Aggressively penalizes high maintenance CAPEX requirements
    'EV_NOPAT': 0.0,         
    'EV_IC': 0.0,
    'EV_EBITDA': 0.0,      
    
    # Structural Financial Health Metrics 0.35  33%
    'ROA': 0.10,               # Asset utilization efficiency indicator
    'Interest_Coverage': 0.05, # Assesses debt service safety margin
    'Debt_to_Equity': 0.0,    
    'ROA_Trajectory': 0.05,    # Rewards improving operational leverage
    'DE_Trajectory': 0.10,     # Heavily rewards structural corporate deleveraging
    'Current_Ratio': 0.0,     
    'Revenue_Growth': 0.05,    # Year-over-year top-line expansion rate
    'CR_Trajectory': 0.0,
    
    # Market Performance and Factor Metrics 0.3  28%
    'Alpha_6F': 0.05,          # Primary proxy for persistent structural advantage
    'Momentum_12M_1M': 0.15,   # Medium-term price trend strength
    'Beta_Mkt': 0.10,          # Penalizes volatility to stabilize the downstream covariance matrix
}

# --- FUNDAMENTAL HEALTH THRESHOLDS ---
DEFAULT_HEALTH_CONFIG = {
    'roa_min': 0.0,                  # Eliminates structural cash-burning companies
    'ebit_min': 0.0,                 # Eliminates companies without core operating profit
    'interest_coverage_min': 1.5,    # Ensures ability to service debt obligations without stress
    'pass_threshold_financial': 4,   # Baseline structural health requirements
    'pass_threshold_standard': 5     # Asset must pass at least 5 out of the 7 discrete health signals
}