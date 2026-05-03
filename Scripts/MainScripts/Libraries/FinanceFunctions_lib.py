import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
import logging
from yahooquery import search
import statsmodels.api as sm
import requests
import zipfile
import io
import re
import concurrent.futures

TRADING_DAYS = 252

def download_historical_prices_batch(tickers, start_date, end_date):
    """
    Downloads historical market data concurrently.
    Uses yf.Ticker().history() to ensure thread safety (preventing cross-contamination)
    and preserves yfinance's native cookie handling to bypass EU consent blocks.
    """
    import yfinance as yf
    import pandas as pd
    import logging
    import concurrent.futures
    import time
    import random

    logging.getLogger('yfinance').setLevel(logging.CRITICAL)

    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date

    ticker_list = list(tickers)
    if not ticker_list:
        return pd.DataFrame()

    print(f"[INFO] Downloading market data for {len(ticker_list)} assets concurrently...")

    series_dict = {}

    def fetch_single_ticker(ticker):
        # A tiny random delay (jitter) prevents triggering rate limits from simultaneous requests
        time.sleep(random.uniform(0.1, 0.4))
        try:
            # yf.Ticker creates an isolated object, fixing the shared-state race condition
            tkr = yf.Ticker(ticker)

            # auto_adjust=False ensures we get raw 'Adj Close' correctly mapped
            df = tkr.history(start=start_str, end=end_str, auto_adjust=False)

            if df.empty:
                return ticker, None

            # Safely extract Adjusted Close or Close
            if 'Adj Close' in df.columns:
                price_series = df['Adj Close']
            else:
                price_series = df['Close']

            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]

            # LSE stocks quoted in GBX (pence) must be divided by 100 to get GBP.
            # yfinance reports currency='GBp' for these (lowercase p = pence).
            try:
                if tkr.fast_info.currency == 'GBp':
                    price_series = price_series / 100
            except Exception:
                pass

            # Legacy cleaning logic to guarantee exact mathematical match
            price_series.index = pd.to_datetime(price_series.index).tz_localize(None).normalize()
            price_series = price_series[~price_series.index.duplicated(keep='first')]

            return ticker, price_series
        except Exception as e:
            print(f"[ERROR] Download failed for {ticker}: {e}")
            return ticker, None

    # Max workers set to 10 is the sweet spot to avoid Yahoo Finance 429 rate limit
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(fetch_single_ticker, t): t for t in ticker_list}
        
        for future in concurrent.futures.as_completed(futures):
            t, res = future.result()
            if res is not None:
                series_dict[t] = res

    if not series_dict:
        print("[WARNING] Concurrent download returned no valid data.")
        return pd.DataFrame()

    # Reconstruct the DataFrame exactly as the original loop did
    price_data = pd.DataFrame(series_dict)
    price_data = price_data.dropna(axis=0, how='all')

    return price_data
    
def get_current_prices(tickers):
    """
    Fetches the latest closing prices for the given tickers.
    Uses a 5-day lookback and forward-fill to safely handle weekends and holidays 
    where crypto trades (24/7) but traditional markets do not.
    """
    
    # Download a 5-day window to guarantee we catch the last active trading day
    data = yf.download(tickers, period="5d", progress=False)
    
    # Safely extract the closing prices
    if isinstance(data.columns, pd.MultiIndex):
        level_zero = data.columns.get_level_values(0)
        if 'Adj Close' in level_zero:
            price_data = data['Adj Close']
        else:
            price_data = data['Close']
    else:
        if 'Adj Close' in data.columns:
            price_data = data['Adj Close']
        else:
            price_data = data['Close']
            
    # Force Series to DataFrame to handle single-ticker queries uniformly
    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame()
        
    # Forward-fill (ffill) carries the last valid Friday price into Saturday/Sunday.
    # Then we safely take the very last row (.iloc[-1]).
    latest_prices = price_data.ffill().iloc[-1]
            
    return latest_prices.to_dict()


def _download_and_clean_data(tickers, start_date, end_date, prices_df=None):
    """
    Retrieves historical market data, cleans missing values safely,
    handles Crypto schedules, and calculates daily returns.
    If prices_df is provided, it extracts data from RAM instead of downloading.
    """
    import pandas as pd
    
    if prices_df is not None:
        # --- RAM EXTRACTION MODE ---
        valid_cols = [t for t in tickers if t in prices_df.columns]
        if not valid_cols:
            print("[WARNING] None of the requested tickers were found in the provided prices_df.")
            return pd.DataFrame()
        
        # Ensure tz-naive dates for slicing
        start_dt = pd.to_datetime(start_date).tz_localize(None)
        end_dt = pd.to_datetime(end_date).tz_localize(None)
        
        price_data = prices_df[valid_cols].loc[start_dt:end_dt].copy()
    else:
        # --- NETWORK DOWNLOAD MODE ---
        price_data = download_historical_prices_batch(tickers, start_date, end_date)

    if price_data.empty or price_data.shape[1] == 0:
        print("[WARNING] Price matrix is empty after retrieval.")
        return pd.DataFrame()

    # Filter out days where the majority of traditional assets are closed
    threshold = max(1, len(tickers) // 2)
    valid_market_days = price_data.dropna(thresh=threshold).index
    price_data = price_data.loc[valid_market_days]

    # Safe to ffill/bfill for isolated national holidays
    price_data = price_data.ffill().bfill()

    daily_returns = price_data.pct_change().dropna(how='all')

    return daily_returns

def compare_portfolio_realized_data(my_portfolio,start_date, end_date,plotbool=True):
    # Save the expected (theoretical) metrics
    fixed_weights = my_portfolio.weights
    tickers = my_portfolio.tags
    expected_return = my_portfolio.get_portfolio_return()
    expected_variance = my_portfolio.get_portfolio_variance()

    print(f"\nLocked Weights for Testing Period: {np.round(fixed_weights, 4)}")


    # --- B. Testing Phase (Out-of-Sample: 1 Year Ago -> Today) ---
    realized_returns, realized_cov = get_historical_metrics(tickers, start_date, end_date, False)

    # Apply our fixed weights to the realized market data
    realized_portfolio_return = np.dot(fixed_weights, realized_returns)
    realized_portfolio_variance = np.dot(fixed_weights.T, np.dot(realized_cov, fixed_weights))

    if plotbool:
            print("==========================================")
            print(" FINAL RESULTS: EXPECTED VS REALIZED")
            print("==========================================")
            print(f"Return   -> Expected: {expected_return:>7.2%} | Realized: {realized_portfolio_return:>7.2%}")
            print(f"Variance -> Expected: {expected_variance:>7.6f} | Realized: {realized_portfolio_variance:>7.6f}")
            print("==========================================\n")

    return realized_portfolio_return, realized_portfolio_variance


def get_historical_metrics(tickers, start_date, end_date,plotbool=True):
    """
    Fetches historical market data and calculates the annualized 
    expected returns and covariance matrix for a given timeframe.
    """
    # Handle date formatting for the print statement safely
    # Fetch and clean daily returns
    daily_returns = _download_and_clean_data(tickers, start_date, end_date)
    
    if daily_returns.empty:
        return pd.Series(), np.array([])

    # ==========================================
    # 2. Calculate Returns, Expected Returns, and Covariance
    # ==========================================
    expected_returns = daily_returns.mean() * TRADING_DAYS
    covariance_matrix = daily_returns.cov() * TRADING_DAYS
    cov_matrix_array = covariance_matrix.to_numpy()

    # ==========================================
    # 3. Display Results
    # ==========================================
    if plotbool:
        print(f"--- Annualized Expected Returns (\u03bc) ---")
        for ticker, return_val in expected_returns.items():
            print(f"{ticker}: {return_val:.2%}")

        print(f"\n--- Annualized Covariance Matrix (\u03a3) ---")
        print(np.round(cov_matrix_array, 5))
        print("=" * 40 + "\n")

        # Return the Series of returns and the numpy array of the covariance matrix
    return expected_returns, cov_matrix_array

def get_ewma_expected_returns(tickers, start_date, end_date, span=60, plotbool=True):
    """
    Fetches historical market data and calculates the annualized 
    expected returns using an Exponentially Weighted Moving Average (EWMA).
    
    :param span: The lookback window for the exponential weighting (default 60 days).
    """
    # Fetch and clean daily returns
    daily_returns = _download_and_clean_data(tickers, start_date, end_date)
    
    if daily_returns.empty:
        return pd.Series()

    # Calculate EWMA Returns
    # Calculate the exponential moving average and extract the final row (.iloc[-1])
    ewma_daily_returns = daily_returns.ewm(span=span).mean().iloc[-1]
    expected_returns = ewma_daily_returns * TRADING_DAYS

    if plotbool:
        print(f"--- Annualized EWMA Expected Returns (\u03bc) [Span: {span} days] ---")
        for ticker, return_val in expected_returns.items():
            print(f"{ticker}: {return_val:.2%}")
        print("=" * 40 + "\n")

    return expected_returns

def get_european_ticker(original_ticker):
    """
    Finds the European dual-listed ticker (Frankfurt/Xetra) for a given stock.
    Filters strictly for EQUITIES to avoid mapping US stocks to European Mutual Funds (0P...).
    """
    try:
        result = search(original_ticker)
        quotes = result.get('quotes', [])
        
        for quote in quotes:
            exchange = quote.get('exchange', '')
            quote_type = quote.get('quoteType', '')
            
            # Enforce 'EQUITY' type to reject Mutual Funds and ETFs
            if exchange in ['FRA', 'GER', 'XET'] and quote_type == 'EQUITY':
                print(f"[CORRECT MAPPING] EUR equivalent equity found for '{original_ticker}': {quote['symbol']}")
                return quote['symbol']
        
        print(f"[WARNING] No EUR equivalent equity found for '{original_ticker}'. Keeping original.")
        return original_ticker
        
    except Exception as e:
        print(f"[ERROR] Search failed for '{original_ticker}': {e}")
        return original_ticker

def get_prices_at_date(price_df, target_date):
    """
    Extracts the closest available prices for a specific historical date safely.
    """
    # Slice up to the target date
    sliced_df = price_df.loc[:target_date]
    if sliced_df.empty:
        return {}
    # Get the very last available row (simulating the closing price on that day)
    return sliced_df.iloc[-1].to_dict()

def fetch_ff_data(filename):
    print(f"Fetching {filename}...")
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{filename}"
    
    response = requests.get(url)
    response.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            lines = [line.decode('utf-8-sig', errors='ignore') for line in f.readlines()]
            
    header = ""
    valid_lines = []
    
    for line in lines:
        # Catch the header row regardless of exact casing or hidden spaces
        if 'Mkt-RF' in line or 'Mkt-Rf' in line or 'Mom' in line or 'WML' in line:
            header = line
        # Catch valid data rows (8-digit date followed by a comma)
        elif re.match(r'^\s*\d{8}\s*,', line):
            valid_lines.append(line)
            
    if not header and len(valid_lines) > 0:
        header = lines[0]
        
    clean_csv = io.StringIO(header + "".join(valid_lines))
    df = pd.read_csv(clean_csv, skipinitialspace=True)
    
    # 1. Clean up column names strictly
    df.columns = df.columns.str.strip()
    
    # 2. Force correct naming to prevent KeyErrors down the line
    rename_map = {df.columns[0]: 'Date'}
    for col in df.columns:
        col_upper = col.upper()
        if 'RF' in col_upper and 'MKT' not in col_upper:
            rename_map[col] = 'RF'
        elif 'MKT' in col_upper:
            rename_map[col] = 'Mkt-RF'
        elif 'MOM' in col_upper or 'WML' in col_upper:
            rename_map[col] = 'Mom'
            
    df.rename(columns=rename_map, inplace=True)
    
    # Format Date and metrics
    df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(), format='%Y%m%d')
    
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df.set_index('Date', inplace=True)
    return df

# Mapping of universe keys to their Kenneth French Data Library filenames.
# All files are hosted at: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
#
# Universe selection guide for your specific case (S&P500 mapped to EUR +
# EUROSTOXX600 + IBEX35, all executed in EUR via Trade Republic):
#
#   "europe"       → Only European-listed equities. Best fit if your S&P500
#                    names are genuinely traded on Xetra with real EUR price
#                    discovery (not just a nominal listing with zero volume).
#
#   "developed"    → All developed markets (NA + Europe + Pacific). Best fit
#                    if your S&P500 mapped tickers still have their fundamental
#                    economics driven by the US market.
#
#   "developed_ex_us" → Developed ex-US. Use this if you want to deliberately
#                       exclude US factor exposure from the benchmark.
#
# For a mixed universe like yours, "developed" is the most defensible default.

FF_UNIVERSE_CONFIG = {
    "europe": {
        "ff5": "Europe_5_Factors_daily_CSV.zip",
        "mom": "Europe_MOM_Factor_daily_CSV.zip",
    },
    "developed": {
        "ff5": "Developed_5_Factors_daily_CSV.zip",
        "mom": "Developed_MOM_Factor_daily_CSV.zip",
    },
    "developed_ex_us": {
        "ff5": "Developed_ex_US_5_Factors_daily_CSV.zip",
        "mom": "Developed_ex_US_MOM_Factor_daily_CSV.zip",
    },
}

def fetch_global_fama_french_daily(start_date, end_date, universe="developed"):

    if universe not in FF_UNIVERSE_CONFIG:
        raise ValueError(
            f"[ERROR] Unknown universe '{universe}'. "
            f"Choose from: {list(FF_UNIVERSE_CONFIG.keys())}"
        )

    config = FF_UNIVERSE_CONFIG[universe]

    print(f"[INFO] Fetching Fama-French factors for universe='{universe}'...")
    print(f"       5-Factor file : {config['ff5']}")
    print(f"       Momentum file : {config['mom']}")

    # fetch_ff_data constructs the full URL internally as:
    #   f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{filename}"
    # so pass only the filename, never the full URL.
    df_5f  = fetch_ff_data(config["ff5"])
    df_mom = fetch_ff_data(config["mom"])

    if "Mom" not in df_mom.columns:
        non_date_cols = [c for c in df_mom.columns if c != "Date"]
        if len(non_date_cols) == 1:
            df_mom.rename(columns={non_date_cols[0]: "Mom"}, inplace=True)
        else:
            raise ValueError(
                f"[ERROR] Cannot identify momentum column in {config['mom']}. "
                f"Columns found: {df_mom.columns.tolist()}"
            )

    df_mom = df_mom[["Mom"]]

    print("[INFO] Merging 5-Factor and Momentum datasets on Date index...")
    merged_df = df_5f.join(df_mom, how="inner")
    merged_df.reset_index(inplace=True)

    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)
    mask  = (merged_df["Date"] >= start) & (merged_df["Date"] <= end)
    final_df = merged_df.loc[mask].copy()

    if final_df.empty:
        raise ValueError(
            f"[ERROR] No Fama-French data available for universe='{universe}' "
            f"between {start_date} and {end_date}."
        )

    print(
        f"[INFO] Fama-French ({universe}) extracted: "
        f"{len(final_df)} trading days, "
        f"{final_df['Date'].min().date()} → {final_df['Date'].max().date()}"
    )
    return final_df

def get_james_stein_expected_returns(tickers, start_date, end_date, plotbool=True):
    """
    Fetches historical market data and calculates the annualized 
    expected returns using the James-Stein Shrinkage estimator.
    """
    daily_returns = _download_and_clean_data(tickers, start_date, end_date)
    
    if daily_returns.empty:
        return pd.Series(0.0, index=tickers)

    valid_tickers = daily_returns.columns.tolist()
    historical_means = daily_returns.mean() * TRADING_DAYS

    grand_mean = historical_means.mean()
    target_vector = np.full(len(valid_tickers), grand_mean)
    
    n_observations = len(daily_returns)
    time_series_variance = daily_returns.var().mean() * TRADING_DAYS
    cross_sectional_variance = historical_means.var()
    
    if cross_sectional_variance > 0:
        shrinkage_factor = (time_series_variance / n_observations) / cross_sectional_variance
    else:
        shrinkage_factor = 1.0
        
    w = max(0.0, min(1.0, shrinkage_factor))
    js_returns_array = (1 - w) * historical_means.to_numpy() + w * target_vector
    js_expected_returns = pd.Series(js_returns_array, index=valid_tickers)

    # Re-inject missing tickers with 0.0 expected return
    missing_tickers = [t for t in tickers if t not in valid_tickers]
    for t in missing_tickers:
        js_expected_returns[t] = 0.0
        
    # Ensure the output Series order perfectly matches the requested tickers
    js_expected_returns = js_expected_returns[tickers]

    if plotbool:
        print(f"--- Annualized James-Stein Expected Returns (\u03bc) ---")
        print(f"Grand Mean Target: {grand_mean:.2%}")
        print(f"Applied Shrinkage Factor (w): {w:.4f}")
        for ticker, return_val in js_expected_returns.items():
            print(f"{ticker}: {return_val:.2%}")
        print("=" * 40 + "\n")

    return js_expected_returns

def get_ledoit_wolf_covariance(tickers, start_date, end_date, plotbool=True, prices_df=None):
    """
    Fetches historical market data and calculates the annualized 
    covariance matrix using the Ledoit-Wolf shrinkage estimator.
    """
    daily_returns = _download_and_clean_data(tickers, start_date, end_date, prices_df=prices_df)
    
    # Return a safe identity matrix placeholder if everything fails
    if daily_returns.empty:
        return pd.DataFrame(np.eye(len(tickers)) * 0.04, index=tickers, columns=tickers)

    valid_tickers = daily_returns.columns.tolist()

    # Calculate Ledoit-Wolf Covariance
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf()
    lw.fit(daily_returns)
    
    cov_matrix_array = lw.covariance_ * TRADING_DAYS
    cov_matrix_df = pd.DataFrame(cov_matrix_array, index=valid_tickers, columns=valid_tickers)

    # Re-inject missing tickers with neutral dummy values
    missing_tickers = [t for t in tickers if t not in valid_tickers]
    if missing_tickers:
        cov_matrix_df = cov_matrix_df.reindex(index=tickers, columns=tickers, fill_value=0.0)
        for t in missing_tickers:
            cov_matrix_df.loc[t, t] = 0.04  

    if plotbool:
        print(f"--- Annualized Ledoit-Wolf Covariance Matrix (\u03a3) ---")
        print(np.round(cov_matrix_df, 5))
        print(f"Calculated Shrinkage Coefficient: {lw.shrinkage_:.4f}")
        print("=" * 40 + "\n")

    return cov_matrix_df

def get_james_stein_jorion_expected_returns(tickers, start_date, end_date, plotbool=True, prices_df=None):
    """
    Calculates annualized expected returns using the Jorion (1986) Bayes-Stein
    shrinkage estimator. Shrinks individual asset means toward the GMV portfolio.
    """
    daily_returns = _download_and_clean_data(tickers, start_date, end_date, prices_df=prices_df)

    if daily_returns.empty:
        return pd.Series(0.0, index=tickers)

    valid_tickers = daily_returns.columns.tolist()
    T, n = daily_returns.shape

    mu = daily_returns.mean().to_numpy()                     
    S  = daily_returns.cov().to_numpy()                       

    # Regularize S to guarantee invertibility
    S_reg = S + np.eye(n) * 1e-8

    try:
        S_inv = np.linalg.inv(S_reg)
    except np.linalg.LinAlgError:
        print("[WARNING] Covariance matrix is singular. Falling back to grand mean shrinkage.")
        return get_james_stein_expected_returns(tickers, start_date, end_date, plotbool, prices_df)

    # ... (El resto de esta función se mantiene exactamente igual hasta el final)
    # [Asegúrate de copiar el resto del código original de esta función de tu script]
    ones = np.ones(n)
    S_inv_ones  = S_inv @ ones
    denom_gmv   = ones @ S_inv_ones                           
    w_gmv       = S_inv_ones / denom_gmv                      
    mu_g        = float(w_gmv @ mu)                           
    diff        = mu - mu_g * ones                            
    delta       = float(T * diff @ S_inv @ diff)              
    w_star      = (n + 2) / (n + 2 + delta)
    mu_bs_daily = (1.0 - w_star) * mu + w_star * mu_g * ones
    mu_bs_annual = mu_bs_daily * TRADING_DAYS
    js_expected_returns = pd.Series(mu_bs_annual, index=valid_tickers)

    missing_tickers = [t for t in tickers if t not in valid_tickers]
    for t in missing_tickers:
        js_expected_returns[t] = 0.0
        
    js_expected_returns = js_expected_returns[tickers]
    return js_expected_returns