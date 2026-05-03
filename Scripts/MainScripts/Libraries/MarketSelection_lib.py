
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.covariance import LedoitWolf
from scipy.linalg import eigh
import warnings
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from yahooquery import Ticker
from tqdm import tqdm
import pandas as pd
import logging
from scipy.stats import linregress
import concurrent.futures
import os
import pickle
import time
import random
from yahooquery import Ticker as YQTicker
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from FinanceFunctions_lib import fetch_global_fama_french_daily, download_historical_prices_batch

class MarketAnalyzer:
    """
    Centralized OOP engine to manage historical market data, fundamental metrics, 
    and ranking logic for portfolio universe selection.
    """
    def __init__(self, start_date, end_date, indices_dict, market_proxy="VWCE.DE", ff_universe="developed"):
        """
        Initializes the MarketAnalyzer engine.
        Sets up the investment universe and triggers the caching sequence 
        for prices, fundamentals, statistical factors, and FX rates.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.indices_dict = indices_dict
        self.market_proxy = market_proxy
        self.ff_universe = ff_universe
        
        # Flatten the indices dictionary to get a unique list of all targeted tickers
        self.all_tickers = []
        for index_name, tickers in self.indices_dict.items():
            self.all_tickers.extend(tickers)
            
        # Remove duplicates while preserving list structure
        self.all_tickers = list(dict.fromkeys(self.all_tickers))
        
        # Ensure the market proxy is included in the download universe for Beta calculations
        if self.market_proxy not in self.all_tickers:
            self.all_tickers.append(self.market_proxy)
            
        # Initialize internal RAM databases
        self.prices_db = None
        self.fundamentals_db = None
        self.ff_factors_db = None
        self.fx_rates_db = {}
        
        # Trigger the data ingestion pipeline
        self._initialize_database()

    def _initialize_database(self):
        """
        Loads the core data components into the analyzer's RAM.
        Delegates the heavy lifting to the centralized caching function to 
        bypass redundant network I/O during backtest loops.
        """
        print(f"[INFO] Initializing MarketAnalyzer universe for {len(self.all_tickers)} unique assets...")
        
        # The cache returns 4 distinct dataframes/dictionaries now
        self.prices_db, self.fundamentals_db, self.ff_factors_db, self.fx_rates_db = \
            load_or_update_market_cache(
                tickers=self.all_tickers,
                start_date=self.start_date,
                end_date=self.end_date,
                ff_universe=self.ff_universe
            )
            
        print(">>> [INFO] MarketAnalyzer DB (Prices, Fundamentals, FF Factors, and FX Cache) successfully loaded into RAM.")
        

    def get_market_performance_coefficients(self, window_start, window_end,
                                            min_days=256, momentum_window=21,
                                            ff_universe="developed"):
        """
        Calculates Multi-Factor Alpha (Fama-French 5-Factor + Momentum) and
        Factor Loadings for each asset in the universe.

        Methodology fix vs. previous version
        -------------------------------------
        The previous implementation converted EUR prices to USD before running
        the OLS regression, arguing that this "preserves the covariance structure"
        against USD-denominated FF factors. That reasoning is circular and wrong
        for two reasons:

        1. The standard FF datasets from French's library are available for
            geographic universes (Europe, Developed, Global). Using the matching
            geographic factors in the asset's *own* currency is the correct and
            academically standard approach. See:
            - Fama & French (2012) "Size, value, and momentum in international
            stock returns", Journal of Financial Economics.
            - Fama & French (2017) "International tests of a five-factor asset
            pricing model", Journal of Financial Economics.

        2. Multiplying all EUR prices by the same EURUSD series before computing
            returns adds a common FX return component (r_EURUSD) to every asset
            identically. This does NOT change relative rankings but DOES inflate
            intercepts (alphas) and contaminate betas with FX exposure that is
            orthogonal to the equity factors being estimated.

        The fix: use FF factors denominated in the same base currency as the
        traded assets (EUR for European/Frankfurt-listed equities) by selecting
        the "developed" or "europe" universe from French's library, and run the
        regression directly on EUR log-returns without any FX transformation.

        Parameters
        ----------
        ff_universe : str
            Passed to fetch_global_fama_french_daily. Use "developed" for a
            mixed universe that includes fundamentally-US companies traded on
            Frankfurt, "europe" for a purely European universe.
        """
        print(
            f"[INFO] Extracting 6-factor coefficients [{ff_universe} factors] "
            f"between {window_start.strftime('%Y-%m-%d')} and "
            f"{window_end.strftime('%Y-%m-%d')}..."
        )

        prices_df = self.prices_db.loc[window_start:window_end].copy()

        if prices_df.empty:
            print("[WARNING] Sliced price dataframe is empty for the requested dates.")
            return None

        # --- 1. Clean prices ---
        prices_df = prices_df.where(prices_df > 0, np.nan)
        prices_df.ffill(inplace=True)

        # --- 2. Compute EUR returns (no FX conversion) ---
        start_dt = pd.to_datetime(window_start).tz_localize(None)
        end_dt   = pd.to_datetime(window_end).tz_localize(None)

        # Use arithmetic returns to match Fama-French methodology
        asset_returns = prices_df.pct_change().dropna(how="all")
        
        # Security filter: Drop assets with unadjusted stock splits or corrupted API data
        # Any genuine daily equity move > 50% or < -50% is extremely rare.
        clean_assets = []
        for col in asset_returns.columns:
            if asset_returns[col].max() < 0.5 and asset_returns[col].min() > -0.5:
                clean_assets.append(col)
            #else:
                #print(f"[WARNING] Dropping {col} due to corrupted price data (extreme daily jump).")
                
        asset_returns = asset_returns[clean_assets]
        asset_returns.index = pd.to_datetime(asset_returns.index).tz_localize(None)

        # --- 3. Load the geographically matching FF factors ---
        if self.ff_factors_db is None or self.ff_factors_db.empty:
            raise ValueError(
                "[ERROR] Fama-French factors database is empty. "
                "Check that load_or_update_market_cache ran successfully."
            )

        ff_factors = self.ff_factors_db.copy()

        # Normalise index to tz-naive datetime
        if "Date" in ff_factors.columns:
            ff_factors["Date"] = pd.to_datetime(ff_factors["Date"]).dt.tz_localize(None)
            ff_factors.set_index("Date", inplace=True)

        if hasattr(ff_factors.index, "tz") and ff_factors.index.tz is not None:
            ff_factors.index = ff_factors.index.tz_localize(None)

        mask = (ff_factors.index >= start_dt) & (ff_factors.index <= end_dt)
        ff_factors = ff_factors.loc[mask].copy()

        if ff_factors.empty:
            raise ValueError(
                f"[ERROR] No Fama-French data available in cache for "
                f"{start_dt.date()} → {end_dt.date()}. "
                f"Rebuild the cache with ff_universe='{ff_universe}'."
            )

        # French files are already in decimal form after fetch_ff_data divides by 100.
        # Defensive re-check: if any factor column has absolute mean > 0.1 it is
        # still in percentage form and needs conversion.
        ff_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
        for col in ff_cols:
            if col in ff_factors.columns:
                ff_factors[col] = pd.to_numeric(ff_factors[col], errors="coerce")
                ff_factors[col] /= 100.0

        ff_factors.rename(columns={"RF": "RF_rate"}, inplace=True)

        # --- 4. Inner join: only regress on days where both asset and factors exist ---
        aligned_data = pd.merge(
            asset_returns,
            ff_factors,
            left_index=True,
            right_index=True,
            how="inner"
        )

        if aligned_data.empty:
            print("[WARNING] No overlapping dates between asset returns and FF factors.")
            return None

        rf_daily  = aligned_data["RF_rate"]
        factors_X = aligned_data[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]
        X_mat     = sm.add_constant(factors_X)

        # --- 5. OLS regression per asset ---
        results_list = []

       # print(
       #     f"[INFO] Running 6-factor OLS on EUR returns "
       #     f"({ff_universe} factors, {len(aligned_data)} overlapping days)..."
       # )

        for ticker in asset_returns.columns:
            if ticker not in aligned_data.columns:
                continue

            # Excess return of the asset over the risk-free rate
            y = aligned_data[ticker] - rf_daily

            valid_idx = y.notna()
            if valid_idx.sum() < min_days:
                continue

            model = sm.OLS(y[valid_idx], X_mat.loc[valid_idx])
            res   = model.fit()

            # Annualise the daily intercept (alpha).
            # Regression uses arithmetic returns (pct_change), so compound annualisation
            # is (1 + alpha_daily)^252 - 1, not the log-return approximation alpha*252.
            daily_alpha      = res.params["const"]
            annualized_alpha = (1 + daily_alpha) ** 252 - 1

            results_list.append({
                "Ticker":    ticker,
                "Alpha_6F":  annualized_alpha,
                "Beta_Mkt":  res.params.get("Mkt-RF", np.nan),
                "Beta_SMB":  res.params.get("SMB",    np.nan),
                "Beta_HML":  res.params.get("HML",    np.nan),
                "Beta_Mom":  res.params.get("Mom",    np.nan),
                "R_Squared": res.rsquared,
            })

        if not results_list:
            print("[WARNING] No assets had enough data points for regression.")
            return None

        results_df = pd.DataFrame(results_list)

        # --- 6. Momentum features on EUR prices (unchanged: currency-consistent) ---
        if len(prices_df) >= min_days:
            mom_df     = compute_momentum_features(prices_df)
            results_df = pd.merge(results_df, mom_df, on="Ticker", how="left")

        print(
            f"[INFO] Alphas and Betas extracted for {len(results_df)} assets "
            f"using {ff_universe} FF factors in EUR."
        )
        return results_df.dropna()

    def _assess_financial_health(self, info, financials, balance_sheet, ticker="Unknown", health_params=None):
        """
        Evaluates structural financial health using a discrete scoring system.
        All thresholds can be customized via the health_params dictionary.
        """
        # --- 1. Define Default Parameters ---
        defaults = {
            'roa_min': 0.0,
            'ebit_min': 0.0,
            'interest_coverage_min': 1.5,
            'pass_threshold_financial': 3,
            'pass_threshold_standard': 4,
            'max_score_financial': 5,
            'max_score_standard': 7
        }
        
        # Update defaults with user-provided parameters if any
        if health_params:
            defaults.update(health_params)
            
        score = 0
        sector = info.get('sector', '')
        is_financial_or_reit = sector in ['Financial Services', 'Real Estate']
        
        def safe_float(val):
            try: return float(val) if val is not None else np.nan
            except (ValueError, TypeError): return np.nan

        try:
            # ==========================================
            # 1. PROFITABILITY & EFFICIENCY SIGNALS
            # ==========================================
            # Signal 1: Positive Return on Assets (ROA)
            net_income_0 = safe_float(financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else np.nan)
            total_assets_0 = safe_float(balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else np.nan)
            
            roa_0 = net_income_0 / total_assets_0 if total_assets_0 > 0 else np.nan
            if pd.notna(roa_0) and roa_0 > defaults['roa_min']:
                score += 1

            # Signal 2: Positive Operating Income (EBIT)
            ebit_0 = safe_float(financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else np.nan)
            if pd.notna(ebit_0) and ebit_0 > defaults['ebit_min']:
                score += 1

            # Signal 3: Improving ROA (YoY Trajectory)
            if financials.shape[1] > 1 and balance_sheet.shape[1] > 1:
                net_income_1 = safe_float(financials.loc['Net Income'].iloc[1] if 'Net Income' in financials.index else np.nan)
                total_assets_1 = safe_float(balance_sheet.loc['Total Assets'].iloc[1] if 'Total Assets' in balance_sheet.index else np.nan)
                
                roa_1 = net_income_1 / total_assets_1 if total_assets_1 > 0 else np.nan
                if pd.notna(roa_0) and pd.notna(roa_1) and roa_0 > roa_1:
                    score += 1

            # Signal 4: Positive Revenue Growth (YoY)
            if 'Total Revenue' in financials.index and financials.shape[1] > 1:
                rev_0 = safe_float(financials.loc['Total Revenue'].iloc[0])
                rev_1 = safe_float(financials.loc['Total Revenue'].iloc[1])
                
                if pd.notna(rev_0) and pd.notna(rev_1) and rev_1 > 0:
                    if rev_0 > rev_1:
                        score += 1

            # ==========================================
            # 2. LEVERAGE & LIQUIDITY SIGNALS 
            # ==========================================
            if not is_financial_or_reit:
                # Signal 5: Decreasing Leverage (YoY Debt to Equity)
                total_debt_0 = safe_float(balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else np.nan)
                total_equity_0 = safe_float(balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else np.nan)
                de_0 = total_debt_0 / total_equity_0 if total_equity_0 > 0 else np.nan

                if balance_sheet.shape[1] > 1:
                    total_debt_1 = safe_float(balance_sheet.loc['Total Debt'].iloc[1] if 'Total Debt' in balance_sheet.index else np.nan)
                    total_equity_1 = safe_float(balance_sheet.loc['Stockholders Equity'].iloc[1] if 'Stockholders Equity' in balance_sheet.index else np.nan)
                    de_1 = total_debt_1 / total_equity_1 if total_equity_1 > 0 else np.nan
                    
                    if pd.notna(de_0) and pd.notna(de_1) and de_0 < de_1:
                        score += 1

                # Signal 6: Improving Short-Term Liquidity (Current Ratio YoY)
                ca_keys = ['Total Current Assets', 'Current Assets', 'CurrentAssets']
                cl_keys = ['Total Current Liabilities', 'Current Liabilities', 'CurrentLiabilities']
                
                ca_row = next((balance_sheet.loc[k] for k in ca_keys if k in balance_sheet.index), None)
                cl_row = next((balance_sheet.loc[k] for k in cl_keys if k in balance_sheet.index), None)

                current_assets_0 = safe_float(ca_row.iloc[0] if ca_row is not None else np.nan)
                current_liabs_0 = safe_float(cl_row.iloc[0] if cl_row is not None else np.nan)
                cr_0 = current_assets_0 / current_liabs_0 if current_liabs_0 > 0 else np.nan

                if balance_sheet.shape[1] > 1:
                    current_assets_1 = safe_float(ca_row.iloc[1] if ca_row is not None else np.nan)
                    current_liabs_1 = safe_float(cl_row.iloc[1] if cl_row is not None else np.nan)
                    cr_1 = current_assets_1 / current_liabs_1 if current_liabs_1 > 0 else np.nan

                    if pd.notna(cr_0) and pd.notna(cr_1) and cr_0 > cr_1:
                        score += 1

            # ==========================================
            # 3. SAFETY SNAPSHOT (Altman Z-Score logic)
            # ==========================================
            # Signal 7: Safe Interest Coverage
            interest_expense = safe_float(financials.loc['Interest Expense'].iloc[0] if 'Interest Expense' in financials.index else np.nan)
            if pd.notna(ebit_0) and pd.notna(interest_expense):
                interest_expense = abs(interest_expense)
                if interest_expense > 0:
                    interest_coverage = ebit_0 / interest_expense
                    if interest_coverage > defaults['interest_coverage_min']:
                        score += 1

            # ==========================================
            # VERDICT
            # ==========================================
            pass_threshold = defaults['pass_threshold_financial'] if is_financial_or_reit else defaults['pass_threshold_standard']
            
            return score >= pass_threshold

        except Exception as e:
            print(f"[WARNING] Structural health check failed for {ticker}: {e}")
            return False

    def get_multipliers(self, target_date, current_prices, min_market_cap=1e9, lookback_years=3, health_params=None):
        """
        Calculates valuation multipliers using the internal fundamentals database.
        FX conversion is 100% RAM-based using the pre-cached fx_rates_db to avoid network bottlenecks.
        """
        import pandas as pd
        import numpy as np
        
        target_date_pd = pd.to_datetime(target_date).tz_localize(None)
        current_date_pd = pd.to_datetime('today').tz_localize(None)
        
        is_live_trading = (current_date_pd - target_date_pd).days <= 5
        
        if is_live_trading:
            safe_target_date = target_date_pd
            print(f"[INFO] Live execution detected. Extracting fundamental multipliers for {len(self.all_tickers)} assets at {target_date_pd.strftime('%Y-%m-%d')}...")
        else:
            safe_target_date = target_date_pd - pd.Timedelta(days=60)
            print(f"[INFO] Backtest execution detected. Applying 60-day fundamental lag for target date {target_date_pd.strftime('%Y-%m-%d')}...")
            
        def get_dynamic_fx_rate(from_curr, to_curr, date_target):
            """Internal helper to fetch cached historical cross-currency rates directly from RAM."""
            if from_curr == 'GBp' and to_curr == 'GBP': return 0.01
            if from_curr == 'GBP' and to_curr == 'GBp': return 100.0

            # GBp to any non-GBP currency: convert pence to pounds (÷100) then apply FX rate.
            # Without this scale, aligned_price would be 100x too large (pence treated as pounds).
            pence_scale = 1.0
            if from_curr == 'GBp':
                pence_scale = 0.01
                from_curr = 'GBP'
            if to_curr == 'GBp': to_curr = 'GBP'
            if from_curr == to_curr: return 1.0 * pence_scale

            pair = f"{from_curr}{to_curr}=X"
            series = self.fx_rates_db.get(pair)

            if series is not None and not series.empty:
                # Ensure index is timezone-naive for safe slicing
                if series.index.tz is not None:
                    series.index = series.index.tz_localize(None)
                sliced = series.loc[:date_target]
                if not sliced.empty:
                    return sliced.iloc[-1] * pence_scale
            return None

        results = []
        for ticker in self.all_tickers:
            try:
                if ticker not in self.fundamentals_db or self.fundamentals_db[ticker] is None:
                    continue
                    
                data = self.fundamentals_db[ticker]
                info = data.get('info', {})
                cf = data.get('cash_flow')
                financials = data.get('financials')
                bs = data.get('balance_sheet')
                
                # Check for empty dataframes
                if cf is None or financials is None or bs is None or cf.empty or financials.empty or bs.empty:
                    continue

                cf = cf.copy()
                financials = financials.copy()
                bs = bs.copy()

                # Strip future data and limit lookback window
                cf = cf.loc[:, cf.columns.tz_localize(None) <= safe_target_date].iloc[:, :lookback_years]
                financials = financials.loc[:, financials.columns.tz_localize(None) <= safe_target_date].iloc[:, :lookback_years]
                bs = bs.loc[:, bs.columns.tz_localize(None) <= safe_target_date].iloc[:, :lookback_years]
                
                if cf.empty or financials.empty or bs.empty:
                    continue
                    
                shares_out = info.get('sharesOutstanding')
                historical_price = current_prices.get(ticker) if current_prices else info.get('previousClose')
                
                if shares_out is None or historical_price is None or pd.isna(historical_price):
                    continue

                price_currency = info.get('currency', 'USD')
                financial_currency = info.get('financialCurrency', 'USD')
                aligned_price = historical_price
                
                # Fast RAM-based FX conversion
                if price_currency != financial_currency:
                    conversion_rate = get_dynamic_fx_rate(price_currency, financial_currency, safe_target_date)
                    if conversion_rate is not None:
                        aligned_price = historical_price * conversion_rate
                    else:
                        print(f"[CRITICAL] Missing FX data ({price_currency}->{financial_currency}) for {ticker}. Asset discarded.")
                        continue
                
                # Calculate Market Cap precisely in the FINANCIAL CURRENCY
                market_cap = shares_out * aligned_price
                
                if market_cap < min_market_cap:
                    continue

                total_debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index and not bs.loc['Total Debt'].empty else 0
                cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index and not bs.loc['Cash And Cash Equivalents'].empty else 0
                
                ev = market_cap + total_debt - cash
                
                if ev <= 0:
                    continue
                    
                total_equity = bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index and not bs.loc['Stockholders Equity'].empty else np.nan
                
                is_healthy = self._assess_financial_health(info, financials, bs, ticker=ticker, health_params=health_params)
                if not is_healthy:
                    continue

                ebit = financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else (financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index else np.nan)
                ebitda = financials.loc['EBITDA'].iloc[0] if 'EBITDA' in financials.index else (financials.loc['Normalized EBITDA'].iloc[0] if 'Normalized EBITDA' in financials.index else np.nan)
                capex = abs(cf.loc['Capital Expenditure'].iloc[0]) if 'Capital Expenditure' in cf.index else 0
                
                tax_provision = financials.loc['Tax Provision'].iloc[0] if 'Tax Provision' in financials.index else 0
                pretax_income = financials.loc['Pretax Income'].iloc[0] if 'Pretax Income' in financials.index else 1
                tax_rate = max(0, min(1, tax_provision / pretax_income)) if pd.notna(tax_provision) and pretax_income > 0 else 0.21
                
                nopat = ebit * (1 - tax_rate) if pd.notna(ebit) else np.nan
                ocf = cf.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cf.index else 0
                interest_expense = financials.loc['Interest Expense'].iloc[0] if 'Interest Expense' in financials.index else 0
                
                ufcf = ocf - capex + (abs(interest_expense) * (1 - tax_rate))
                invested_capital = total_debt + total_equity - cash if pd.notna(total_equity) else np.nan
                
                ev_ufcf = ev / ufcf if ufcf > 0 else np.nan
                ev_nopat = ev / nopat if pd.notna(nopat) and nopat > 0 else np.nan
                ev_ic = ev / invested_capital if pd.notna(invested_capital) and invested_capital > 0 else np.nan
                
                ebitda_minus_capex = ebitda - capex if pd.notna(ebitda) else np.nan
                ev_ebitda_capex = ev / ebitda_minus_capex if pd.notna(ebitda_minus_capex) and ebitda_minus_capex > 0 else np.nan
                
                ev_ebitda = ev / ebitda if pd.notna(ebitda) and ebitda > 0 else np.nan
                ev_ebit = ev / ebit if pd.notna(ebit) and ebit > 0 else np.nan
                fcf_yield = (ocf - capex) / market_cap if pd.notna(ocf) and market_cap > 0 else np.nan
                earnings_yield = ebit / ev if pd.notna(ebit) and ev > 0 else np.nan

                results.append({
                    'Ticker': ticker,
                    'Sector': info.get('sector', 'Unknown'),
                    'EV_EBITDA': ev_ebitda,
                    'EV_EBIT': ev_ebit,
                    'FCF_Yield': fcf_yield,
                    'Earnings_Yield': earnings_yield,
                    'EV_UFCF': ev_ufcf,
                    'EV_NOPAT': ev_nopat,
                    'EV_IC': ev_ic,
                    'EV_EBITDA_CAPEX': ev_ebitda_capex
                })
                
            except Exception as e:
                # Silently skip errors to avoid spamming the console during backtest loops
                continue

        df = pd.DataFrame(results)
        if df.empty:
            print("[WARNING] No valid fundamental data extracted.")
            return None
            
        for col in ['EV_UFCF', 'EV_NOPAT', 'EV_IC', 'EV_EBITDA_CAPEX']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    
    def get_financials(self, target_date, lookback_years=3):
        """
        Extracts structural financial health variables (inspired by Piotroski F-Score 
        and Altman Z-Score) as continuous metrics. Returns a DataFrame formatted 
        to be ranked by the cross-sectional Z-score engine.
        """
        target_date_pd = pd.to_datetime(target_date).tz_localize(None)
        current_date_pd = pd.to_datetime('today').tz_localize(None)
        
        is_live_trading = (current_date_pd - target_date_pd).days <= 5
        if is_live_trading:
            safe_target_date = target_date_pd
            print(f"[INFO] Extracting continuous financial health metrics for {len(self.all_tickers)} assets at {target_date_pd.strftime('%Y-%m-%d')}...")
        else:
            safe_target_date = target_date_pd - pd.Timedelta(days=60)
            print(f"[INFO] Applying 60-day lag. Extracting financial metrics for target date {target_date_pd.strftime('%Y-%m-%d')}...")

        results = []

        def safe_float(val):
            try: return float(val) if val is not None else np.nan
            except (ValueError, TypeError): return np.nan

        for ticker in self.all_tickers:
            try:
                if ticker not in self.fundamentals_db or self.fundamentals_db[ticker] is None:
                    continue
                    
                data = self.fundamentals_db[ticker]
                info = data['info']
                financials = data['financials'].copy()
                bs = data['balance_sheet'].copy()
                
                # Strip future data strictly using the lagged target date
                financials = financials.loc[:, financials.columns.tz_localize(None) <= safe_target_date].iloc[:, :lookback_years]
                bs = bs.loc[:, bs.columns.tz_localize(None) <= safe_target_date].iloc[:, :lookback_years]
                
                if financials.empty or bs.empty:
                    continue
                    
                # ==========================================
                # 1. PROFITABILITY & EFFICIENCY METRICS
                # ==========================================
                ni_0 = safe_float(financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else np.nan)
                ta_0 = safe_float(bs.loc['Total Assets'].iloc[0] if 'Total Assets' in bs.index else np.nan)
                roa_0 = ni_0 / ta_0 if ta_0 > 0 else np.nan
                
                ebit_0 = safe_float(financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else np.nan)
                
                roa_trajectory = np.nan
                if financials.shape[1] > 1 and bs.shape[1] > 1:
                    ni_1 = safe_float(financials.loc['Net Income'].iloc[1] if 'Net Income' in financials.index else np.nan)
                    ta_1 = safe_float(bs.loc['Total Assets'].iloc[1] if 'Total Assets' in bs.index else np.nan)
                    roa_1 = ni_1 / ta_1 if ta_1 > 0 else np.nan
                    
                    if pd.notna(roa_0) and pd.notna(roa_1):
                        roa_trajectory = roa_0 - roa_1 # Higher is better
                        
                rev_growth = np.nan
                if 'Total Revenue' in financials.index and financials.shape[1] > 1:
                    rev_0 = safe_float(financials.loc['Total Revenue'].iloc[0])
                    rev_1 = safe_float(financials.loc['Total Revenue'].iloc[1])
                    if pd.notna(rev_0) and pd.notna(rev_1) and rev_1 > 0:
                        rev_growth = (rev_0 / rev_1) - 1 # Higher is better

                # ==========================================
                # 2. LEVERAGE & LIQUIDITY METRICS
                # ==========================================
                td_0 = safe_float(bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else np.nan)
                te_0 = safe_float(bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else np.nan)
                de_0 = td_0 / te_0 if te_0 > 0 else np.nan # Lower is better
                
                de_trajectory = np.nan
                if bs.shape[1] > 1:
                    td_1 = safe_float(bs.loc['Total Debt'].iloc[1] if 'Total Debt' in bs.index else np.nan)
                    te_1 = safe_float(bs.loc['Stockholders Equity'].iloc[1] if 'Stockholders Equity' in bs.index else np.nan)
                    de_1 = td_1 / te_1 if te_1 > 0 else np.nan
                    if pd.notna(de_0) and pd.notna(de_1):
                        # If debt decreases, DE_0 < DE_1, yielding a negative value.
                        de_trajectory = de_0 - de_1 # Lower is better

                # Robust lookup for Current Assets and Current Liabilities
                ca_keys = ['Total Current Assets', 'Current Assets', 'CurrentAssets']
                cl_keys = ['Total Current Liabilities', 'Current Liabilities', 'CurrentLiabilities']
                
                ca_row = next((bs.loc[k] for k in ca_keys if k in bs.index), None)
                cl_row = next((bs.loc[k] for k in cl_keys if k in bs.index), None)
                
                ca_0 = safe_float(ca_row.iloc[0] if ca_row is not None else np.nan)
                cl_0 = safe_float(cl_row.iloc[0] if cl_row is not None else np.nan)
                cr_0 = ca_0 / cl_0 if cl_0 > 0 else np.nan # Higher is better
                
                cr_trajectory = np.nan
                if bs.shape[1] > 1:
                    ca_1 = safe_float(ca_row.iloc[1] if ca_row is not None else np.nan)
                    cl_1 = safe_float(cl_row.iloc[1] if cl_row is not None else np.nan)
                    cr_1 = ca_1 / cl_1 if cl_1 > 0 else np.nan
                    if pd.notna(cr_0) and pd.notna(cr_1):
                        cr_trajectory = cr_0 - cr_1 # Higher is better

                # ==========================================
                # 3. SAFETY METRIC (INTEREST COVERAGE)
                # ==========================================
                int_exp = safe_float(financials.loc['Interest Expense'].iloc[0] if 'Interest Expense' in financials.index else np.nan)
                int_coverage = np.nan
                if pd.notna(ebit_0) and pd.notna(int_exp):
                    int_exp = abs(int_exp)
                    if int_exp > 0:
                        int_coverage = ebit_0 / int_exp # Higher is better

                results.append({
                    'Ticker': ticker,
                    'Sector': info.get('sector', 'Unknown'),
                    'ROA': roa_0,
                    'ROA_Trajectory': roa_trajectory,
                    'Revenue_Growth': rev_growth,
                    'Debt_to_Equity': de_0,
                    'DE_Trajectory': de_trajectory,
                    'Current_Ratio': cr_0,
                    'CR_Trajectory': cr_trajectory,
                    'Interest_Coverage': int_coverage
                })
                
            except Exception as e:
                print(f"[ERROR] Processing continuous financial metrics for {ticker}: {e}")
                continue

        df = pd.DataFrame(results)
        if df.empty:
            print("[WARNING] No valid financial metrics extracted.")
            return None
            
        # Ensure correct numeric types
        numeric_cols = ['ROA', 'ROA_Trajectory', 'Revenue_Growth', 'Debt_to_Equity', 
                        'DE_Trajectory', 'Current_Ratio', 'CR_Trajectory', 'Interest_Coverage']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    

    def rank_and_score(self, metrics_df, custom_weights=None):
        """
        Cross-Sectional Scoring System.
        """
        print("[INFO] Applying sector-neutral Z-score ranking...")
        
        default_weights = {
            'EV_UFCF': 0.15,         
            'FCF_Yield': 0.12,        
            'EV_EBIT': 0.10,         
            'Earnings_Yield': 0.05,
            'EV_EBITDA_CAPEX': 0.03,
            'EV_NOPAT': 0.00,         
            'EV_IC': 0.00,
            'EV_EBITDA': 0.00,      

            'ROA': 0.10,               # Leverage-neutral operational efficiency.
            'Interest_Coverage': 0.10, # Pure survival: debt servicing capability over half a decade.
            'Debt_to_Equity': 0.08,    # Structural balance sheet risk.
            'ROA_Trajectory': 0.06,    # Rewards improving long-term efficiency.
            'DE_Trajectory': 0.05,     # Rewards corporate deleveraging.
            'Current_Ratio': 0.04,     # Short-term liquidity buffer.
            'Revenue_Growth': 0.02,
            'CR_Trajectory': 0.00,

            # --- 3. TRUE ALPHA & MOMENTUM (8%) ---
            'Alpha_6F': 0.08,          # Persistent structural advantage.
            'Momentum_12M_1M': 0.00,   # KILLED: Momentum decays fast and triggers rapid ranking drops, forcing unwanted sales.

            # --- 4. RISK & PRICING (2%) ---
            'Beta_Mkt': 0.02,          # Slight preference for low volatility to stabilize the Ledoit-Wolf matrix downstream.
            'PER': 0.00,
            'Price_Book': 0.00
        }
        
        weights = custom_weights if custom_weights else default_weights
        
        # Group metrics by their directional logic
        # Group metrics by their directional logic
        metrics_lower_is_better = [
            'EV_UFCF', 'EV_NOPAT', 'EV_IC', 'EV_EBITDA_CAPEX', 'Beta_Mkt', 'PER', 'Price_Book', 'EV_EBITDA', 'EV_EBIT',
            'Debt_to_Equity', 'DE_Trajectory' # Added Financial Health Metrics
        ]
        
        metrics_higher_is_better = [
            'Alpha_6F', 'Momentum_12M_1M', 'ROE', 'Div_Yield', 'FCF_Yield', 'Earnings_Yield',
            'ROA', 'ROA_Trajectory', 'Revenue_Growth', 'Current_Ratio', 'CR_Trajectory', 'Interest_Coverage' # Added Financial Health Metrics
        ]
        
        scored_df = metrics_df.copy()
        grouped_sectors = scored_df.groupby('Sector')
        
        scored_df['Total_Master_Score'] = 0.0
        total_applied_weight = 0.0
        
        def calculate_winsorized_zscore(series, direction=1):
            """
            Calculates Z-Score (standard deviations from sector mean) 
            and clips extreme outliers to +/- 3 sigmas to prevent model skewing.
            """
            s = series.dropna()
            if len(s) < 2:
                return pd.Series(0, index=series.index)
            
            z = (s - s.mean()) / (s.std() + 1e-6)
            z = z.clip(lower=-3.0, upper=3.0) # Winsorization
            return z * direction
            
        # 1. Process Lower-is-Better metrics (Value & Low-Vol)
        for col in metrics_lower_is_better:
            if col in scored_df.columns:
                z_col = f"{col}_Z"
                # Direction = -1 inverts the Z-score so lower values get positive points
                scored_df[z_col] = grouped_sectors[col].transform(lambda x: calculate_winsorized_zscore(x, direction=-1))
                
                # Severe penalty (-3 standard deviations) for missing crucial financial data
                scored_df[z_col] = scored_df[z_col].fillna(-3.0) 
                
                weight = weights.get(col, 0.0)
                scored_df['Total_Master_Score'] += scored_df[z_col] * weight
                total_applied_weight += weight
                
        # 2. Process Higher-is-Better metrics (Alpha & Momentum)
        for col in metrics_higher_is_better:
            if col in scored_df.columns:
                z_col = f"{col}_Z"
                # Direction = 1 keeps standard Z-score direction
                scored_df[z_col] = grouped_sectors[col].transform(lambda x: calculate_winsorized_zscore(x, direction=1))
                
                scored_df[z_col] = scored_df[z_col].fillna(-3.0)
                
                weight = weights.get(col, 0.0)
                scored_df['Total_Master_Score'] += scored_df[z_col] * weight
                total_applied_weight += weight
                
        # Normalize the composite Z-score to account for any missing columns dynamically
        if total_applied_weight > 0:
            scored_df['Total_Master_Score'] = scored_df['Total_Master_Score'] / total_applied_weight
            
        # Optional: Convert the final composite Z-Score back into an easy-to-read 0-100 Percentile 
        # specifically for portfolio construction readability.
        scored_df['Final_Percentile_Rank'] = grouped_sectors['Total_Master_Score'].rank(pct=True) * 100
        
        
        return scored_df.sort_values(by='Total_Master_Score', ascending=False).reset_index(drop=True)

    def get_prices_in_range(self, window_start, window_end):
        """
        Slices the price database for a specific date range.
        """
        sliced_df = self.prices_db.loc[window_start:window_end]
        if sliced_df.empty:
            print("[WARNING] Sliced price dataframe is empty for the requested dates.")
        return sliced_df

def get_index_tickers(index_name="SP500"):
    """
    Fetches ticker symbols for a given market index using a custom User-Agent.
    Dynamically maps European countries to their Yahoo Finance suffixes.
    """
    print(f"Fetching ticker list for {index_name} with custom User-Agent...")
    
    index_config = {
        "SP500": {
            "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "table_match": "Symbol",
            "ticker_col": "Symbol",
            "country_col": None,
            "suffix": ""
        },
        "IBEX35": {
            "url": "https://en.wikipedia.org/wiki/IBEX_35",
            "table_match": "Ticker", 
            "ticker_col": "Ticker",
            "country_col": None,
            "suffix": ".MC"
        },
        "EUROSTOXX600": {
            "url": "https://en.wikipedia.org/wiki/STOXX_Europe_600",
            "table_match": "ICB Sector",
            "ticker_col": "Ticker",
            "country_col": "Country",
            "suffix": ""
        }
    }
    
    index_key = str(index_name).upper().replace(" ", "").replace("&", "")
    
    if index_key not in index_config:
        raise ValueError(f"[ERROR] Scraper configuration for '{index_name}' not found.")
        
    config = index_config[index_key]
    url = config["url"]
    
    custom_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        tables = pd.read_html(url, match=config["table_match"], storage_options=custom_headers)
        table = tables[0]
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed fetching data for {index_name}: {e}")
        return []

    # Dictionary to map European countries to Yahoo Finance ticker suffixes
    country_to_suffix = {
        'United Kingdom': '.L',
        'Switzerland': '.SW',
        'France': '.PA',
        'Germany': '.DE',
        'Netherlands': '.AS',
        'Spain': '.MC',
        'Italy': '.MI',
        'Sweden': '.ST',
        'Finland': '.HE',
        'Denmark': '.CO',
        'Norway': '.OL',
        'Belgium': '.BR',
        'Ireland': '.IR', 
        'Austria': '.VI',
        'Portugal': '.LS',
        'Luxembourg': '.PA'
    }

    cleaned_tickers = []
    
    for idx, row in table.iterrows():
        t = str(row[config["ticker_col"]]).strip()
        
        if pd.isna(t) or t == 'nan':
            continue
            
        # Base suffix configuration
        suffix = config["suffix"]
        
        # Dynamic suffix injection for pan-European indices
        if config.get("country_col") and config["country_col"] in table.columns:
            country = str(row[config["country_col"]]).strip()
            suffix = country_to_suffix.get(country, "")
            
        # 1. Strip Wikipedia's pre-existing suffixes to avoid duplication (e.g. SAN.MC -> SAN)
        if suffix and t.endswith(suffix):
            t = t[:-len(suffix)]
        if suffix and '.' in t:
            t = t.split('.')[0]
            
        # 2. Fix US dual-class shares format (BRK.B -> BRK-B)
        if not suffix:
            t = t.replace('.', '-').replace(' ', '-')
            
        # 3. Append the correct Yahoo suffix
        if suffix:
            t = f"{t}{suffix}"
            
        # Prevent duplicates
        if t not in cleaned_tickers:
            cleaned_tickers.append(t)
        
    print(f"Successfully scraped and cleaned {len(cleaned_tickers)} tickers for {index_name}.")
    
    return cleaned_tickers

def load_market_data(tickers, start_date=None, end_date=None):
    """
    Downloads historical prices using the unified batch function.
    Applies specific filtering for asset selection criteria.
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    import pandas as pd
    
    if start_date is None:
        start_date = datetime.now() - relativedelta(years=1)
    if end_date is None:
        end_date = datetime.now()

    # 1. Fetch unified batch data
    data = download_historical_prices_batch(tickers, start_date, end_date)

    if data.empty:
        raise ValueError("[CRITICAL ERROR] yfinance returned no data for any ticker in the list.")

    # 2. Specific filtering for MarketSelection (Minimum valid history threshold)
    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date

    days_in_period = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days
    years = max(days_in_period / 365.25, 1.0)
    
    if len(data) > 0:
        # Bypass strict history threshold if fetching isolated missing assets
        if len(tickers) <= 3:
            print("[INFO] Small batch detected. Bypassing strict history threshold to recover newer ETFs.")
            min_valid_days = 5  # Keep the asset as long as it has at least 5 valid trading days
        else:
            min_valid_days = int(252 * years * 0.75)
            
        data = data.dropna(axis=1, thresh=min_valid_days)
        
    # Backfill missing historical data for young ETFs to prevent NaN propagation
    data = data.ffill().bfill()

    print(f"[INFO] Data successfully loaded. Valid assets remaining: {data.shape[1]}")

    if data.empty or data.shape[1] == 0:
        raise ValueError("[CRITICAL ERROR] Price matrix is empty after cleaning.")

    return data

def _fetch_single_fundamental(ticker):
    """
    Helper function to fetch fundamentals for a single ticker to be cached.
    Relies entirely on yfinance's native curl_cffi session to bypass Cloudflare.
    """
    # Stagger requests to avoid triggering the rate limiter
    time.sleep(random.uniform(0.5, 1.5)) 
    try:
        # We no longer pass the 'session' parameter
        stock = yf.Ticker(ticker) 
        info = stock.info
        cf = stock.cash_flow
        fin = stock.financials
        bs = stock.balance_sheet
        
        if cf.empty or fin.empty or bs.empty:
            return ticker, None
            
        return ticker, {'info': info, 'financials': fin, 'balance_sheet': bs, 'cash_flow': cf}
    except Exception as e:
        # Muted generic exceptions to avoid console spam, but you can uncomment for debugging
        print(f"[ERROR] Failed fetching data for {ticker}: {e}")
        return ticker, None

CACHE_VERSION = "v2_gbp_pence_fix"

def load_or_update_market_cache(tickers, start_date, end_date, cache_file=os.path.join("CacheData", "Local_Market_Cache.pkl"), ff_universe="developed"):
    """
    Centralized cache manager for market prices, fundamentals, Fama-French factors,
    and required FX exchange rates. Persists data to disk to minimize network calls.
    """
    import os
    import pickle
    import yfinance as yf
    import pandas as pd
    from FinanceFunctions_lib import download_historical_prices_batch, fetch_global_fama_french_daily

    # Initialize default cache structure
    cache_data = {
        'last_date': None,
        'prices': pd.DataFrame(),
        'fundamentals': {},
        'attempted_tickers': [],
        'ff_factors': pd.DataFrame(),
        'fx_rates': {},
        'cache_version': None
    }

    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date
    cache_modified = False

    # Load existing cache from disk if available
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                loaded_cache = pickle.load(f)
                cache_data.update(loaded_cache)
                print("[INFO] Successfully loaded local market cache.")
        except Exception as e:
            print(f"[WARNING] Could not load cache file: {e}. Building new cache.")

    # Version check: if cache was built before the GBp fix, force a price refresh
    if cache_data.get('cache_version') != CACHE_VERSION:
        print(f"[INFO] Cache version mismatch (found '{cache_data.get('cache_version')}', expected '{CACHE_VERSION}'). Forcing price refresh to apply GBX->GBP correction...")
        cache_data['last_date'] = None
        cache_data['cache_version'] = CACHE_VERSION
        cache_modified = True

    # --- 1. Update Prices ---
    if cache_data['last_date'] != end_str:
        print(f"[INFO] Cache date mismatch or missing. Updating prices to {end_str}...")
        # Note: load_market_data must be defined in this file or imported
        cache_data['prices'] = load_market_data(tickers, start_date, end_date)
        cache_data['last_date'] = end_str
        cache_modified = True

    # --- 2. Update Fundamentals ---
    missing_tickers = [t for t in tickers if t not in cache_data['fundamentals'] and t not in cache_data['attempted_tickers']]
    
    if missing_tickers:
        print(f"[INFO] Downloading fundamentals for {len(missing_tickers)} new assets...")
        for ticker in missing_tickers:
            cache_data['attempted_tickers'].append(ticker)
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                # Ensure it's a valid equity/asset with basic fundamental data
                if 'sector' in info or 'previousClose' in info:
                    cache_data['fundamentals'][ticker] = {
                        'info': info,
                        'cash_flow': stock.cash_flow,
                        'financials': stock.financials,
                        'balance_sheet': stock.balance_sheet
                    }
                else:
                    cache_data['fundamentals'][ticker] = None
            except Exception as e:
                print(f"[WARNING] Failed to fetch fundamentals for {ticker}: {e}")
                cache_data['fundamentals'][ticker] = None
            
        cache_modified = True

    # --- 3. Update Fama-French Factors ---
    if cache_data['ff_factors'].empty or cache_data['last_date'] != end_str:
        print("[INFO] Updating Fama-French factors...")
        try:
            cache_data['ff_factors'] = fetch_global_fama_french_daily(start_date, end_date, universe=ff_universe)
            cache_modified = True
        except Exception as e:
            print(f"[ERROR] Could not update FF factors: {e}")

    # --- 4. Proactive FX Discovery & Download ---
    fundamentals = cache_data['fundamentals']
    required_fx_pairs = set()
    
    # Scan the fundamental data to detect currency mismatches
    for t in tickers:
        if t in fundamentals and fundamentals[t] is not None:
            info = fundamentals[t].get('info', {})
            p_curr = info.get('currency', 'USD')
            f_curr = info.get('financialCurrency', 'USD')
            
            # Standardize UK Pence to GBP
            if p_curr == 'GBp': p_curr = 'GBP'
            if f_curr == 'GBp': f_curr = 'GBP'
            
            if p_curr != f_curr:
                pair = f"{p_curr}{f_curr}=X"
                if pair not in cache_data['fx_rates']:
                    required_fx_pairs.add(pair)

    # Always ensure base EUR conversion pairs are cached (portfolio is EUR-denominated)
    for _pair in ('EURUSD=X', 'GBPEUR=X'):
        if _pair not in cache_data['fx_rates']:
            required_fx_pairs.add(_pair)

    if required_fx_pairs:
        print(f"[INFO] New currency mismatches detected. Downloading {len(required_fx_pairs)} FX pairs for cache...")
        # Extend the start date slightly to ensure we have data covering weekends/holidays around the start
        extended_start = start_date - pd.Timedelta(days=10)
        fx_data = download_historical_prices_batch(list(required_fx_pairs), extended_start, end_date)
        
        for pair in required_fx_pairs:
            if not fx_data.empty and pair in fx_data.columns:
                # Drop NAs to keep the series clean
                clean_series = fx_data[pair].dropna()
                cache_data['fx_rates'][pair] = clean_series
                cache_modified = True
            else:
                print(f"[WARNING] Could not retrieve data for FX pair {pair}")

    # --- 5. Save Cache to Disk ---
    if cache_modified:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"[INFO] Market cache successfully updated and saved to {cache_file}.")
        except Exception as e:
            print(f"[ERROR] Failed to save cache to disk: {e}")

    return cache_data['prices'], cache_data['fundamentals'], cache_data['ff_factors'], cache_data['fx_rates']

def compute_momentum_features(prices_df):
    returns = np.log(prices_df / prices_df.shift(1))

    momentum_data = []

    for ticker in prices_df.columns:
        series = prices_df[ticker].dropna()
        
        if len(series) < 252:
            continue

        try:
            # --- Indexing ---
            p_t = series.iloc[-1]
            p_1m = series.iloc[-21]
            p_6m = series.iloc[-126]
            p_12m = series.iloc[-252]

            # --- Core signals ---
            mom_12_1 = np.log(p_1m / p_12m)   # 12M excluding last month
            mom_6_1  = np.log(p_1m / p_6m)
            mom_1m   = np.log(p_t / p_1m)

            # --- Volatility ---
            vol = returns[ticker].rolling(252).std().iloc[-1]

            if pd.isna(vol) or vol == 0:
                continue

            # --- Composite momentum ---
            raw_momentum = (
                0.6 * mom_12_1 +
                0.2 * mom_6_1 -
                0.2 * mom_1m
            )

            # --- Risk-adjusted momentum ---
            momentum_score = raw_momentum / vol

            momentum_data.append({
                'Ticker': ticker,
                'Momentum_12M_1M': momentum_score
            })

        except Exception:
            continue

    return pd.DataFrame(momentum_data)