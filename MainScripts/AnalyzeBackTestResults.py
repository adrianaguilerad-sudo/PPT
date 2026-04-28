import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import os
import glob

# --- Configuration ---
search_pattern = './BackTests/Backtest_Results_*.xlsx'
list_of_files = glob.glob(search_pattern)
if not list_of_files:
    raise FileNotFoundError("No se encontraron archivos de backtest en la carpeta ./BackTests/")
file_path = max(list_of_files)
#file_path = './BackTests/Backtest_Results_20260428_163321.xlsx'
print(f"[INFO] Analizando el archivo más reciente: {file_path}")


excluded_tickers = ['TOTAL INVESTED', 'TOTAL VALUE', 'ROI (%)']

def load_and_preprocess_data(path, exclude_list):
    """Loads all sheets from Excel, cleans weights, and filters active assets."""
    excel_data = pd.ExcelFile(path)
    data_list = []
    
    for sheet in excel_data.sheet_names:
        try:
            # Parse sheet name as date
            date_val = datetime.strptime(sheet.strip(), '%Y-%m-%d')
            df = pd.read_excel(path, sheet_name=sheet)
            df['Date'] = date_val
            data_list.append(df)
        except ValueError:
            # Skip non-date sheets
            continue
            
    if not data_list:
        return pd.DataFrame()
        
    all_df = pd.concat(data_list, ignore_index=True)
    
    # Numeric cleaning
    all_df['Target_Weight_Num'] = pd.to_numeric(all_df['Target_Weight'].astype(str).str.replace('%', ''), errors='coerce').fillna(0)
    all_df['Action_Taken'] = pd.to_numeric(all_df['Action_Taken'], errors='coerce').fillna(0)
    all_df['Current_Market_Value'] = pd.to_numeric(all_df['Current_Market_Value'], errors='coerce').fillna(0)
    
    # Filter: Exclude totals
    mask = (~all_df['Ticker'].isin(exclude_list))
    return all_df[mask].copy()

def get_transaction_stats(df):
    """Calculates buys, sells, and realized profit per month."""
    df = df.sort_values(['Date', 'Ticker'])
    dates = sorted(df['Date'].unique())
    ticker_basis = {} # Tracks cost basis per ticker
    stats = []
    
    for d in dates:
        month_data = df[df['Date'] == d]
        buys = 0
        sells = 0
        profit_this_month = 0
        
        for _, row in month_data.iterrows():
            ticker = row['Ticker']
            action = row['Action_Taken']
            val_after = row['Current_Market_Value']
            val_before = val_after - action # Estimating value before rebalancing
            
            if ticker not in ticker_basis:
                ticker_basis[ticker] = 0
            
            if action > 0:
                buys += 1
                ticker_basis[ticker] += action
            elif action < 0:
                sells += 1
                if val_before > 0:
                    # Calculate proportion of position sold
                    p = min(1.0, abs(action) / val_before)
                    cost_of_sold_part = p * ticker_basis[ticker]
                    # Realized Profit = Revenue - Cost
                    profit_this_month += (abs(action) - cost_of_sold_part)
                    # Update remaining cost basis
                    ticker_basis[ticker] -= cost_of_sold_part
                else:
                    # If value before was 0 (unexpected), treat whole sale as profit/revenue
                    profit_this_month += abs(action)
                    ticker_basis[ticker] = 0
        
        stats.append({
            'Date': d,
            'Purchases': buys,
            'Sales': sells,
            'Realized_Profit': profit_this_month
        })
    return pd.DataFrame(stats)

def get_gantt_intervals(df):
    """Calculates start and end intervals for the Gantt chart."""
    records = []
    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values('Date')
        dates = group['Date'].tolist()
        if not dates: continue
        
        start_date = dates[0]
        end_date = dates[0]
        
        for i in range(1, len(dates)):
            # Check continuity (approx 1 month)
            if (dates[i] - end_date).days <= 35:
                end_date = dates[i]
            else:
                # Close interval and start new one
                records.append({'Ticker': ticker, 'Start': start_date, 'End': end_date + timedelta(days=30)})
                start_date = dates[i]
                end_date = dates[i]
        
        # Add the final interval for the ticker
        records.append({'Ticker': ticker, 'Start': start_date, 'End': end_date + timedelta(days=30)})
    return pd.DataFrame(records)

def get_streak_lengths(df):
    """Calculates consecutive holding months for the histogram."""
    streaks = []
    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values('Date')
        dates = group['Date'].tolist()
        if not dates: continue
        
        current_streak = 1
        for i in range(1, len(dates)):
            if (dates[i] - dates[i-1]).days <= 35:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        streaks.append(current_streak)
    return streaks

def get_total_profit_history(path):
    """Extracts 'TOTAL VALUE' and 'TOTAL INVESTED' to calculate profit over time."""
    excel_data = pd.ExcelFile(path)
    records = []
    
    for sheet in excel_data.sheet_names:
        try:
            date_val = datetime.strptime(sheet.strip(), '%Y-%m-%d')
            df = pd.read_excel(path, sheet_name=sheet)
            
            # Retrieve specific rows for calculations
            t_value = df[df['Ticker'] == 'TOTAL VALUE']['Current_Market_Value'].values[0]
            t_invested = df[df['Ticker'] == 'TOTAL INVESTED']['Current_Market_Value'].values[0]
            
            records.append({
                'Date': date_val,
                'Total_Value': t_value,
                'Total_Invested': t_invested,
                'Total_Profit': t_value - t_invested
            })
        except (ValueError, IndexError):
            # Skip sheets that don't match date format or missing data
            continue
            
    return pd.DataFrame(records).sort_values('Date')

# --- Main Execution ---
df_active = load_and_preprocess_data(file_path, excluded_tickers)

profit_df = get_total_profit_history(file_path)

if not profit_df.empty:
    # Generate the Profit Chart
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(profit_df['Date'], profit_df['Total_Profit'], color='darkblue', 
            marker='o', linestyle='-', linewidth=2, label='Total Profit (Value - Invested)')
    
    # Fill area for better visualization
    ax.fill_between(profit_df['Date'], profit_df['Total_Profit'], color='skyblue', alpha=0.3)

    # Labels and Title
    ax.set_title('Portfolio Total Profit Over Time', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Profit ($)', fontsize=12)
    
    # Date formatting for X-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.show()
     
    # 1. New Transaction and Profit Chart
    stats_df = get_transaction_stats(df_active)
    fig4, ax_trans = plt.subplots(figsize=(12, 6))
    
    x_indices = np.arange(len(stats_df))
    bar_width = 0.35
    
    # Bars for Buys and Sells
    ax_trans.bar(x_indices - bar_width/2, stats_df['Purchases'], bar_width, label='Purchases', color='royalblue', alpha=0.8)
    ax_trans.bar(x_indices + bar_width/2, stats_df['Sales'], bar_width, label='Sales', color='tomato', alpha=0.8)
    
    ax_trans.set_xlabel('Month')
    ax_trans.set_ylabel('Number of Transactions')
    ax_trans.set_title('Monthly Transactions and Realized Profit')
    ax_trans.set_xticks(x_indices)
    ax_trans.set_xticklabels(stats_df['Date'].dt.strftime('%Y-%m'), rotation=45)
    
    # Secondary axis for Profit
    ax_profit = ax_trans.twinx()
    ax_profit.plot(x_indices, stats_df['Realized_Profit'], color='forestgreen', marker='o', linewidth=2, label='Realized Profit')
    ax_profit.set_ylabel('Realized Profit ($)', color='forestgreen')
    ax_profit.tick_params(axis='y', labelcolor='forestgreen')
    ax_profit.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    
    # Combined Legend
    h1, l1 = ax_trans.get_legend_handles_labels()
    h2, l2 = ax_profit.get_legend_handles_labels()
    ax_trans.legend(h1 + h2, l1 + l2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

    # 1. Generate Gantt Chart
    intervals = get_gantt_intervals(df_active)
    tickers = sorted(intervals['Ticker'].unique(), reverse=True)
    y_mapping = {t: i for i, t in enumerate(tickers)}
    
    fig1, ax1 = plt.subplots(figsize=(12, max(6, len(tickers) * 0.4)))
    for _, row in intervals.iterrows():
        start, end = mdates.date2num(row['Start']), mdates.date2num(row['End'])
        ax1.barh(y_mapping[row['Ticker']], end - start, left=start, height=0.6, color='skyblue', edgecolor='navy')
        
    ax1.set_yticks(range(len(tickers)))
    ax1.set_yticklabels(tickers)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    ax1.set_title('Portfolio Company Permanence (Gantt Chart)')
    ax1.set_xlabel('Timeline')
    ax1.set_ylabel('Tickers')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 2. Generate Histogram for Streaks
    streaks = get_streak_lengths(df_active)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    max_streak = max(streaks) if streaks else 1
    bins = np.arange(1, max_streak + 2) - 0.5
    counts, _, _ = ax2.hist(streaks, bins=bins, color='seagreen', edgecolor='black', alpha=0.7, rwidth=0.8)
    
    ax2.set_xticks(range(1, max_streak + 1))
    ax2.set_title('Distribution of Consecutive Months in Portfolio')
    ax2.set_xlabel('Consecutive Months')
    ax2.set_ylabel('Number of Occurrences')
    
    for count, x in zip(counts, range(1, max_streak + 1)):
        if count > 0:
            ax2.text(x, count + 0.1, str(int(count)), ha='center', fontweight='bold', color='darkgreen')
            
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 3. Generate Percentage of Presence Since First Appearance
    all_dates = sorted(df_active['Date'].unique())
    presence_percentages = {}
    
    for ticker, group in df_active.groupby('Ticker'):
        # Total unique months the ticker was held
        actual_months = len(group['Date'].unique())
        first_appearance = group['Date'].min()
        
        # Calculate how many total backtest months have passed since it first appeared
        possible_months = len([d for d in all_dates if d >= first_appearance])
        
        if possible_months > 0:
            percentage = (actual_months / possible_months) * 100
        else:
            percentage = 0
            
        presence_percentages[ticker] = percentage

    # Sort series to display highest percentages at the top
    ticker_percentages = pd.Series(presence_percentages).sort_values(ascending=True)
    
    fig3, ax3 = plt.subplots(figsize=(12, max(6, len(ticker_percentages) * 0.4)))
    bars = ax3.barh(ticker_percentages.index, ticker_percentages.values, color='lightcoral', edgecolor='maroon')
    
    ax3.set_title('Percentage of Presence Since First Appearance', fontsize=14)
    ax3.set_xlabel('Presence Percentage (%)', fontsize=12)
    ax3.set_ylabel('Tickers', fontsize=12)
    
    # Expand X axis slightly so the 100% labels don't get cut off
    ax3.set_xlim(0, 110) 
    
    # Add value labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax3.text(width + 1.5, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', va='center', fontweight='bold', color='maroon')
                 
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 4. Generate Histogram of Presence Percentages
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    
    # Create bins for every 10% (0-10, 10-20, ..., 90-100)
    bins_presence = np.arange(0, 110, 5)
    
    counts_p, _, _ = ax5.hist(
        ticker_percentages.values, 
        bins=bins_presence, 
        color='mediumpurple', 
        edgecolor='black', 
        alpha=0.8, 
        rwidth=0.85
    )
    
    ax5.set_xticks(bins_presence)
    ax5.set_title('Distribution of Presence Percentages Since First Appearance', fontsize=14)
    ax5.set_xlabel('Presence Percentage (%)', fontsize=12)
    ax5.set_ylabel('Number of Companies', fontsize=12)
    
    # Add count labels on top of the bars
    for count, x in zip(counts_p, bins_presence[:-1]):
        if count > 0:
            ax5.text(x + 5, count + (max(counts_p)*0.02), str(int(count)), ha='center', fontweight='bold', color='indigo')
            
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# 5. Generate Portfolio ROI Over Time Plot
    
    roi_dates = []
    roi_values = []
    
    # Reload just to extract the ROI summary rows from each sheet
    excel_data_roi = pd.ExcelFile(file_path)
    for sheet in excel_data_roi.sheet_names:
        try:
            # Parse sheet name as date
            date_val = datetime.strptime(sheet.strip(), '%Y-%m-%d')
            df_sheet = pd.read_excel(file_path, sheet_name=sheet)
            
            # Locate the exact row for ROI
            roi_row = df_sheet[df_sheet['Ticker'] == 'ROI (%)']
            if not roi_row.empty:
                roi_val = float(roi_row['Current_Market_Value'].iloc[0])
                roi_dates.append(date_val)
                roi_values.append(roi_val)
        except ValueError:
            # Skip sheets that don't match the date format
            continue
            
    # Proceed only if data was successfully extracted
    if roi_dates:
        # Sort the data chronologically
        sorted_indices = np.argsort(roi_dates)
        roi_dates = np.array(roi_dates)[sorted_indices]
        roi_values = np.array(roi_values)[sorted_indices]
        
        # Initialize the plot
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        
        # Plot the ROI line
        ax6.plot(roi_dates, roi_values, marker='o', linestyle='-', color='forestgreen', linewidth=2, markersize=8)
        
        # Add value annotations above each point
        for i, val in enumerate(roi_values):
            ax6.annotate(f"{val:.2f}%", 
                         (roi_dates[i], roi_values[i]), 
                         textcoords="offset points", 
                         xytext=(0, 10), 
                         ha='center',
                         fontsize=9,
                         fontweight='bold',
                         color='darkgreen')
                         
        # Formatting titles, labels and grid
        ax6.set_title('Portfolio ROI (%) Over Time', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Date', fontsize=12)
        ax6.set_ylabel('Return on Investment (%)', fontsize=12)
        ax6.grid(True, linestyle='--', alpha=0.6)
        
        # Format the x-axis for dates properly
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    else:
        print("Warning: Could not find any ROI data in the provided Excel sheets.")