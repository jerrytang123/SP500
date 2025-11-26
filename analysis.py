import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

# Load the S&P 500 data
df = pd.read_csv('SPX_500_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Filter data to start from 1980
df = df[df['Date'] >= '1980-01-01'].reset_index(drop=True)

# Calculate daily returns
df['Daily_Return_Pct'] = df['% Gain/Loss (Close)']
df['Price_Ratio'] = (df['Daily_Return_Pct'] / 100) + 1

# Calculate 2-day consecutive returns
df['Next_Day_Return_Pct'] = df['Daily_Return_Pct'].shift(-1)
df['Two_Day_Cumulative_Return'] = ((df['Price_Ratio'] * (1 + df['Next_Day_Return_Pct']/100)) - 1) * 100

print(f"Data spans from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Total trading days: {len(df)}")

# Inflation adjustment factors - accurate data from Bureau of Labor Statistics
# $100 in 1980 equivalent value in each year (purchasing power)
# Normalized to 1980 = 100.00
inflation_data = {
    1980: 100.00, 1981: 110.25, 1982: 117.09, 1983: 120.82, 1984: 126.07, 1985: 130.50, 1986: 132.96, 
    1987: 137.80, 1988: 143.48, 1989: 150.40, 1990: 158.55, 1991: 165.23, 1992: 170.23, 1993: 175.32, 
    1994: 179.80, 1995: 184.93, 1996: 190.38, 1997: 194.67, 1998: 197.81, 1999: 202.15, 2000: 208.98, 
    2001: 214.88, 2002: 218.30, 2003: 223.25, 2004: 229.17, 2005: 236.99, 2006: 244.61, 2007: 251.58, 
    2008: 261.22, 2009: 260.28, 2010: 264.51, 2011: 272.93, 2012: 278.59, 2013: 282.50, 2014: 287.35, 
    2015: 287.69, 2016: 291.19, 2017: 297.42, 2018: 304.71, 2019: 310.24, 2020: 314.11, 2021: 328.90
}

# Add inflation adjustment to dataframe
df['Year'] = df['Date'].dt.year
df['Inflation_Adjusted_Value'] = df['Year'].map(inflation_data)
df['Investment_Amount_1980_Dollars'] = 100.0  # Always $100 in 1980 dollars
df['Investment_Amount_Current_Dollars'] = df['Investment_Amount_1980_Dollars'] * (df['Inflation_Adjusted_Value'] / 100.0)

# STRATEGY 1: EVENLY SPACED INVESTMENTS (Baseline)
def create_evenly_spaced_strategy(df, num_investments=1000):
    """Create evenly spaced investment dates"""
    total_days = len(df)
    interval = total_days // num_investments
    
    investment_indices = []
    for i in range(num_investments):
        idx = i * interval
        if idx < total_days:
            investment_indices.append(idx)
    
    # Take only first 1000 to ensure exactly 1000 investments
    investment_indices = investment_indices[:num_investments]
    
    strategy_df = df.iloc[investment_indices].copy()
    strategy_df['Strategy'] = 'Evenly_Spaced'
    strategy_df['Investment_Number'] = range(1, len(strategy_df) + 1)
    
    return strategy_df

# STRATEGY 2: INVEST AFTER BEST 2-DAY PERIODS
def create_best_days_strategy(df, num_investments=1000):
    """Find top 1000 best 2-day periods and invest the day after the period ends"""
    # Filter out rows where we can't calculate 2-day returns (last row)
    df_filtered = df[~df['Two_Day_Cumulative_Return'].isna()].copy()
    
    # Sort by 2-day cumulative return to find best 2-day periods
    best_periods = df_filtered.nlargest(num_investments, 'Two_Day_Cumulative_Return').copy()
    
    # Get the day after each 2-day period ends
    investment_data = []
    for _, row in best_periods.iterrows():
        period_end_idx = row.name + 1  # End of the 2-day period
        investment_idx = period_end_idx + 1  # Day after the 2-day period
        
        if investment_idx < len(df):  # Make sure investment day exists
            investment_data.append({
                'Date': df.loc[investment_idx, 'Date'],
                'Close': df.loc[investment_idx, 'Close'],
                'Year': df.loc[investment_idx, 'Year'],
                'Inflation_Adjusted_Value': df.loc[investment_idx, 'Inflation_Adjusted_Value'],
                'Investment_Amount_1980_Dollars': 100.0,
                'Investment_Amount_Current_Dollars': 100.0 * (df.loc[investment_idx, 'Inflation_Adjusted_Value'] / 100.0),
                'Previous_2Day_Return': row['Two_Day_Cumulative_Return'],
                'Period_Start_Date': row['Date'],
                'Strategy': 'After_Best_2Day_Periods'
            })
    
    strategy_df = pd.DataFrame(investment_data)
    strategy_df = strategy_df.sort_values('Date').reset_index(drop=True)
    strategy_df['Investment_Number'] = range(1, len(strategy_df) + 1)
    
    return strategy_df

# STRATEGY 3: INVEST AFTER WORST 2-DAY PERIODS
def create_worst_days_strategy(df, num_investments=1000):
    """Find top 1000 worst 2-day periods and invest the day after the period ends"""
    # Filter out rows where we can't calculate 2-day returns (last row)
    df_filtered = df[~df['Two_Day_Cumulative_Return'].isna()].copy()
    
    # Sort by 2-day cumulative return to find worst 2-day periods
    worst_periods = df_filtered.nsmallest(num_investments, 'Two_Day_Cumulative_Return').copy()
    
    # Get the day after each 2-day period ends
    investment_data = []
    for _, row in worst_periods.iterrows():
        period_end_idx = row.name + 1  # End of the 2-day period
        investment_idx = period_end_idx + 1  # Day after the 2-day period
        
        if investment_idx < len(df):  # Make sure investment day exists
            investment_data.append({
                'Date': df.loc[investment_idx, 'Date'],
                'Close': df.loc[investment_idx, 'Close'],
                'Year': df.loc[investment_idx, 'Year'],
                'Inflation_Adjusted_Value': df.loc[investment_idx, 'Inflation_Adjusted_Value'],
                'Investment_Amount_1980_Dollars': 100.0,
                'Investment_Amount_Current_Dollars': 100.0 * (df.loc[investment_idx, 'Inflation_Adjusted_Value'] / 100.0),
                'Previous_2Day_Return': row['Two_Day_Cumulative_Return'],
                'Period_Start_Date': row['Date'],
                'Strategy': 'After_Worst_2Day_Periods'
            })
    
    strategy_df = pd.DataFrame(investment_data)
    strategy_df = strategy_df.sort_values('Date').reset_index(drop=True)
    strategy_df['Investment_Number'] = range(1, len(strategy_df) + 1)
    
    return strategy_df

# Create all three strategies
evenly_spaced = create_evenly_spaced_strategy(df)
after_best_days = create_best_days_strategy(df)
after_worst_days = create_worst_days_strategy(df)

print(f"\n=== STRATEGY SUMMARIES ===")
print(f"Evenly Spaced: {len(evenly_spaced)} investments from {evenly_spaced['Date'].min().strftime('%Y-%m-%d')} to {evenly_spaced['Date'].max().strftime('%Y-%m-%d')}")
print(f"After Best 2-Day Periods: {len(after_best_days)} investments from {after_best_days['Date'].min().strftime('%Y-%m-%d')} to {after_best_days['Date'].max().strftime('%Y-%m-%d')}")
print(f"After Worst 2-Day Periods: {len(after_worst_days)} investments from {after_worst_days['Date'].min().strftime('%Y-%m-%d')} to {after_worst_days['Date'].max().strftime('%Y-%m-%d')}")

# PORTFOLIO CALCULATION FUNCTION
def calculate_portfolio_performance(investment_df, final_date):
    """Calculate the final portfolio value for a strategy"""
    final_value = 0
    final_shares = 0
    total_invested_1980_dollars = 0
    total_invested_current_dollars = 0
    
    # Get final price
    final_price = df[df['Date'] <= final_date]['Close'].iloc[-1]
    
    for _, investment in investment_df.iterrows():
        investment_price = investment['Close']
        investment_amount_current = investment['Investment_Amount_Current_Dollars']
        investment_amount_1980 = investment['Investment_Amount_1980_Dollars']
        
        # Calculate shares purchased
        shares_purchased = investment_amount_current / investment_price
        final_shares += shares_purchased
        
        total_invested_1980_dollars += investment_amount_1980
        total_invested_current_dollars += investment_amount_current
    
    final_value = final_shares * final_price
    
    return {
        'final_value': final_value,
        'final_shares': final_shares,
        'total_invested_1980_dollars': total_invested_1980_dollars,
        'total_invested_current_dollars': total_invested_current_dollars,
        'final_price': final_price
    }

# Calculate performance for all strategies (use the last date in our dataset)
final_date = df['Date'].max()

evenly_spaced_perf = calculate_portfolio_performance(evenly_spaced, final_date)
best_days_perf = calculate_portfolio_performance(after_best_days, final_date)
worst_days_perf = calculate_portfolio_performance(after_worst_days, final_date)

# RESULTS COMPARISON
print(f"\n" + "="*80)
print(f"INVESTMENT STRATEGY COMPARISON RESULTS")
print(f"="*80)
print(f"Final valuation date: {final_date.strftime('%Y-%m-%d')}")
print(f"Final S&P 500 price: ${evenly_spaced_perf['final_price']:.2f}")

print(f"\n--- TOTAL INVESTMENT AMOUNTS ---")
print(f"Each strategy: 1000 investments of $100 each (1980 dollars)")
print(f"Total in 1980 dollars: ${evenly_spaced_perf['total_invested_1980_dollars']:.2f}")

print(f"\nEvenly Spaced - Total invested (inflation-adjusted): ${evenly_spaced_perf['total_invested_current_dollars']:,.2f}")
print(f"After Best 2-Day Periods - Total invested (inflation-adjusted): ${best_days_perf['total_invested_current_dollars']:,.2f}")
print(f"After Worst 2-Day Periods - Total invested (inflation-adjusted): ${worst_days_perf['total_invested_current_dollars']:,.2f}")

print(f"\n--- FINAL PORTFOLIO VALUES ---")
print(f"Evenly Spaced Strategy:")
print(f"  Final value: ${evenly_spaced_perf['final_value']:,.2f}")
print(f"  Total shares: {evenly_spaced_perf['final_shares']:.2f}")
print(f"  Return on 1980 dollars: {(evenly_spaced_perf['final_value'] / evenly_spaced_perf['total_invested_1980_dollars'] - 1) * 100:.1f}%")
print(f"  Return on inflation-adjusted: {(evenly_spaced_perf['final_value'] / evenly_spaced_perf['total_invested_current_dollars'] - 1) * 100:.1f}%")

print(f"\nAfter Best 2-Day Periods Strategy:")
print(f"  Final value: ${best_days_perf['final_value']:,.2f}")
print(f"  Total shares: {best_days_perf['final_shares']:.2f}")
print(f"  Return on 1980 dollars: {(best_days_perf['final_value'] / best_days_perf['total_invested_1980_dollars'] - 1) * 100:.1f}%")
print(f"  Return on inflation-adjusted: {(best_days_perf['final_value'] / best_days_perf['total_invested_current_dollars'] - 1) * 100:.1f}%")

print(f"\nAfter Worst 2-Day Periods Strategy:")
print(f"  Final value: ${worst_days_perf['final_value']:,.2f}")
print(f"  Total shares: {worst_days_perf['final_shares']:.2f}")
print(f"  Return on 1980 dollars: {(worst_days_perf['final_value'] / worst_days_perf['total_invested_1980_dollars'] - 1) * 100:.1f}%")
print(f"  Return on inflation-adjusted: {(worst_days_perf['final_value'] / worst_days_perf['total_invested_current_dollars'] - 1) * 100:.1f}%")

# RELATIVE PERFORMANCE
print(f"\n--- RELATIVE PERFORMANCE (vs Evenly Spaced) ---")
best_vs_evenly = (best_days_perf['final_value'] / evenly_spaced_perf['final_value'] - 1) * 100
worst_vs_evenly = (worst_days_perf['final_value'] / evenly_spaced_perf['final_value'] - 1) * 100

print(f"After Best 2-Day Periods vs Evenly Spaced: {best_vs_evenly:+.1f}%")
print(f"After Worst 2-Day Periods vs Evenly Spaced: {worst_vs_evenly:+.1f}%")
print(f"After Worst 2-Day Periods vs After Best 2-Day Periods: {(worst_days_perf['final_value'] / best_days_perf['final_value'] - 1) * 100:+.1f}%")

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Investment Strategy Comparison: 1000 × $100 Investments (1980 Dollars)', fontsize=16, fontweight='bold')

# Plot 1: Investment timing
ax1 = axes[0, 0]
ax1.scatter(evenly_spaced['Date'], evenly_spaced['Close'], alpha=0.7, s=30, color='blue', label='Evenly Spaced')
ax1.scatter(after_best_days['Date'], after_best_days['Close'], alpha=0.7, s=30, color='green', label='After Best 2-Day Periods')
ax1.scatter(after_worst_days['Date'], after_worst_days['Close'], alpha=0.7, s=30, color='red', label='After Worst 2-Day Periods')
ax1.plot(df['Date'].values, df['Close'].values, alpha=0.3, color='gray', linewidth=0.5, label='S&P 500')
ax1.set_title('Investment Timing')
ax1.set_xlabel('Date')
ax1.set_ylabel('S&P 500 Price')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Previous 2-day returns for timing strategies
ax2 = axes[0, 1]
# Calculate bin edges with fixed width of 0.5%
min_return = min(after_best_days['Previous_2Day_Return'].min(), after_worst_days['Previous_2Day_Return'].min())
max_return = max(after_best_days['Previous_2Day_Return'].max(), after_worst_days['Previous_2Day_Return'].max())
bin_width = 0.5  # 0.5% bins
bins = np.arange(min_return - bin_width, max_return + bin_width, bin_width)

ax2.hist(after_best_days['Previous_2Day_Return'], bins=bins, alpha=0.7, color='green', label='After Best 2-Day Periods')
ax2.hist(after_worst_days['Previous_2Day_Return'], bins=bins, alpha=0.7, color='red', label='After Worst 2-Day Periods')
ax2.set_title('Previous 2-Day Returns (Investment Trigger)')
ax2.set_xlabel('Previous 2-Day Return (%)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Investment amounts over time (inflation-adjusted)
ax3 = axes[1, 0]
ax3.plot(evenly_spaced['Date'].values, evenly_spaced['Investment_Amount_Current_Dollars'].values, 'o-', alpha=0.7, color='blue', label='Evenly Spaced', markersize=4)
ax3.plot(after_best_days['Date'].values, after_best_days['Investment_Amount_Current_Dollars'].values, 'o-', alpha=0.7, color='green', label='After Best 2-Day Periods', markersize=4)
ax3.plot(after_worst_days['Date'].values, after_worst_days['Investment_Amount_Current_Dollars'].values, 'o-', alpha=0.7, color='red', label='After Worst 2-Day Periods', markersize=4)
ax3.set_title('Investment Amounts (Inflation-Adjusted)')
ax3.set_xlabel('Date')
ax3.set_ylabel('Investment Amount ($)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Final performance comparison
ax4 = axes[1, 1]
strategies = ['Evenly\nSpaced', 'After Best\n2-Day Periods', 'After Worst\n2-Day Periods']
final_values = [evenly_spaced_perf['final_value'], best_days_perf['final_value'], worst_days_perf['final_value']]
colors = ['blue', 'green', 'red']

bars = ax4.bar(strategies, final_values, color=colors, alpha=0.7)
ax4.set_title('Final Portfolio Values')
ax4.set_ylabel('Portfolio Value ($)')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, final_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# DETAILED ANALYSIS
print(f"\n" + "="*80)
print(f"DETAILED ANALYSIS")
print(f"="*80)

# Best and worst trigger periods - show top 50 of each for readability
df_filtered = df[~df['Two_Day_Cumulative_Return'].isna()].copy()

print(f"\n--- TOP 50 BEST 2-DAY PERIODS (of 1000 Investment Triggers) ---")
best_triggers = df_filtered.nlargest(1000, 'Two_Day_Cumulative_Return')[['Date', 'Two_Day_Cumulative_Return', 'Close']]
for i, (_, row) in enumerate(best_triggers.head(50).iterrows(), 1):
    period_end_date = df.loc[row.name + 1, 'Date'] if row.name + 1 < len(df) else row['Date']
    print(f"{i:3d}. {row['Date'].strftime('%Y-%m-%d')} to {period_end_date.strftime('%Y-%m-%d')}: +{row['Two_Day_Cumulative_Return']:.2f}% (Start Price: ${row['Close']:.2f})")

print(f"\n--- TOP 50 WORST 2-DAY PERIODS (of 1000 Investment Triggers) ---")
worst_triggers = df_filtered.nsmallest(1000, 'Two_Day_Cumulative_Return')[['Date', 'Two_Day_Cumulative_Return', 'Close']]
for i, (_, row) in enumerate(worst_triggers.head(50).iterrows(), 1):
    period_end_date = df.loc[row.name + 1, 'Date'] if row.name + 1 < len(df) else row['Date']
    print(f"{i:3d}. {row['Date'].strftime('%Y-%m-%d')} to {period_end_date.strftime('%Y-%m-%d')}: {row['Two_Day_Cumulative_Return']:.2f}% (Start Price: ${row['Close']:.2f})")

print(f"\n--- SUMMARY OF ALL 1000 TRIGGER PERIODS ---")
print(f"Best 2-Day Periods Range: +{best_triggers['Two_Day_Cumulative_Return'].max():.2f}% to +{best_triggers['Two_Day_Cumulative_Return'].min():.2f}%")
print(f"Worst 2-Day Periods Range: {worst_triggers['Two_Day_Cumulative_Return'].min():.2f}% to {worst_triggers['Two_Day_Cumulative_Return'].max():.2f}%")

# Average purchase prices
print(f"\n--- AVERAGE PURCHASE PRICES ---")
print(f"Evenly Spaced: ${evenly_spaced['Close'].mean():.2f}")
print(f"After Best 2-Day Periods: ${after_best_days['Close'].mean():.2f}")
print(f"After Worst 2-Day Periods: ${after_worst_days['Close'].mean():.2f}")

# Investment timing distribution
print(f"\n--- INVESTMENT TIMING BY DECADE ---")
decades = [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

for strategy_name, strategy_df in [('Evenly Spaced', evenly_spaced), 
                                   ('After Best 2-Day Periods', after_best_days), 
                                   ('After Worst 2-Day Periods', after_worst_days)]:
    print(f"\n{strategy_name}:")
    for i in range(len(decades)-1):
        start_year = decades[i]
        end_year = decades[i+1]
        count = len(strategy_df[(strategy_df['Year'] >= start_year) & (strategy_df['Year'] < end_year)])
        if count > 0:
            print(f"  {start_year}s: {count} investments")

print(f"\n" + "="*80)
print(f"CONCLUSION")
print(f"="*80)

winner = "Evenly Spaced"
winner_value = evenly_spaced_perf['final_value']

if best_days_perf['final_value'] > winner_value:
    winner = "After Best Days"
    winner_value = best_days_perf['final_value']

if worst_days_perf['final_value'] > winner_value:
    winner = "After Worst Days"
    winner_value = worst_days_perf['final_value']

print(f"🏆 WINNING STRATEGY: {winner}")
print(f"💰 Final Value: ${winner_value:,.2f}")

if winner == "After Worst 2-Day Periods":
    print(f"🎯 KEY INSIGHT: Buying after market crashes (worst 2-day periods) outperformed!")
    print(f"   This demonstrates the power of contrarian investing and buying the dip.")
elif winner == "After Best 2-Day Periods":
    print(f"🎯 KEY INSIGHT: Buying after strong market periods (momentum) outperformed!")
    print(f"   This suggests momentum effects in the market.")
else:
    print(f"🎯 KEY INSIGHT: Regular dollar-cost averaging outperformed market timing!")
    print(f"   This supports the efficient market hypothesis and consistent investing.")

print(f"\n📊 Performance spreads:")
print(f"   Highest to Lowest: ${max(final_values) - min(final_values):,.2f} difference")
print(f"   Best vs Worst: {(max(final_values) / min(final_values) - 1) * 100:.1f}% outperformance") 