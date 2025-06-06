import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from scipy import stats

# Load the S&P 500 data
df = pd.read_csv('SPX_500_Data.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date to ensure chronological order
df = df.sort_values('Date')

# Create the plot
plt.figure(figsize=(12, 8))

# Convert to numpy arrays to avoid pandas/matplotlib compatibility issues
dates = df['Date'].values
prices = df['Close'].values

# Plot closing price with log scale on y-axis
plt.plot(dates, prices, linewidth=0.8, color='blue', alpha=0.8)
plt.yscale('log')

plt.title('S&P 500 Closing Price (Log Scale)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price (USD) - Log Scale', fontsize=12)
plt.grid(True, alpha=0.3)

# Format x-axis to show years nicely
years = mdates.YearLocator(10)  # Every 10 years
years_fmt = mdates.DateFormatter('%Y')
plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(years_fmt)
plt.xticks(rotation=45)

# Add some styling
plt.tight_layout()

# Show basic statistics
print(f"Data range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Number of data points: {len(df)}")
print(f"Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
print(f"Total return: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.1f}%")

plt.show()

# Calculate daily returns
df['Daily_Return'] = df['Close'] / df['Close'].shift(1)
df['Next_Day_Return'] = df['Daily_Return'].shift(-1)

# Remove NaN values (first and last rows)
returns_df = df[['Daily_Return', 'Next_Day_Return']].dropna()

# Create scatter plot for correlation analysis
plt.figure(figsize=(10, 8))

# Sample data for better visualization if dataset is too large
sample_size = min(5000, len(returns_df))  # Use max 5000 points for cleaner visualization
sampled_data = returns_df.sample(n=sample_size, random_state=42)

plt.scatter(sampled_data['Daily_Return'], sampled_data['Next_Day_Return'], 
           alpha=0.5, s=20, color='darkblue')

plt.xlabel('Current Day Return (Price Ratio)', fontsize=12)
plt.ylabel('Next Day Return (Price Ratio)', fontsize=12)
plt.title('Correlation: Current Day vs Next Day Returns', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add reference lines at 1.0 (no change)
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No change (1.0)')
plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)

# Calculate and display correlation
correlation = returns_df['Daily_Return'].corr(returns_df['Next_Day_Return'])
slope, intercept, r_value, p_value, std_err = stats.linregress(
    returns_df['Daily_Return'], returns_df['Next_Day_Return']
)

# Add trend line
x_trend = np.linspace(returns_df['Daily_Return'].min(), returns_df['Daily_Return'].max(), 100)
y_trend = slope * x_trend + intercept
plt.plot(x_trend, y_trend, color='red', linewidth=2, alpha=0.8, label=f'Trend line (r={correlation:.4f})')

plt.legend()
plt.tight_layout()

# Print correlation statistics
print(f"\n=== Daily Return Correlation Analysis ===")
print(f"Correlation coefficient: {correlation:.6f}")
print(f"R-squared: {r_value**2:.6f}")
print(f"P-value: {p_value:.2e}")
print(f"Sample size used for correlation: {len(returns_df)}")

# Interpret the correlation
if abs(correlation) < 0.1:
    interpretation = "Very weak to no correlation"
elif abs(correlation) < 0.3:
    interpretation = "Weak correlation"
elif abs(correlation) < 0.5:
    interpretation = "Moderate correlation"
elif abs(correlation) < 0.7:
    interpretation = "Strong correlation"
else:
    interpretation = "Very strong correlation"

print(f"Interpretation: {interpretation}")

# Additional statistics
print(f"\n=== Daily Return Statistics ===")
print(f"Mean daily return: {returns_df['Daily_Return'].mean():.6f}")
print(f"Standard deviation: {returns_df['Daily_Return'].std():.6f}")
print(f"Min daily return: {returns_df['Daily_Return'].min():.6f}")
print(f"Max daily return: {returns_df['Daily_Return'].max():.6f}")

plt.show()

# === GENERALIZED STREAK ANALYSIS FUNCTION ===

def analyze_n_day_streaks(df, n_days):
    """
    Analyze n-day winning and losing streaks
    """
    # Reset the dataframe for streak analysis
    df_streak = df.copy()
    df_streak['Daily_Return'] = df_streak['Close'] / df_streak['Close'].shift(1)
    df_streak['Win'] = df_streak['Daily_Return'] > 1.0
    df_streak['Loss'] = df_streak['Daily_Return'] < 1.0
    
    # Create columns for checking consecutive wins/losses
    for i in range(1, n_days):
        df_streak[f'Win_{i}'] = df_streak['Win'].shift(i)
        df_streak[f'Loss_{i}'] = df_streak['Loss'].shift(i)
    
    # Check for n-day winning streak (all n days are wins)
    win_conditions = [df_streak['Win']] + [df_streak[f'Win_{i}'] for i in range(1, n_days)]
    df_streak[f'{n_days}_Day_Win_Streak'] = win_conditions[0]
    for condition in win_conditions[1:]:
        df_streak[f'{n_days}_Day_Win_Streak'] = df_streak[f'{n_days}_Day_Win_Streak'] & condition
    
    # Check for n-day losing streak (all n days are losses)
    loss_conditions = [df_streak['Loss']] + [df_streak[f'Loss_{i}'] for i in range(1, n_days)]
    df_streak[f'{n_days}_Day_Loss_Streak'] = loss_conditions[0]
    for condition in loss_conditions[1:]:
        df_streak[f'{n_days}_Day_Loss_Streak'] = df_streak[f'{n_days}_Day_Loss_Streak'] & condition
    
    # Calculate n-day total return for streaks
    df_streak[f'{n_days}_Day_Total_Return'] = df_streak['Daily_Return']
    for i in range(1, n_days):
        df_streak[f'{n_days}_Day_Total_Return'] *= df_streak['Daily_Return'].shift(i)
    
    # Get next day return for analysis
    df_streak['Next_Day_Return'] = df_streak['Daily_Return'].shift(-1)
    
    # Filter for n-day winning streaks
    win_streaks = df_streak[df_streak[f'{n_days}_Day_Win_Streak'] == True].copy()
    win_streaks = win_streaks[[f'{n_days}_Day_Total_Return', 'Next_Day_Return']].dropna()
    
    # Filter for n-day losing streaks
    loss_streaks = df_streak[df_streak[f'{n_days}_Day_Loss_Streak'] == True].copy()
    loss_streaks = loss_streaks[[f'{n_days}_Day_Total_Return', 'Next_Day_Return']].dropna()
    
    return win_streaks, loss_streaks

# Analyze multiple streak lengths
streak_lengths = [2, 3, 4, 5]
all_results = {}

for n in streak_lengths:
    win_streaks, loss_streaks = analyze_n_day_streaks(df, n)
    all_results[n] = {'win': win_streaks, 'loss': loss_streaks}
    
    print(f"\n=== {n}-Day Streak Analysis ===")
    print(f"Number of {n}-day winning streaks: {len(win_streaks)}")
    print(f"Number of {n}-day losing streaks: {len(loss_streaks)}")
    
    if len(win_streaks) > 0:
        win_corr = win_streaks[f'{n}_Day_Total_Return'].corr(win_streaks['Next_Day_Return'])
        print(f"Winning streak correlation: {win_corr:.6f}")
        print(f"Average {n}-day winning return: {win_streaks[f'{n}_Day_Total_Return'].mean():.6f}")
        print(f"Average next day return after winning: {win_streaks['Next_Day_Return'].mean():.6f}")
    
    if len(loss_streaks) > 0:
        loss_corr = loss_streaks[f'{n}_Day_Total_Return'].corr(loss_streaks['Next_Day_Return'])
        print(f"Losing streak correlation: {loss_corr:.6f}")
        print(f"Average {n}-day losing return: {loss_streaks[f'{n}_Day_Total_Return'].mean():.6f}")
        print(f"Average next day return after losing: {loss_streaks['Next_Day_Return'].mean():.6f}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Streak Analysis: Correlation with Next Day Returns', fontsize=16, fontweight='bold')

for i, n in enumerate(streak_lengths):
    win_streaks = all_results[n]['win']
    loss_streaks = all_results[n]['loss']
    
    # Winning streaks subplot
    ax_win = axes[0, i]
    if len(win_streaks) > 0:
        ax_win.scatter(win_streaks[f'{n}_Day_Total_Return'], win_streaks['Next_Day_Return'], 
                      alpha=0.6, s=20, color='green')
        win_corr = win_streaks[f'{n}_Day_Total_Return'].corr(win_streaks['Next_Day_Return'])
        
        # Add trend line
        if len(win_streaks) > 1:
            slope, intercept, _, _, _ = stats.linregress(
                win_streaks[f'{n}_Day_Total_Return'], win_streaks['Next_Day_Return']
            )
            x_trend = np.linspace(win_streaks[f'{n}_Day_Total_Return'].min(), 
                                 win_streaks[f'{n}_Day_Total_Return'].max(), 100)
            y_trend = slope * x_trend + intercept
            ax_win.plot(x_trend, y_trend, color='red', linewidth=2, alpha=0.8)
        
        ax_win.set_title(f'{n}-Day Winning Streaks\n(r={win_corr:.4f}, n={len(win_streaks)})', fontsize=10)
    else:
        ax_win.set_title(f'{n}-Day Winning Streaks\n(No data)', fontsize=10)
    
    ax_win.set_xlabel(f'{n}-Day Total Return', fontsize=10)
    ax_win.set_ylabel('Next Day Return', fontsize=10)
    ax_win.grid(True, alpha=0.3)
    ax_win.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax_win.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    
    # Losing streaks subplot
    ax_loss = axes[1, i]
    if len(loss_streaks) > 0:
        ax_loss.scatter(loss_streaks[f'{n}_Day_Total_Return'], loss_streaks['Next_Day_Return'], 
                       alpha=0.6, s=20, color='red')
        loss_corr = loss_streaks[f'{n}_Day_Total_Return'].corr(loss_streaks['Next_Day_Return'])
        
        # Add trend line
        if len(loss_streaks) > 1:
            slope, intercept, _, _, _ = stats.linregress(
                loss_streaks[f'{n}_Day_Total_Return'], loss_streaks['Next_Day_Return']
            )
            x_trend = np.linspace(loss_streaks[f'{n}_Day_Total_Return'].min(), 
                                 loss_streaks[f'{n}_Day_Total_Return'].max(), 100)
            y_trend = slope * x_trend + intercept
            ax_loss.plot(x_trend, y_trend, color='darkred', linewidth=2, alpha=0.8)
        
        ax_loss.set_title(f'{n}-Day Losing Streaks\n(r={loss_corr:.4f}, n={len(loss_streaks)})', fontsize=10)
    else:
        ax_loss.set_title(f'{n}-Day Losing Streaks\n(No data)', fontsize=10)
    
    ax_loss.set_xlabel(f'{n}-Day Total Return', fontsize=10)
    ax_loss.set_ylabel('Next Day Return', fontsize=10)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax_loss.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()

# Summary table of correlations
print(f"\n=== SUMMARY: Correlation Coefficients ===")
print(f"{'Streak Length':<15}{'Winning Corr':<15}{'Losing Corr':<15}{'Win Count':<12}{'Loss Count'}")
print(f"{'-'*70}")

for n in streak_lengths:
    win_streaks = all_results[n]['win']
    loss_streaks = all_results[n]['loss']
    
    if len(win_streaks) > 0:
        win_corr = win_streaks[f'{n}_Day_Total_Return'].corr(win_streaks['Next_Day_Return'])
    else:
        win_corr = np.nan
    
    if len(loss_streaks) > 0:
        loss_corr = loss_streaks[f'{n}_Day_Total_Return'].corr(loss_streaks['Next_Day_Return'])
    else:
        loss_corr = np.nan
    
    print(f"{n}-day{'':<10}{win_corr:<15.6f}{loss_corr:<15.6f}{len(win_streaks):<12}{len(loss_streaks)}")

plt.show()

# === NEW ANALYSIS: Average Return vs Streak Total Return with Standard Deviation ===

def create_binned_analysis(data, n_bins=10, min_samples=5):
    """
    Create binned analysis of streak returns vs next day returns
    """
    if len(data) < min_samples:
        return None, None, None, None
    
    # Create bins based on streak total return
    streak_col = [col for col in data.columns if 'Day_Total_Return' in col][0]
    data_sorted = data.sort_values(streak_col)
    
    # Create approximately equal-sized bins
    bin_size = len(data_sorted) // n_bins
    if bin_size < min_samples:
        n_bins = len(data_sorted) // min_samples
        bin_size = len(data_sorted) // n_bins
    
    bin_centers = []
    avg_returns = []
    std_returns = []
    bin_counts = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else len(data_sorted)
        
        bin_data = data_sorted.iloc[start_idx:end_idx]
        
        if len(bin_data) >= min_samples:
            bin_center = bin_data[streak_col].mean()
            avg_return = bin_data['Next_Day_Return'].mean()
            std_return = bin_data['Next_Day_Return'].std()
            
            bin_centers.append(bin_center)
            avg_returns.append(avg_return)
            std_returns.append(std_return)
            bin_counts.append(len(bin_data))
    
    return bin_centers, avg_returns, std_returns, bin_counts

# Create the new analysis plots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Average Next-Day Return vs Streak Magnitude (with Standard Deviation)', fontsize=16, fontweight='bold')

for i, n in enumerate(streak_lengths):
    win_streaks = all_results[n]['win']
    loss_streaks = all_results[n]['loss']
    
    # Winning streaks analysis
    ax_win = axes[0, i]
    if len(win_streaks) > 10:  # Need sufficient data
        bin_centers, avg_returns, std_returns, bin_counts = create_binned_analysis(win_streaks)
        
        if bin_centers:
            ax_win.errorbar(bin_centers, avg_returns, yerr=std_returns, 
                           fmt='o-', color='green', capsize=5, capthick=2, 
                           markersize=6, linewidth=2, alpha=0.8)
            
            # Add sample size annotations
            for bc, ar, count in zip(bin_centers, avg_returns, bin_counts):
                ax_win.annotate(f'n={count}', (bc, ar), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=8, alpha=0.7)
    
    ax_win.set_title(f'{n}-Day Winning Streaks\n(n={len(win_streaks)})', fontsize=11, fontweight='bold')
    ax_win.set_xlabel(f'{n}-Day Total Return', fontsize=10)
    ax_win.set_ylabel('Average Next Day Return', fontsize=10)
    ax_win.grid(True, alpha=0.3)
    ax_win.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No change')
    if len(win_streaks) > 0:
        ax_win.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    
    # Losing streaks analysis
    ax_loss = axes[1, i]
    if len(loss_streaks) > 10:  # Need sufficient data
        bin_centers, avg_returns, std_returns, bin_counts = create_binned_analysis(loss_streaks)
        
        if bin_centers:
            ax_loss.errorbar(bin_centers, avg_returns, yerr=std_returns, 
                            fmt='o-', color='red', capsize=5, capthick=2, 
                            markersize=6, linewidth=2, alpha=0.8)
            
            # Add sample size annotations
            for bc, ar, count in zip(bin_centers, avg_returns, bin_counts):
                ax_loss.annotate(f'n={count}', (bc, ar), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=8, alpha=0.7)
    
    ax_loss.set_title(f'{n}-Day Losing Streaks\n(n={len(loss_streaks)})', fontsize=11, fontweight='bold')
    ax_loss.set_xlabel(f'{n}-Day Total Return', fontsize=10)
    ax_loss.set_ylabel('Average Next Day Return', fontsize=10)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No change')
    if len(loss_streaks) > 0:
        ax_loss.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()

# Print detailed binned analysis
print(f"\n=== BINNED ANALYSIS SUMMARY ===")
for n in streak_lengths:
    win_streaks = all_results[n]['win']
    loss_streaks = all_results[n]['loss']
    
    print(f"\n{n}-Day Streaks:")
    
    # Winning streaks binned analysis
    if len(win_streaks) > 10:
        bin_centers, avg_returns, std_returns, bin_counts = create_binned_analysis(win_streaks)
        if bin_centers:
            print(f"  Winning Streaks (binned):")
            for bc, ar, sr, count in zip(bin_centers, avg_returns, std_returns, bin_counts):
                print(f"    Streak Return {bc:.4f}: Next Day Avg={ar:.6f} ±{sr:.6f} (n={count})")
    
    # Losing streaks binned analysis
    if len(loss_streaks) > 10:
        bin_centers, avg_returns, std_returns, bin_counts = create_binned_analysis(loss_streaks)
        if bin_centers:
            print(f"  Losing Streaks (binned):")
            for bc, ar, sr, count in zip(bin_centers, avg_returns, std_returns, bin_counts):
                print(f"    Streak Return {bc:.4f}: Next Day Avg={ar:.6f} ±{sr:.6f} (n={count})")

plt.show()

# === TRADING STRATEGY BACKTEST ===

def create_trading_strategy(df):
    """
    Create trading strategy based on streak analysis insights
    
    Strategy Rules:
    1. BUY SIGNAL: After 3-4 day losing streak with total return < 0.92 (8%+ loss)
    2. SELL SIGNAL: After 4-5 day winning streak with total return > 1.07 (7%+ gain)
    3. HOLD: No clear signal
    """
    
    strategy_df = df.copy()
    strategy_df['Daily_Return'] = strategy_df['Close'] / strategy_df['Close'].shift(1)
    strategy_df['Win'] = strategy_df['Daily_Return'] > 1.0
    strategy_df['Loss'] = strategy_df['Daily_Return'] < 1.0
    
    # Calculate rolling streak indicators
    for n in [3, 4, 5]:
        # Create win/loss indicators for past n days
        win_conditions = [strategy_df['Win'].shift(i) for i in range(n)]
        loss_conditions = [strategy_df['Loss'].shift(i) for i in range(n)]
        
        # N-day winning/losing streaks
        strategy_df[f'{n}_day_win_streak'] = win_conditions[0]
        strategy_df[f'{n}_day_loss_streak'] = loss_conditions[0]
        
        for condition in win_conditions[1:]:
            strategy_df[f'{n}_day_win_streak'] = strategy_df[f'{n}_day_win_streak'] & condition
        for condition in loss_conditions[1:]:
            strategy_df[f'{n}_day_loss_streak'] = strategy_df[f'{n}_day_loss_streak'] & condition
        
        # Calculate total return for the streak
        strategy_df[f'{n}_day_total_return'] = strategy_df['Daily_Return']
        for i in range(1, n):
            strategy_df[f'{n}_day_total_return'] *= strategy_df['Daily_Return'].shift(i)
    
    # Generate trading signals
    strategy_df['Signal'] = 'HOLD'  # Default
    
    # BUY SIGNALS: After severe losing streaks
    buy_condition = (
        ((strategy_df['3_day_loss_streak']) & (strategy_df['3_day_total_return'] < 0.92)) |
        ((strategy_df['4_day_loss_streak']) & (strategy_df['4_day_total_return'] < 0.92))
    )
    
    # SELL SIGNALS: After strong winning streaks  
    sell_condition = (
        ((strategy_df['4_day_win_streak']) & (strategy_df['4_day_total_return'] > 1.07)) |
        ((strategy_df['5_day_win_streak']) & (strategy_df['5_day_total_return'] > 1.07))
    )
    
    strategy_df.loc[buy_condition, 'Signal'] = 'BUY'
    strategy_df.loc[sell_condition, 'Signal'] = 'SELL'
    
    return strategy_df

def backtest_strategy(strategy_df, initial_capital=100):
    """
    Backtest the trading strategy
    """
    
    # Initialize portfolio
    cash = initial_capital
    shares = 0
    portfolio_value = []
    position = []  # Track if we're in the market or cash
    signals_executed = []
    
    # Track buy and hold for comparison
    buy_hold_shares = initial_capital / strategy_df['Close'].iloc[0]
    buy_hold_value = []
    
    current_position = 'CASH'  # Start in cash
    
    for i, row in strategy_df.iterrows():
        current_price = row['Close']
        signal = row['Signal']
        
        # Execute buy and hold strategy
        buy_hold_value.append(buy_hold_shares * current_price)
        
        # Execute trading strategy
        if signal == 'BUY' and current_position == 'CASH':
            # Buy with all cash
            shares = cash / current_price
            cash = 0
            current_position = 'INVESTED'
            signals_executed.append((row['Date'], 'BUY', current_price))
            
        elif signal == 'SELL' and current_position == 'INVESTED':
            # Sell all shares
            cash = shares * current_price
            shares = 0
            current_position = 'CASH'
            signals_executed.append((row['Date'], 'SELL', current_price))
        
        # Calculate current portfolio value
        if current_position == 'INVESTED':
            portfolio_value.append(shares * current_price)
            position.append('INVESTED')
        else:
            portfolio_value.append(cash)
            position.append('CASH')
    
    strategy_df['Portfolio_Value'] = portfolio_value
    strategy_df['Position'] = position
    strategy_df['Buy_Hold_Value'] = buy_hold_value
    
    return strategy_df, signals_executed

# Create and backtest the strategy
print(f"\n=== TRADING STRATEGY BACKTEST ===")
strategy_df = create_trading_strategy(df)
backtest_df, signals = backtest_strategy(strategy_df)

# Calculate performance metrics
final_strategy_value = backtest_df['Portfolio_Value'].iloc[-1]
final_buy_hold_value = backtest_df['Buy_Hold_Value'].iloc[-1]

strategy_return = (final_strategy_value / 100 - 1) * 100
buy_hold_return = (final_buy_hold_value / 100 - 1) * 100

print(f"Initial Capital: $100.00")
print(f"Final Strategy Value: ${final_strategy_value:.2f}")
print(f"Final Buy & Hold Value: ${final_buy_hold_value:.2f}")
print(f"Strategy Return: {strategy_return:.1f}%")
print(f"Buy & Hold Return: {buy_hold_return:.1f}%")
print(f"Outperformance: {strategy_return - buy_hold_return:.1f} percentage points")
print(f"Number of trades: {len(signals)}")

# Print trade details
print(f"\n=== TRADE EXECUTION DETAILS ===")
for date, action, price in signals[:10]:  # Show first 10 trades
    print(f"{date.strftime('%Y-%m-%d')}: {action} at ${price:.2f}")
if len(signals) > 10:
    print(f"... and {len(signals) - 10} more trades")

# Calculate additional metrics
strategy_returns = backtest_df['Portfolio_Value'].pct_change().dropna()
buy_hold_returns = backtest_df['Buy_Hold_Value'].pct_change().dropna()

strategy_volatility = strategy_returns.std() * np.sqrt(252) * 100  # Annualized
buy_hold_volatility = buy_hold_returns.std() * np.sqrt(252) * 100

print(f"\n=== RISK METRICS ===")
print(f"Strategy Volatility (annualized): {strategy_volatility:.1f}%")
print(f"Buy & Hold Volatility (annualized): {buy_hold_volatility:.1f}%")

# Calculate Sharpe ratios (assuming 0% risk-free rate)
strategy_sharpe = (strategy_return / (len(backtest_df) / 252)) / (strategy_volatility / 100)
buy_hold_sharpe = (buy_hold_return / (len(backtest_df) / 252)) / (buy_hold_volatility / 100)

print(f"Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"Buy & Hold Sharpe Ratio: {buy_hold_sharpe:.2f}")

# Plot strategy performance comparison
plt.figure(figsize=(15, 10))

# Main performance plot
plt.subplot(2, 1, 1)

# Convert to numpy arrays to fix pandas/matplotlib compatibility
dates_array = backtest_df['Date'].values
portfolio_values = backtest_df['Portfolio_Value'].values
buy_hold_values = backtest_df['Buy_Hold_Value'].values

plt.plot(dates_array, portfolio_values, 
         label='Trading Strategy', linewidth=2, color='blue')
plt.plot(dates_array, buy_hold_values, 
         label='Buy & Hold', linewidth=2, color='red', alpha=0.8)

# Mark buy/sell signals
buy_signals = [s for s in signals if s[1] == 'BUY']
sell_signals = [s for s in signals if s[1] == 'SELL']

if buy_signals:
    buy_dates, _, buy_prices = zip(*buy_signals)
    # Get portfolio values at buy dates
    buy_values = [backtest_df[backtest_df['Date'] == date]['Portfolio_Value'].iloc[0] 
                  for date in buy_dates]
    plt.scatter(buy_dates, buy_values, color='green', marker='^', s=100, 
               label=f'Buy Signals ({len(buy_signals)})', zorder=5)

if sell_signals:
    sell_dates, _, sell_prices = zip(*sell_signals)
    # Get portfolio values at sell dates
    sell_values = [backtest_df[backtest_df['Date'] == date]['Portfolio_Value'].iloc[0] 
                   for date in sell_dates]
    plt.scatter(sell_dates, sell_values, color='red', marker='v', s=100, 
               label=f'Sell Signals ({len(sell_signals)})', zorder=5)

plt.title('Trading Strategy vs Buy & Hold Performance', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Position tracking subplot
plt.subplot(2, 1, 2)
position_numeric = [1 if pos == 'INVESTED' else 0 for pos in backtest_df['Position']]
plt.fill_between(dates_array, 0, position_numeric, alpha=0.3, color='blue')
plt.title('Position Over Time (1 = Invested, 0 = Cash)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Position', fontsize=12)
plt.ylim(-0.1, 1.1)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate maximum drawdown
def calculate_max_drawdown(values):
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return drawdown.min() * 100

strategy_max_dd = calculate_max_drawdown(backtest_df['Portfolio_Value'])
buy_hold_max_dd = calculate_max_drawdown(backtest_df['Buy_Hold_Value'])

print(f"\n=== DRAWDOWN ANALYSIS ===")
print(f"Strategy Maximum Drawdown: {strategy_max_dd:.1f}%")
print(f"Buy & Hold Maximum Drawdown: {buy_hold_max_dd:.1f}%")

# Summary
print(f"\n=== STRATEGY SUMMARY ===")
print(f"{'Metric':<25}{'Strategy':<15}{'Buy & Hold':<15}{'Difference'}")
print(f"{'-'*65}")
print(f"{'Total Return':<25}{strategy_return:<14.1f}%{buy_hold_return:<14.1f}%{strategy_return - buy_hold_return:>+7.1f}pp")
print(f"{'Volatility':<25}{strategy_volatility:<14.1f}%{buy_hold_volatility:<14.1f}%{strategy_volatility - buy_hold_volatility:>+7.1f}pp")
print(f"{'Sharpe Ratio':<25}{strategy_sharpe:<14.2f}{buy_hold_sharpe:<14.2f}{strategy_sharpe - buy_hold_sharpe:>+7.2f}")
print(f"{'Max Drawdown':<25}{strategy_max_dd:<14.1f}%{buy_hold_max_dd:<14.1f}%{strategy_max_dd - buy_hold_max_dd:>+7.1f}pp")

# === IMPROVED STRATEGY: Tactical Allocation with Mean Reversion ===

def create_improved_strategy(df):
    """
    Improved strategy: Start with buy-and-hold base but use mean reversion for tactical allocation
    
    Strategy Rules:
    1. Base allocation: Always stay 70-100% invested (never go to cash completely)
    2. INCREASE to 100%: After 3-4 day losing streak with total return < 0.92 (8%+ loss)
    3. REDUCE to 70%: After 4-5 day winning streak with total return > 1.08 (8%+ gain)
    4. This preserves long-term growth while capturing mean reversion opportunities
    """
    
    strategy_df = df.copy()
    strategy_df['Daily_Return'] = strategy_df['Close'] / strategy_df['Close'].shift(1)
    strategy_df['Win'] = strategy_df['Daily_Return'] > 1.0
    strategy_df['Loss'] = strategy_df['Daily_Return'] < 1.0
    
    # Calculate rolling streak indicators
    for n in [3, 4, 5]:
        # Create win/loss indicators for past n days
        win_conditions = [strategy_df['Win'].shift(i) for i in range(n)]
        loss_conditions = [strategy_df['Loss'].shift(i) for i in range(n)]
        
        # N-day winning/losing streaks
        strategy_df[f'{n}_day_win_streak'] = win_conditions[0]
        strategy_df[f'{n}_day_loss_streak'] = loss_conditions[0]
        
        for condition in win_conditions[1:]:
            strategy_df[f'{n}_day_win_streak'] = strategy_df[f'{n}_day_win_streak'] & condition
        for condition in loss_conditions[1:]:
            strategy_df[f'{n}_day_loss_streak'] = strategy_df[f'{n}_day_loss_streak'] & condition
        
        # Calculate total return for the streak
        strategy_df[f'{n}_day_total_return'] = strategy_df['Daily_Return']
        for i in range(1, n):
            strategy_df[f'{n}_day_total_return'] *= strategy_df['Daily_Return'].shift(i)
    
    # Generate allocation signals (70% or 100% invested)
    strategy_df['Allocation'] = 0.85  # Default allocation (85%)
    
    # INCREASE ALLOCATION to 100%: After severe losing streaks
    increase_condition = (
        ((strategy_df['3_day_loss_streak']) & (strategy_df['3_day_total_return'] < 0.92)) |
        ((strategy_df['4_day_loss_streak']) & (strategy_df['4_day_total_return'] < 0.90))
    )
    
    # REDUCE ALLOCATION to 70%: After strong winning streaks  
    reduce_condition = (
        ((strategy_df['4_day_win_streak']) & (strategy_df['4_day_total_return'] > 1.08)) |
        ((strategy_df['5_day_win_streak']) & (strategy_df['5_day_total_return'] > 1.06))
    )
    
    strategy_df.loc[increase_condition, 'Allocation'] = 1.0  # 100% invested
    strategy_df.loc[reduce_condition, 'Allocation'] = 0.7   # 70% invested
    
    return strategy_df

def backtest_improved_strategy(strategy_df, initial_capital=100):
    """
    Backtest the improved tactical allocation strategy
    """
    
    # Initialize portfolio
    cash = initial_capital * 0.15  # Start with 85% invested
    shares = (initial_capital * 0.85) / strategy_df['Close'].iloc[0]
    portfolio_value = []
    allocation_changes = []
    
    # Track buy and hold for comparison
    buy_hold_shares = initial_capital / strategy_df['Close'].iloc[0]
    buy_hold_value = []
    
    current_allocation = 0.85
    
    for i, row in strategy_df.iterrows():
        current_price = row['Close']
        target_allocation = row['Allocation']
        
        # Execute buy and hold strategy
        buy_hold_value.append(buy_hold_shares * current_price)
        
        # Execute tactical allocation strategy
        if target_allocation != current_allocation:
            # Rebalance portfolio
            total_value = cash + shares * current_price
            target_invested = total_value * target_allocation
            current_invested = shares * current_price
            
            if target_invested > current_invested:
                # Need to buy more shares
                cash_to_invest = target_invested - current_invested
                if cash >= cash_to_invest:
                    new_shares = cash_to_invest / current_price
                    shares += new_shares
                    cash -= cash_to_invest
                    allocation_changes.append((row['Date'], f'INCREASE to {target_allocation*100:.0f}%', current_price))
            else:
                # Need to sell some shares
                value_to_sell = current_invested - target_invested
                shares_to_sell = value_to_sell / current_price
                if shares >= shares_to_sell:
                    shares -= shares_to_sell
                    cash += value_to_sell
                    allocation_changes.append((row['Date'], f'REDUCE to {target_allocation*100:.0f}%', current_price))
            
            current_allocation = target_allocation
        
        # Calculate current portfolio value
        portfolio_value.append(cash + shares * current_price)
    
    strategy_df['Portfolio_Value_Improved'] = portfolio_value
    strategy_df['Buy_Hold_Value'] = buy_hold_value
    
    return strategy_df, allocation_changes

# Run improved strategy
print(f"\n=== IMPROVED TACTICAL ALLOCATION STRATEGY ===")
improved_strategy_df = create_improved_strategy(df)
improved_backtest_df, improved_signals = backtest_improved_strategy(improved_strategy_df)

# Calculate performance metrics for improved strategy
final_improved_value = improved_backtest_df['Portfolio_Value_Improved'].iloc[-1]
final_buy_hold_value = improved_backtest_df['Buy_Hold_Value'].iloc[-1]

improved_return = (final_improved_value / 100 - 1) * 100
buy_hold_return = (final_buy_hold_value / 100 - 1) * 100

print(f"Initial Capital: $100.00")
print(f"Final Improved Strategy Value: ${final_improved_value:.2f}")
print(f"Final Buy & Hold Value: ${final_buy_hold_value:.2f}")
print(f"Improved Strategy Return: {improved_return:.1f}%")
print(f"Buy & Hold Return: {buy_hold_return:.1f}%")
print(f"Outperformance: {improved_return - buy_hold_return:.1f} percentage points")
print(f"Number of allocation changes: {len(improved_signals)}")

# Print allocation changes
print(f"\n=== ALLOCATION CHANGES (First 10) ===")
for date, action, price in improved_signals[:10]:
    print(f"{date.strftime('%Y-%m-%d')}: {action} at ${price:.2f}")
if len(improved_signals) > 10:
    print(f"... and {len(improved_signals) - 10} more allocation changes")

# Calculate additional metrics for improved strategy
improved_returns = improved_backtest_df['Portfolio_Value_Improved'].pct_change().dropna()
improved_volatility = improved_returns.std() * np.sqrt(252) * 100

print(f"\n=== IMPROVED STRATEGY RISK METRICS ===")
print(f"Improved Strategy Volatility (annualized): {improved_volatility:.1f}%")
print(f"Buy & Hold Volatility (annualized): {buy_hold_volatility:.1f}%")

# Calculate max drawdown for improved strategy
improved_max_dd = calculate_max_drawdown(improved_backtest_df['Portfolio_Value_Improved'])

print(f"Improved Strategy Maximum Drawdown: {improved_max_dd:.1f}%")

# Plot comparison of all three approaches
plt.figure(figsize=(15, 12))

# Performance comparison
plt.subplot(3, 1, 1)
dates_array = improved_backtest_df['Date'].values
improved_values = improved_backtest_df['Portfolio_Value_Improved'].values
buy_hold_values = improved_backtest_df['Buy_Hold_Value'].values

plt.plot(dates_array, improved_values, 
         label='Improved Tactical Strategy', linewidth=2, color='green')
plt.plot(dates_array, buy_hold_values, 
         label='Buy & Hold', linewidth=2, color='red', alpha=0.8)

# Mark allocation changes
if improved_signals:
    signal_dates = [s[0] for s in improved_signals]
    signal_values = [improved_backtest_df[improved_backtest_df['Date'] == date]['Portfolio_Value_Improved'].iloc[0] 
                     for date in signal_dates]
    plt.scatter(signal_dates, signal_values, color='blue', marker='o', s=50, 
               label=f'Allocation Changes ({len(improved_signals)})', alpha=0.7, zorder=5)

plt.title('Improved Tactical Strategy vs Buy & Hold Performance', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Allocation tracking
plt.subplot(3, 1, 2)
allocations = improved_backtest_df['Allocation'].values
plt.fill_between(dates_array, 0, allocations, alpha=0.3, color='green')
plt.plot(dates_array, allocations, color='darkgreen', linewidth=1)
plt.title('Stock Allocation Over Time', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Allocation to Stocks', fontsize=12)
plt.ylim(0.6, 1.05)
plt.grid(True, alpha=0.3)

# Rolling outperformance
plt.subplot(3, 1, 3)
rolling_window = 252  # 1 year
if len(improved_backtest_df) > rolling_window:
    rolling_outperformance = []
    for i in range(rolling_window, len(improved_backtest_df)):
        start_idx = i - rolling_window
        improved_1yr = improved_values[i] / improved_values[start_idx] - 1
        buyhold_1yr = buy_hold_values[i] / buy_hold_values[start_idx] - 1
        rolling_outperformance.append((improved_1yr - buyhold_1yr) * 100)
    
    rolling_dates = dates_array[rolling_window:]
    plt.plot(rolling_dates, rolling_outperformance, color='purple', linewidth=1)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.title('1-Year Rolling Outperformance vs Buy & Hold', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Outperformance (%)', fontsize=12)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final comparison
print(f"\n=== FINAL STRATEGY COMPARISON ===")
print(f"{'Strategy':<25}{'Return':<12}{'Volatility':<12}{'Max DD':<12}{'Allocation Changes'}")
print(f"{'-'*75}")
print(f"{'Buy & Hold':<25}{buy_hold_return:<11.1f}%{buy_hold_volatility:<11.1f}%{buy_hold_max_dd:<11.1f}%{'-':<15}")
print(f"{'Original Strategy':<25}{strategy_return:<11.1f}%{strategy_volatility:<11.1f}%{strategy_max_dd:<11.1f}%{len(signals):<15}")
print(f"{'Improved Strategy':<25}{improved_return:<11.1f}%{improved_volatility:<11.1f}%{improved_max_dd:<11.1f}%{len(improved_signals):<15}")

# Success evaluation
if improved_return > buy_hold_return:
    print(f"\n🎉 SUCCESS! The improved strategy beat buy-and-hold by {improved_return - buy_hold_return:.1f} percentage points!")
else:
    print(f"\n⚠️  The improved strategy underperformed buy-and-hold by {buy_hold_return - improved_return:.1f} percentage points.")
    print("This demonstrates the challenge of beating the market even with apparent statistical patterns.")

print(f"\nKey Insight: Even with clear mean reversion patterns, beating a simple buy-and-hold strategy")
print(f"over the long term is extremely difficult due to the market's strong upward trend and")
print(f"the power of compound growth.")
