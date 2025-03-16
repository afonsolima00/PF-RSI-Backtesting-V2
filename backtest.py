import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def backtest_rsi_strategy(ticker='BTC-USD', start_date='2017-01-01', end_date='2025-02-20', 
                         rsi_period=14, buy_threshold=25, sell_threshold=75,
                         use_position_sizing=False, use_stop_loss=False, 
                         risk_per_trade=0.02, stop_loss_pct=0.05):
    """
    Backtest RSI strategy with optional position sizing and stop-loss
    """
    # Fetch weekly historical data
    df = yf.download(ticker, start=start_date, end=end_date, interval='1wk')
    
    # Calculate RSI
    df['RSI'] = calculate_rsi(df['Close'], rsi_period)
    
    # Initialize columns
    df['Position'] = 0  # Position size (0 to 1)
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = 0.0
    df['Equity'] = 1.0  # Start with $1
    
    # Tracking variables
    in_position = False
    entry_price = 0
    stop_loss_price = 0
    position_size = 1.0  # Default full position
    
    # For tracking trades
    trades = []
    
    # Backtest logic
    for i in range(1, len(df)):
        current_rsi = df['RSI'].iloc[i].item()  # Get scalar value
        current_price = df['Close'].iloc[i].item()  # Current price
        
        # Check if stop loss was hit
        if in_position and use_stop_loss and current_price <= stop_loss_price:
            # Close position
            df.loc[df.index[i], 'Position'] = 0
            in_position = False
            
            # Record the trade
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': df.index[i],
                'Entry Price': entry_price,
                'Exit Price': current_price,
                'Position Size': position_size,
                'Exit Type': 'Stop Loss',
                'Return': (current_price/entry_price - 1) * position_size
            })
            
            print(f"Stop Loss at {df.index[i]} - Entry: {entry_price:.2f}, " 
                  f"Exit: {current_price:.2f}, Loss: {(current_price/entry_price - 1):.2%}")
            
        # Buy signal
        elif not in_position and current_rsi < buy_threshold:
            # Calculate position size if position sizing is enabled
            if use_position_sizing:
                # Position sizing based on risk per trade
                if use_stop_loss:
                    position_size = risk_per_trade / stop_loss_pct
                else:
                    # If no stop loss, limit risk to a fixed percentage
                    position_size = risk_per_trade * 5  # Arbitrary multiplier 
                
                # Cap position size at 100%
                position_size = min(position_size, 1.0)
            else:
                position_size = 1.0  # Use full capital
                
            df.loc[df.index[i], 'Position'] = position_size
            in_position = True
            entry_price = current_price
            entry_date = df.index[i]
            
            # Set stop loss price if stop loss is enabled
            if use_stop_loss:
                stop_loss_price = entry_price * (1 - stop_loss_pct)
            
            print(f"Buy at {df.index[i]} - Price: {entry_price:.2f}, RSI: {current_rsi:.2f}, "
                  f"Position Size: {position_size:.2%}")
            
        # Sell signal (based on RSI)
        elif in_position and current_rsi > sell_threshold:
            # Close position
            df.loc[df.index[i], 'Position'] = 0
            in_position = False
            
            # Record the trade
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': df.index[i],
                'Entry Price': entry_price,
                'Exit Price': current_price,
                'Position Size': position_size,
                'Exit Type': 'RSI Signal',
                'Return': (current_price/entry_price - 1) * position_size
            })
            
            print(f"Sell at {df.index[i]} - Entry: {entry_price:.2f}, "
                  f"Exit: {current_price:.2f}, Return: {(current_price/entry_price - 1):.2%}")
        
        # Maintain position
        elif in_position:
            df.loc[df.index[i], 'Position'] = position_size
    
    # Calculate strategy returns
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
    
    # Update equity curve with cumulative returns
    df['Equity'] = (1 + df['Strategy_Returns']).cumprod()
    
    # Calculate performance metrics
    total_return = df['Equity'].iloc[-1] - 1
    buy_and_hold_return = (df['Close'].iloc[-1].item() / df['Close'].iloc[0].item()) - 1
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    if not trades_df.empty:
        win_rate = len(trades_df[trades_df['Return'] > 0]) / len(trades_df)
    else:
        win_rate = 0
    
    # Additional metrics
    strategy_returns = df['Strategy_Returns'].fillna(0)
    mean_return = strategy_returns.mean() * 52  # Annualized mean return
    std_dev = strategy_returns.std() * np.sqrt(52)  # Annualized std dev
    sharpe_ratio = mean_return / std_dev if std_dev != 0 else 0
    max_drawdown = calculate_max_drawdown(strategy_returns)
    
    years = (df.index[-1] - df.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    strategy_type = "Original RSI Strategy"
    if use_position_sizing and use_stop_loss:
        strategy_type = "RSI with Position Sizing & Stop-Loss"
    elif use_position_sizing:
        strategy_type = "RSI with Position Sizing"
    elif use_stop_loss:
        strategy_type = "RSI with Stop-Loss"
        
    # Summary
    print(f"\nPerformance Summary: {strategy_type}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Total Strategy Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Buy & Hold Return: {buy_and_hold_return:.2%}")
    print(f"Number of Trades: {len(trades_df)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    # Compile results for comparison
    results = {
        'Strategy': strategy_type,
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Buy & Hold Return': buy_and_hold_return,
        'Number of Trades': len(trades_df),
        'Win Rate': win_rate,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown
    }
    
    # Save trade log if there are trades
    if not trades_df.empty:
        trades_filename = f"{strategy_type.replace(' ', '_')}_{ticker.replace('-', '_')}_trades.csv"
        trades_df.to_csv(trades_filename, index=False)
        print(f"Trade log saved to {trades_filename}")
    
    return df, results, trades_df

def compare_strategies(ticker='BTC-USD', start_date='2017-01-01', end_date='2025-02-20', 
                       rsi_period=14, buy_threshold=25, sell_threshold=75,
                       risk_per_trade=0.02, stop_loss_pct=0.05):
    """
    Compare original RSI strategy against enhanced version with risk controls
    """
    # Run the original strategy
    df_original, results_original, trades_original = backtest_rsi_strategy(
        ticker, start_date, end_date, rsi_period, buy_threshold, sell_threshold,
        use_position_sizing=False, use_stop_loss=False
    )
    
    # Run the enhanced strategy with both position sizing and stop-loss
    df_enhanced, results_enhanced, trades_enhanced = backtest_rsi_strategy(
        ticker, start_date, end_date, rsi_period, buy_threshold, sell_threshold,
        use_position_sizing=True, use_stop_loss=True, 
        risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct
    )
    
    # Compile results into a dataframe
    results_df = pd.DataFrame([results_original, results_enhanced])
    
    # Save results to CSV
    results_filename = f"rsi_strategy_comparison_{ticker.replace('-', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nResults saved to {results_filename}")
    
    # Create performance comparison chart
    plt.figure(figsize=(12, 10))
    
    # Plot equity curves
    plt.subplot(3, 1, 1)
    plt.plot(df_original.index, df_original['Equity'], label='Original RSI')
    plt.plot(df_enhanced.index, df_enhanced['Equity'], label='Enhanced RSI (Position Sizing & Stop-Loss)')
    plt.plot(df_original.index, (1 + df_original['Returns']).cumprod(), label='Buy & Hold', linestyle='--')
    
    plt.title(f'Equity Curves Comparison - {ticker}')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    
    # Plot drawdowns
    plt.subplot(3, 1, 2)
    
    # Calculate drawdowns for each strategy
    def calculate_drawdown_series(returns):
        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return drawdown
    
    plt.plot(df_original.index, calculate_drawdown_series(df_original['Strategy_Returns']), label='Original RSI')
    plt.plot(df_enhanced.index, calculate_drawdown_series(df_enhanced['Strategy_Returns']), label='Enhanced RSI')
    
    plt.title('Drawdown Comparison')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.legend()
    
    # Plot RSI for reference
    plt.subplot(3, 1, 3)
    plt.plot(df_original.index, df_original['RSI'], color='purple')
    plt.axhline(y=buy_threshold, color='g', linestyle='--', label=f'Buy Threshold ({buy_threshold})')
    plt.axhline(y=sell_threshold, color='r', linestyle='--', label=f'Sell Threshold ({sell_threshold})')
    
    plt.title('RSI Indicator')
    plt.ylabel('RSI Value')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    chart_filename = f"rsi_strategy_comparison_{ticker.replace('-', '_')}_{datetime.now().strftime('%Y%m%d')}.png"
    plt.tight_layout()
    plt.savefig(chart_filename, dpi=300)
    print(f"Chart saved to {chart_filename}")
    
    # Display the comparison results
    print("\nStrategy Comparison Summary:")
    comparison_metrics = ['Total Return', 'Sharpe Ratio', 'Maximum Drawdown', 'Number of Trades', 'Win Rate']
    comparison_df = results_df[['Strategy'] + comparison_metrics]
    
    # Format percentages for better readability
    for col in ['Total Return', 'Maximum Drawdown', 'Win Rate']:
        comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2%}")
        
    print(comparison_df.to_string(index=False))
    
    # Print improvement percentages
    print("\nImprovements with Risk Controls:")
    if float(results_enhanced['Sharpe Ratio']) > float(results_original['Sharpe Ratio']):
        sharpe_improvement = (results_enhanced['Sharpe Ratio'] / results_original['Sharpe Ratio'] - 1) * 100
        print(f"Sharpe Ratio Improvement: +{sharpe_improvement:.1f}%")
    else:
        sharpe_decline = (1 - results_enhanced['Sharpe Ratio'] / results_original['Sharpe Ratio']) * 100
        print(f"Sharpe Ratio Decline: -{sharpe_decline:.1f}%")
        
    if results_enhanced['Maximum Drawdown'] > results_original['Maximum Drawdown']:
        drawdown_worse = (results_enhanced['Maximum Drawdown'] / results_original['Maximum Drawdown'] - 1) * 100
        print(f"Maximum Drawdown Worsened: +{drawdown_worse:.1f}%")
    else:
        drawdown_improvement = (1 - results_enhanced['Maximum Drawdown'] / results_original['Maximum Drawdown']) * 100
        print(f"Maximum Drawdown Improvement: +{drawdown_improvement:.1f}%")
    
    return {
        'original': (df_original, results_original, trades_original),
        'enhanced': (df_enhanced, results_enhanced, trades_enhanced),
        'comparison': results_df
    }

# Execute backtest comparison
if __name__ == "__main__":
    # Run the comparison with default parameters
    results = compare_strategies()
