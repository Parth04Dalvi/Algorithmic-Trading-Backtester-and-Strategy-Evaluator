# --- Algorithmic Trading Backtester and Strategy Evaluator ---
# This version includes multiple strategies, transaction costs, and advanced metrics (CAGR).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# --- CONFIGURATION ---
INITIAL_CAPITAL = 10000.0
# SMA Strategy Windows
SHORT_WINDOW = 40  
LONG_WINDOW = 100 
# RSI Strategy Window
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
# Transaction Cost
TRANSACTION_COST = 0.0005  # 0.05% per trade (realistic commission/slippage mock)

# --- 1. MOCK DATA GENERATION ---

def generate_mock_stock_data(days: int = 500) -> pd.DataFrame:
    """
    Generates mock time-series data for a stock, simulating price fluctuations.
    """
    print(f"Generating {days} days of mock stock data...")
    np.random.seed(42)
    
    start_price = 100.0
    # Add a slight upward drift to simulate a general market trend
    daily_returns = np.random.normal(loc=0.0005, scale=0.01, size=days)
    
    price_series = start_price * (1 + daily_returns).cumprod()
    
    data = pd.DataFrame({
        'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=days)),
        'Close': price_series
    })
    
    data.set_index('Date', inplace=True)
    return data

# --- 2. STRATEGY IMPLEMENTATION (RSI Calculation Utility) ---

def calculate_rsi(data: pd.DataFrame, window: int) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def implement_sma_crossover(data: pd.DataFrame) -> pd.DataFrame:
    """Implements the Simple Moving Average (SMA) Crossover Strategy."""
    data['Short_SMA'] = data['Close'].rolling(window=SHORT_WINDOW, min_periods=1).mean()
    data['Long_SMA'] = data['Close'].rolling(window=LONG_WINDOW, min_periods=1).mean()
    
    data['Signal'] = 0.0
    data['Signal'][LONG_WINDOW:] = np.where(
        data['Short_SMA'][LONG_WINDOW:] > data['Long_SMA'][LONG_WINDOW:], 1.0, 0.0
    )
    
    # Position: Find points where the signal changes (the trade decision: 1=Buy, -1=Sell)
    data['Position'] = data['Signal'].diff()
    return data

def implement_rsi_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Implements the Relative Strength Index (RSI) Strategy.
    
    Strategy:
    - BUY signal (1.0): When RSI crosses BELOW the Oversold threshold (30).
    - SELL signal (-1.0): When RSI crosses ABOVE the Overbought threshold (70).
    """
    data['RSI'] = calculate_rsi(data, RSI_PERIOD)
    data['Signal'] = 0.0
    
    # Buy when RSI drops into Oversold territory
    data['Signal'] = np.where(data['RSI'] < RSI_OVERSOLD, 1.0, data['Signal'])
    
    # Sell when RSI rises into Overbought territory
    data['Signal'] = np.where(data['RSI'] > RSI_OVERBOUGHT, -1.0, data['Signal'])
    
    # Filter the signal to only execute trades at the crossover point
    # We use forward filling (ffill) to hold position until the next signal
    data['Signal'] = data['Signal'].replace(to_replace=0.0, method='ffill').fillna(0.0)

    # Position: Find points where the signal changes (the trade decision)
    data['Position'] = data['Signal'].diff()
    return data

def implement_strategy(data: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
    """Dispatches to the selected strategy implementation."""
    if strategy_type == 'SMA':
        return implement_sma_crossover(data.copy())
    elif strategy_type == 'RSI':
        return implement_rsi_strategy(data.copy())
    else:
        raise ValueError("Invalid strategy type selected.")


# --- 3. BACKTESTING & PERFORMANCE METRICS ---

def run_backtest(data: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
    """
    Simulates trades, applies transaction costs, and calculates portfolio returns.
    """
    
    # 1. Calculate Daily Returns (Logarithmic)
    data['Daily_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # 2. Calculate Strategy Returns: Position * Daily_Return
    data['Strategy_Return'] = data['Position'].shift(1) * data['Daily_Return']
    
    # 3. Apply Transaction Costs (executed only when position changes)
    data['Transaction_Cost'] = np.where(data['Position'].abs() > 0, TRANSACTION_COST, 0)
    data['Strategy_Return'] -= data['Transaction_Cost']
    
    # 4. Calculate Cumulative Strategy Returns (Equity Curve)
    # The first value is the starting capital
    data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod() * initial_capital
    data['Cumulative_Benchmark_Return'] = (1 + data['Daily_Return']).cumprod() * initial_capital
    
    # --- Performance Metrics Calculation ---
    
    daily_strategy_return = data['Strategy_Return'].dropna()
    
    # Compound Annual Growth Rate (CAGR) (New Metric)
    years = (data.index[-1] - data.index[0]).days / 365.25
    cagr = ((data['Cumulative_Strategy_Return'].iloc[-1] / initial_capital) ** (1/years)) - 1
    
    # Sharpe Ratio: Assumes risk-free rate is zero.
    sharpe_ratio = np.sqrt(252) * daily_strategy_return.mean() / daily_strategy_return.std()

    # Maximum Drawdown (MDD)
    data['Cumulative_Peak'] = data['Cumulative_Strategy_Return'].cummax()
    data['Drawdown'] = data['Cumulative_Strategy_Return'] / data['Cumulative_Peak'] - 1
    max_drawdown = data['Drawdown'].min()
    
    final_portfolio_value = data['Cumulative_Strategy_Return'].iloc[-1]
    
    return {
        'final_value': final_portfolio_value,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cagr': cagr,
        'data': data
    }

# --- 4. VISUALIZATION ---

def plot_results(results: Dict[str, Any], strategy_name: str):
    """
    Visualizes the strategy's performance against a benchmark.
    """
    df = results['data']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Algorithmic Trading Strategy Performance: {strategy_name}', fontsize=16, y=0.99)
    
    # Plot 1: Price Action, Indicators, and Trade Signals
    axes[0].plot(df['Close'], label='Close Price', color='gray', alpha=0.7)
    axes[0].set_title(f'Price Action and Trade Signals ({strategy_name})')
    
    if strategy_name == 'SMA Crossover':
        axes[0].plot(df['Short_SMA'], label=f'Short SMA ({SHORT_WINDOW}D)', color='cyan')
        axes[0].plot(df['Long_SMA'], label=f'Long SMA ({LONG_WINDOW}D)', color='magenta')
    elif strategy_name == 'RSI Strategy':
        # Plot RSI indicator in a separate subplot
        ax_rsi = axes[0].twinx()
        ax_rsi.plot(df['RSI'], label='RSI', color='lime', alpha=0.7)
        ax_rsi.axhline(RSI_OVERBOUGHT, color='red', linestyle='--', alpha=0.5)
        ax_rsi.axhline(RSI_OVERSOLD, color='green', linestyle='--', alpha=0.5)
        ax_rsi.set_ylabel('RSI Value', color='lime')
        
    # Plot trade signals (Buy/Sell arrows)
    axes[0].plot(df.loc[df['Position'] == 1.0].index, 
                 df['Close'][df['Position'] == 1.0], 
                 '^', markersize=10, color='green', label='BUY Signal')
    axes[0].plot(df.loc[df['Position'] == -1.0].index, 
                 df['Close'][df['Position'] == -1.0], 
                 'v', markersize=10, color='red', label='SELL Signal')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Equity Curve (Strategy vs. Benchmark)
    axes[1].plot(df['Cumulative_Strategy_Return'], label='Strategy Equity (w/ Costs)', color='blue')
    axes[1].plot(df['Cumulative_Benchmark_Return'], label='Benchmark (Buy & Hold)', color='orange', linestyle='--')
    axes[1].set_title(f'Cumulative Portfolio Value (Initial Capital: ${INITIAL_CAPITAL:,.2f} | Cost: {TRANSACTION_COST*100:.2f}%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    axes[2].plot(df['Drawdown'], label='Drawdown', color='red')
    axes[2].axhline(results['max_drawdown'], color='darkred', linestyle='--', linewidth=1, label=f"Max Drawdown ({results['max_drawdown']*100:.2f}%)")
    axes[2].set_title('Strategy Drawdown (Losses from Peak)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('Date')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 5. EXECUTION ---

def print_summary(strategy_name: str, results: Dict[str, Any], data: pd.DataFrame):
    """Prints formatted summary statistics."""
    print("\n" + "="*60)
    print(f"✨ STRATEGY EVALUATION SUMMARY: {strategy_name} ✨")
    print("="*60)
    print(f"Final Portfolio Value:      ${results['final_value']:,.2f}")
    print(f"Compound Annual Growth Rate: {results['cagr']*100:.2f}%")
    print("-" * 60)
    print(f"Sharpe Ratio (Annualized):  {results['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown (MDD):     {results['max_drawdown']*100:.2f}%")
    print(f"Total Trades Executed:      {data['Position'].abs().sum()}")
    print("="*60)

if __name__ == '__main__':
    
    # 1. Prepare Data
    stock_data = generate_mock_stock_data(days=500)
    
    # --- RUN SMA STRATEGY ---
    sma_data = implement_strategy(stock_data, 'SMA')
    sma_results = run_backtest(sma_data, INITIAL_CAPITAL)
    print_summary('SMA Crossover', sma_results, sma_data)
    plot_results(sma_results, 'SMA Crossover')
    
    # --- RUN RSI STRATEGY ---
    rsi_data = implement_strategy(stock_data, 'RSI')
    rsi_results = run_backtest(rsi_data, INITIAL_CAPITAL)
    print_summary('RSI Strategy', rsi_results, rsi_data)
    plot_results(rsi_results, 'RSI Strategy')
