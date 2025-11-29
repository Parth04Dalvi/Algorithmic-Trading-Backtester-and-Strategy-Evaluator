# Algorithmic-Trading-Backtester-and-Strategy-Evaluator 💻
This project showcases your ability to handle real world data (time-series), apply financial models, and evaluate performance using critical metrics a key area for quantitative roles.

Algorithmic Trading Backtester and Strategy Evaluator 📊

Overview and Project Goal

This project implements a foundational Algorithmic Trading Backtester designed to test and evaluate the performance of quantitative trading strategies using historical time-series data. It is a critical demonstration of expertise in Financial Modeling, Time-Series Analysis, and Performance Evaluation using standard risk-adjusted metrics.

The system processes mock stock price data, executes trade signals based on a strategy, simulates portfolio returns, and calculates key financial risk indicators.

🧠 Strategy and Algorithm

The project employs the classic Moving Average Crossover Strategy (SMA Crossover).
Strategy Logic:
The primary signal is generated based on the relationship between two Simple Moving Averages (SMAs):

A BUY signal is generated when a short-term SMA crosses above a long-term SMA, signaling a potential upward trend.

A SELL signal occurs when the short SMA crosses below the long SMA.

Implementation:
The core logic for time-series data manipulation and rolling window calculations is handled efficiently using Pandas and NumPy.

✨ Key Metrics and Risk Management

The heart of the backtester lies in its ability to quantify risk and return, providing a real-world view of strategy viability:

Sharpe Ratio (Annualized): Measures the risk-adjusted return of the strategy. A higher Sharpe Ratio indicates better performance for the amount of risk taken relative to the volatility of returns.

Maximum Drawdown (MDD): Identifies the largest peak-to-trough decline the portfolio experienced from a historical peak, quantifying the worst-case potential historical loss.

Equity Curve Generation: Plots the cumulative returns of the strategy against a simple Buy-and-Hold benchmark, offering a clear visual comparison of performance over the testing period.

💻 Technology Stack

The project leverages the following technologies:

Core Language: Python

Data Handling: Pandas and NumPy for vectorized time-series operations.

Visualization: Matplotlib for plotting price action, signals, equity curves, and drawdown analysis.

Concepts: Time-Series Modeling, Financial Metrics (Sharpe Ratio, MDD), Algorithmic Trading Logic.

Execution

The project is contained within the single backtester.py file.

To run the simulation and display the plots:

# Ensure dependencies are installed:
pip install pandas numpy matplotlib

# Run the script:
python backtester.py


The console output will provide a numerical summary of the strategy's final value, Sharpe Ratio, and Maximum Drawdown, followed by the three associated performance plots.
