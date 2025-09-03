from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
import numpy as np
from qiskit_aer.primitives import Sampler as AerSampler
import matplotlib.pyplot as plt
import datetime
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_algorithms.utils import algorithm_globals
import qiskit
import pandas_datareader.data as web #In order to pull assets, and analyse them
import datetime #to get assets from certain period of times
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None) # How we are gonna display them
pd.set_option('display.max_rows', None)
from functools import reduce
import yfinance as yf
from qiskit.primitives import Sampler
import matplotlib
matplotlib.use('TkAgg')
import nashpy as nash
import random

#Time set
start = datetime.datetime(2023,1,1)
end = datetime.datetime(2025,12,30)

# Download single stock
def get_stock(ticker):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close']].rename(columns={'Close': ticker})
    return data

# Combine multiple stocks
def combine_stocks(tickers):
    data_frames = [get_stock(ticker) for ticker in tickers]
    df_merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), data_frames)
    df_merged = df_merged.fillna(method='ffill').fillna(method='bfill')
    return df_merged

# Combine Stocks

# All available stocks
all_stocks = ['NFLX', 'WMT', 'PFE', 'AMZN', 'CVX', 'TSLA', 'NVDA', 'JPM', 'KO', 'XOM','DIS','CAT','UBER','NKE','ABNB']

# Select 4 random stocks
selected_stocks = random.sample(all_stocks, 4)

# Build portfolio from selected stocks
portfolio = combine_stocks(selected_stocks)
asset_labels = selected_stocks

# Calculate daily returns
returns = portfolio.pct_change().dropna()

#Covariance Matrix, and Mu
mu = returns.mean().values
sigma = returns.cov().values

# Asset labels
asset_labels = selected_stocks

# Plot Covariance Table with each Asset
plt.figure(figsize=(8, 6))
sns.heatmap(sigma, annot=True, fmt=".6f", xticklabels=asset_labels, yticklabels=asset_labels, cmap="viridis")
plt.title("Covariance Matrix (Sigma)")
plt.show()

from qiskit_optimization import QuadraticProgram

# Get number of assets dynamically from the data
num_assets = len(mu)

# Create Quadratic Program
qp = QuadraticProgram()

# Define variables: 2 binary variables per asset (4 levels)
for i in range(num_assets):
    qp.binary_var(name=f"x_{i}_0")  # First bit for asset i
    qp.binary_var(name=f"x_{i}_1")  # Second bit for asset i

# Constraint: For each asset, only one level selected (sum of basis states ≤ 1)
for i in range(num_assets):
    qp.linear_constraint(
        linear={f"x_{i}_0": 1, f"x_{i}_1": 1},
        sense="<=",
        rhs=1,
        name=f"level_constraint_{i}",
    )

#Mixed strategie Multiple investment levels Strategies

# Mixed strategy matrices (Up)
A_up = np.array([
    [0.9, 0.7, 0.5, 0.4],   # Don't Invest
    [0.8, 1.0, 0.9, 0.7],   # Low
    [0.6, 1.2, 1.4, 1.3],   # Medium
    [0.5, 1.0, 1.3, 1.6]    # High
])
B_up = np.array([
    [0.9, 0.6, 0.5, 0.4],
    [0.7, 1.1, 0.8, 0.7],
    [0.6, 1.2, 1.5, 1.3],
    [0.5, 1.0, 1.3, 1.6]
])

# Mixed strategy matrices (Down)
A_down = np.array([
    [1.0, 0.6, 0.4, 0.3],   # Don't Invest
    [0.6, 0.9, 0.6, 0.5],   # Low
    [0.5, 0.6, 0.8, 0.7],   # Medium
    [0.3, 0.5, 0.6, 0.8]    # High
])

B_down = np.array([
    [1.0, 0.6, 0.4, 0.3],
    [0.6, 0.9, 0.6, 0.5],
    [0.5, 0.6, 0.8, 0.7],
    [0.3, 0.5, 0.6, 0.8]
])

# Mixed Strategy Nash Equilibrium Solver 

def get_4x4_nash_equilibrium(trend_positive, alpha=0.3):
    noise_A = np.random.uniform(-0.05, 0.05, size=(4, 4))
    noise_B = np.random.uniform(-0.05, 0.05, size=(4, 4))

    A = (A_up + noise_A) if trend_positive else (A_down + noise_A)
    B = (B_up + noise_B) if trend_positive else (B_down + noise_B)

    game = nash.Game(A, B)
    equilibria = list(game.vertex_enumeration())

    if not equilibria:
        return np.full(4, 0.25)

    # Average all Player A strategies
    strategies = np.array([eq[0] for eq in equilibria])
    avg_strategy = strategies.mean(axis=0)

    # Smoothing
    uniform = np.full(4, 0.25)
    smoothed = alpha * avg_strategy + (1 - alpha) * uniform
    smoothed /= smoothed.sum()
    return smoothed

# Interpretation of the Results
linear = {}

for i in range(num_assets):
    trend_positive = returns.iloc[-1, i] > 0
    investor_probs = get_4x4_nash_equilibrium(trend_positive)

    print(f"\nAsset {i+1} ({asset_labels[i]}):")
    print("Investor probabilities:")
    print(f"  Don't Invest:   {investor_probs[0]:.2f}")
    print(f"  Invest Low:     {investor_probs[1]:.2f}")
    print(f"  Invest Medium:  {investor_probs[2]:.2f}")
    print(f"  Invest High:    {investor_probs[3]:.2f}")
    
    scaling = np.array([0, 0.3, 0.6, 1.0])
    risk_weight = np.dot(investor_probs, scaling)
    adjusted_mu = mu[i] * risk_weight

# Reward system adjusted to the Game theory Set-up


for i in range(num_assets):
    trend_positive = returns.iloc[-1, i] > 0
    investor_probs = get_4x4_nash_equilibrium(trend_positive)

    scaling = np.array([0, 0.3, 0.6, 1.0])
    risk_weight = np.dot(investor_probs, scaling)
    adjusted_mu = mu[i] * risk_weight

    # Here's the key: map game theory result to each QAOA bit variable
    linear[f"x_{i}_0"] = adjusted_mu * 1.0  # Assume Low investment if bit0 = 1
    linear[f"x_{i}_1"] = adjusted_mu * 2.0  # Assume Medium investment if bit1 = 1

# Risk Term

lambda_risk = 0.5
quadratic = {}
for i in range(num_assets):
    for j in range(num_assets):
        for a_idx, w_a in zip([0, 1], [1.0, 2.0]):
            for b_idx, w_b in zip([0, 1], [1.0, 2.0]):
                var_a = f"x_{i}_{a_idx}"
                var_b = f"x_{j}_{b_idx}"
                weight = lambda_risk * sigma[i, j] * w_a * w_b
                if (var_a, var_b) in quadratic:
                    quadratic[(var_a, var_b)] += weight
                else:
                    quadratic[(var_a, var_b)] = weight

# Maximize expected return
qp.maximize(linear=linear, quadratic=quadratic)

# Solve with QAOA
cobyla = COBYLA()
cobyla.set_options(maxiter=250)
qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla, reps=3)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qp)

# Print results in the format you showed
print("\n---------------- Optimal Result ----------------")
print(f"Optimal: selection {result.x}, value {result.fval:.4f}")

print("\n--- Full result ---")
print("selection    value    probability")
print("---")
for sample in result.samples:
    print(f"{sample.x}   {sample.fval:.4f}    {sample.probability:.4f}")

# Interpret result and prepare data for table
investment_levels = []
investment_amounts = []
Initial_amount= 100000
# Define investment dollar amounts per level
low_investment_amount = Initial_amount/16
medium_investment_amount = Initial_amount/8
high_investment_amount = Initial_amount/4
no_investment_amount = 0

for i in range(num_assets):
    bit0 = result.x[2 * i]
    bit1 = result.x[2 * i + 1]

    if bit0 == 0 and bit1 == 0:
        level = "Don't Invest"
        amount = no_investment_amount
    elif bit0 == 1 and bit1 == 0:
        level = "Invest Low"
        amount = low_investment_amount
    elif bit0 == 0 and bit1 == 1:
        level = "Invest Medium"
        amount = medium_investment_amount
    elif bit0 == 1 and bit1 == 1:
        level = "Invest High"
        amount = high_investment_amount
    else:
        level = "Invalid"
        amount = 0

    investment_levels.append(level)
    investment_amounts.append(amount)

# Create DataFrame for easier table formatting, now including amounts
# Create DataFrame for display
strategy_df = pd.DataFrame({
    'Asset': asset_labels,
    'Investment Level': investment_levels,
    'Investment Amount ($)': investment_amounts
})

# Add total row
total_row = pd.DataFrame({
    'Asset': ['Total'],
    'Investment Level': ['—'],
    'Investment Amount ($)': [sum(investment_amounts)]
})
strategy_df = pd.concat([strategy_df, total_row], ignore_index=True)

# Format currency
strategy_df['Investment Amount ($)'] = strategy_df['Investment Amount ($)'].apply(
    lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x
)

# Display enhanced table
plt.figure(figsize=(10, 4))
ax = plt.gca()
ax.axis('off')

# Create table with styling
table = plt.table(
    cellText=strategy_df.values,
    colLabels=strategy_df.columns,
    loc='center',
    cellLoc='center',
    colColours=['#f7f7f7']*len(strategy_df.columns)
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

# Add title
plt.title("Optimal Investment Strategy", pad=20, fontsize=14)
plt.tight_layout()
plt.show()

# NEW CODE FOR PORTFOLIO SIMULATION OVER 12 MONTHS

# Calculate portfolio weights based on investment amounts
total_invested = sum(investment_amounts)
weights = np.array(investment_amounts) / total_invested

# Portfolio statistics from the data
portfolio_mu = np.dot(weights, mu)
portfolio_sigma = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))

# Simulation parameters
months = 24
initial_value = total_invested
num_simulations = 1  # Just show one path for clarity

# Generate monthly returns from normal distribution
days = 504  # 1 trading year
np.random.seed(42)
daily_returns = np.random.normal(portfolio_mu, portfolio_sigma, days)

# Calculate portfolio values over time
portfolio_values = [initial_value]
for ret in daily_returns:
    portfolio_values.append(portfolio_values[-1] * (1 + ret))

# Create time points for x-axis

days_list = list(range(len(portfolio_values)))  # 0 to 252
plt.plot(days_list, portfolio_values, linestyle='-', color='b')

# Risk-free rate assumption
risk_free_rate = 0.02

# Sharpe Ratio calculation
sharpe_ratio = (portfolio_mu * 504 - risk_free_rate) / (portfolio_sigma * np.sqrt(504))

# Annotated statistics text
stats_text = (
    f"Expected Annual Return: {portfolio_mu * 504:.2%}\n"
    f"Annual Volatility: {portfolio_sigma * np.sqrt(504):.2%}\n"
    f"Sharpe Ratio: {sharpe_ratio:.2f}"
)

# Plot simulation
plt.figure(figsize=(10, 6))
plt.plot(days_list, portfolio_values, linestyle='-', color='b')
plt.title('Simulated Portfolio Value Over 2 years')
plt.xlabel('Days')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.axhline(y=initial_value, color='r', linestyle='--', label='Initial Investment')
plt.legend()

# Add text box with statistics
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

