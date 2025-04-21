import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy import stats
import yfinance as yf
from itertools import combinations
from sklearn.cluster import KMeans
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)

class StatisticalArbitrage:
    def __init__(self, assets, start_date='2018-01-01', end_date='2023-01-01'):
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.pairs = []
        self.cointegrated_pairs = []
        self.models = {}
        self.trade_signals = {}
        self.trade_performance = {}
        self.forecast_performance = {}
        
    def fetch_data(self):
        self.data = yf.download(self.assets, start=self.start_date, end=self.end_date)['Close']
        self.data.dropna(inplace=True)
        return self.data
    
    def find_cointegrated_pairs(self, significance_level=0.05):
        n = len(self.assets)
        pvalues = np.zeros((n, n))
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                stock1 = self.assets[i]
                stock2 = self.assets[j]
                try:
                    result = coint(self.data[stock1], self.data[stock2])
                    pvalue = result[1]
                    pvalues[i, j] = pvalue
                    if pvalue < significance_level:
                        pairs.append((stock1, stock2, pvalue))
                except:
                    # Skip if error
                    continue
        self.cointegrated_pairs = sorted(pairs, key=lambda x: x[2])
        
        return self.cointegrated_pairs
    
    def plot_correlation_heatmap(self):
        returns = self.data.pct_change().dropna()
        corr = returns.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', annot_kws={"size": 8})
        plt.title('Correlation Heatmap of Asset Returns')
        plt.tight_layout()
        plt.show()
        
        return corr
    
    def analyze_pair(self, stock1, stock2):
        if len(self.cointegrated_pairs) == 0:
            self.find_cointegrated_pairs()
        
        S1 = self.data[stock1]
        S2 = self.data[stock2]
        # Use linear regression to find hedge ratio instead of simple normalization
        from statsmodels.regression.linear_model import OLS
        model = OLS(S1, S2).fit()
        hedge_ratio = model.params[0]
        
        # Calculate the spread using the hedge ratio
        spread = S1 - hedge_ratio * S2
        
        # Check if spread has variation
        if spread.std() < 1e-10:
            print(f"Warning: Spread between {stock1} and {stock2} has near-zero variance.")
            # Add a tiny amount of noise to avoid ADF test failure
            spread = spread + np.random.normal(0, 1e-8, len(spread))
        
        # Test for stationarity with error handling
        try:
            adf_result = adfuller(spread)
            adf_pvalue = adf_result[1]
        except ValueError as e:
            print(f"ADF Test failed: {e}")
            adf_pvalue = 1.0  # Set to non-stationary as default
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        S1_plot = (S1 - S1.mean()) / S1.std()
        S2_plot = (S2 - S2.mean()) / S2.std()
        
        ax1.plot(S1_plot, label=f"{stock1} (Normalized)")
        ax1.plot(S2_plot, label=f"{stock2} (Normalized)")
        ax1.set_title(f'Normalized Price Series: {stock1} vs {stock2}')
        ax1.legend()
        ax1.grid(True)        
        spread_norm = (spread - spread.mean()) / spread.std() if spread.std() > 0 else spread
        
        ax2.plot(spread_norm, label='Spread')
        ax2.axhline(y=spread_norm.mean(), color='r', linestyle='-', label='Mean')
        ax2.axhline(y=spread_norm.mean() + 1, color='g', linestyle='--', label='+1 Std Dev')
        ax2.axhline(y=spread_norm.mean() - 1, color='g', linestyle='--', label='-1 Std Dev')
        ax2.axhline(y=spread_norm.mean() + 2, color='y', linestyle='--', label='+2 Std Dev')
        ax2.axhline(y=spread_norm.mean() - 2, color='y', linestyle='--', label='-2 Std Dev')
        ax2.set_title(f'Spread (ADF p-value: {adf_pvalue:.4f})')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return spread
    
    def fit_arima_garch(self, stock1, stock2, arima_order=(1,1,1), garch_order=(1,1)):
        try:
            spread = self.analyze_pair(stock1, stock2)
            arima_model = ARIMA(spread, order=arima_order)
            arima_results = arima_model.fit()
            
            garch_model = arch_model(arima_results.resid, vol='GARCH', p=garch_order[0], q=garch_order[1])
            garch_results = garch_model.fit(disp='off')
            
            self.models[(stock1, stock2)] = {
                'spread': spread,
                'arima': arima_results,
                'garch': garch_results
            }            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))            
            ax1.plot(spread, label='Original Spread')
            ax1.plot(arima_results.fittedvalues, color='red', label='ARIMA Fit')
            ax1.set_title('Spread and ARIMA Fit')
            ax1.legend()
            
            ax2.plot(arima_results.resid)
            ax2.set_title('ARIMA Residuals')            
            plot_acf(arima_results.resid, ax=ax3)
            ax3.set_title('ACF of ARIMA Residuals')
            
            ax4.plot(garch_results.conditional_volatility)
            ax4.set_title('GARCH Conditional Volatility')
            
            plt.tight_layout()
            plt.show()
            
            f, Pxx = periodogram(spread, detrend='linear')
            plt.figure(figsize=(12, 6))
            plt.semilogy(f, Pxx)
            plt.title('Periodogram for Seasonality Detection')
            plt.xlabel('Frequency')
            plt.ylabel('Power Spectral Density')
            plt.grid(True)
            plt.show()
            
            return self.models[(stock1, stock2)]
        except Exception as e:
            print(f"Error fitting ARIMA-GARCH for {stock1}-{stock2}: {str(e)}")
            return None
    
    def evaluate_forecast_performance(self, stock1, stock2, horizons=[1, 5, 20], test_size=252):
        if (stock1, stock2) not in self.models:
            self.fit_arima_garch(stock1, stock2)
            if (stock1, stock2) not in self.models:
                return None
        
        model_data = self.models[(stock1, stock2)]
        spread = model_data['spread']        
        test_size = min(test_size, len(spread) // 2)
        
        results = {}
        forecast_df = pd.DataFrame(index=spread.index[-test_size:])
        
        for h in horizons:
            print(f"Evaluating {h}-day ahead forecasts...")
            forecast_errors = []
            rw_errors = []
            forecast_values = []
            
            for i in range(test_size - h):
                train_end = len(spread) - test_size + i
                train_data = spread.iloc[:train_end]
                test_point = spread.iloc[train_end + h - 1] 
                
                try:
                    arima_order = model_data['arima'].model.order
                    temp_model = ARIMA(train_data, order=arima_order)
                    temp_fit = temp_model.fit()
                    
                    forecast = temp_fit.forecast(steps=h)
                    forecast_value = forecast.iloc[-1]
                    rw_forecast = train_data.iloc[-1]
                    error = test_point - forecast_value
                    rw_error = test_point - rw_forecast
                    
                    forecast_errors.append(error)
                    rw_errors.append(rw_error)
                    forecast_values.append(forecast_value)
                except Exception as e:
                    print(f"Error in forecast iteration {i} for horizon {h}: {str(e)}")
                    continue
            
            if not forecast_errors:
                print(f"No valid forecasts for horizon {h}")
                continue
            forecast_errors = np.array(forecast_errors)
            rw_errors = np.array(rw_errors)
            rmse = np.sqrt(np.mean(forecast_errors**2))
            rw_rmse = np.sqrt(np.mean(rw_errors**2))
            improvement = (rw_rmse - rmse) / rw_rmse * 100
            results[h] = {
                'rmse': rmse,
                'random_walk_rmse': rw_rmse,
                'improvement': improvement,
                'forecast_errors': forecast_errors
            }
            
            forecast_col = f'forecast_{h}day'
            forecast_df[forecast_col] = np.nan
            forecast_df[forecast_col].iloc[h-1:h-1+len(forecast_values)] = forecast_values        
        plt.figure(figsize=(14, 10))
        plt.plot(spread.iloc[-test_size:], label='Actual Spread', color='black')
        
        colors = ['blue', 'red', 'green']
        for i, h in enumerate(horizons):
            if h in results:
                forecast_col = f'forecast_{h}day'
                if forecast_col in forecast_df.columns:
                    plt.plot(forecast_df[forecast_col], 
                             label=f'{h}-day Forecast (RMSE: {results[h]["rmse"]:.3f})', 
                             color=colors[i % len(colors)],
                             alpha=0.7)
        
        plt.title('Forecast Performance Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print("\nForecast Performance Summary:")
        print("-" * 65)
        print(f"{'Horizon':>10} | {'RMSE':>10} | {'RW RMSE':>10} | {'Improvement':>15}")
        print("-" * 65)
        for h in sorted(results.keys()):
            print(f"{h:>10d} | {results[h]['rmse']:>10.3f} | {results[h]['random_walk_rmse']:>10.3f} | {results[h]['improvement']:>15.2f}%")
        
        self.forecast_performance[(stock1, stock2)] = results
        return results
    
    def generate_signals(self, stock1, stock2, entry_threshold=2.0, exit_threshold=0.5):
        if (stock1, stock2) not in self.models:
            result = self.fit_arima_garch(stock1, stock2)
            if result is None:
                return None
        
        model_data = self.models[(stock1, stock2)]
        spread = model_data['spread']
        arima_model = model_data['arima']
        garch_model = model_data['garch']
        
        try:
            mean_spread = spread.mean()
            std_spread = spread.std()
            z_score = (spread - mean_spread) / std_spread
            
            signals = pd.Series(index=spread.index)
            # Long position (spread < -threshold)
            signals[z_score < -entry_threshold] = 1
            # Short position (spread > threshold)
            signals[z_score > entry_threshold] = -1
            # Exit positions (spread reverts to mean)
            signals[(z_score > -exit_threshold) & (z_score < exit_threshold)] = 0
            # Forward fill signals to hold positions
            signals = signals.ffill().fillna(0)
            # Calculate adjusted signals based on volatility forecast
            volatility_forecast = garch_model.conditional_volatility
            volatility_adjusted_threshold = entry_threshold * (volatility_forecast / volatility_forecast.mean())
            adaptive_z_score = (spread - mean_spread) / (std_spread * volatility_adjusted_threshold)
            
            adaptive_signals = pd.Series(index=spread.index)
            adaptive_signals[adaptive_z_score < -1] = 1
            adaptive_signals[adaptive_z_score > 1] = -1
            adaptive_signals[(adaptive_z_score > -0.5) & (adaptive_z_score < 0.5)] = 0
            adaptive_signals = adaptive_signals.ffill().fillna(0)
            
            self.trade_signals[(stock1, stock2)] = {
                'z_score': z_score,
                'standard_signals': signals,
                'adaptive_signals': adaptive_signals,
                'volatility_forecast': volatility_forecast
            }
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            ax1.plot(z_score, label='Z-Score', alpha=0.7)
            ax1.plot(signals, label='Trading Signals', drawstyle='steps-post', linewidth=2)
            ax1.axhline(y=entry_threshold, color='r', linestyle='--', label=f'+{entry_threshold} Std')
            ax1.axhline(y=-entry_threshold, color='g', linestyle='--', label=f'-{entry_threshold} Std')
            ax1.axhline(y=exit_threshold, color='y', linestyle=':', label=f'+{exit_threshold} Std')
            ax1.axhline(y=-exit_threshold, color='y', linestyle=':', label=f'-{exit_threshold} Std')
            ax1.set_title('Standard Trading Signals')
            ax1.legend()
            
            ax2.plot(adaptive_z_score, label='Volatility-Adjusted Z-Score', alpha=0.7)
            ax2.plot(adaptive_signals, label='Adaptive Signals', drawstyle='steps-post', linewidth=2)
            ax2.axhline(y=1, color='r', linestyle='--', label='+1 Adaptive Threshold')
            ax2.axhline(y=-1, color='g', linestyle='--', label='-1 Adaptive Threshold')
            ax2.set_title('Volatility-Adjusted Trading Signals')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
            return self.trade_signals[(stock1, stock2)]
        except Exception as e:
            print(f"Error generating signals for {stock1}-{stock2}: {str(e)}")
            return None
    
    def backtest_strategy(self, stock1, stock2, adaptive=True, transaction_cost=0.001):
        if (stock1, stock2) not in self.trade_signals:
            result = self.generate_signals(stock1, stock2)
            if result is None:
                return None
        
        try:
            signal_data = self.trade_signals[(stock1, stock2)]
            signals = signal_data['adaptive_signals'] if adaptive else signal_data['standard_signals']
            
            S1 = self.data[stock1]
            S2 = self.data[stock2]
            S1_returns = S1.pct_change().dropna()
            S2_returns = S2.pct_change().dropna()
            
            signals = signals.shift(1).dropna()  # Use previous day's signal for today's trade
            common_index = signals.index.intersection(S1_returns.index)
            signals = signals.loc[common_index]
            S1_returns = S1_returns.loc[common_index]
            S2_returns = S2_returns.loc[common_index]
            
            # When signal is 1: Long S1, Short S2
            # When signal is -1: Short S1, Long S2
            pair_returns = signals * (S1_returns - S2_returns)
            
            signal_changes = signals.diff().fillna(0)
            transaction_costs = abs(signal_changes) * transaction_cost * 2  # Cost for both assets
            net_returns = pair_returns - transaction_costs            
            cumulative_returns = (1 + net_returns).cumprod() - 1
            
            sharpe_ratio = net_returns.mean() / net_returns.std() * np.sqrt(252)  # Annualized
            max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
            total_return = cumulative_returns.iloc[-1]
            win_rate = (net_returns > 0).mean()            
            self.trade_performance[(stock1, stock2)] = {
                'returns': net_returns,
                'cumulative_returns': cumulative_returns,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': total_return,
                'win_rate': win_rate
            }
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))            
            ax1.plot(cumulative_returns, label='Strategy Returns')
            ax1.set_title(f'Cumulative Returns (Sharpe: {sharpe_ratio:.2f}, Return: {total_return:.2%})')
            ax1.legend()
            ax1.grid(True)
            
            underwater = cumulative_returns - cumulative_returns.cummax()
            ax2.fill_between(underwater.index, underwater.values, 0, color='r', alpha=0.3)
            ax2.set_title(f'Drawdowns (Max: {max_drawdown:.2%})')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            print(f"Performance Summary for {stock1} - {stock2}:")
            print(f"Total Return: {total_return:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Win Rate: {win_rate:.2%}")
            
            return self.trade_performance[(stock1, stock2)]
        except Exception as e:
            print(f"Error backtesting strategy for {stock1}-{stock2}: {str(e)}")
            return None
    
    def analyze_all_pairs(self, max_pairs=None):
        if len(self.cointegrated_pairs) == 0:
            self.find_cointegrated_pairs()
        
        if not self.cointegrated_pairs:
            print("No cointegrated pairs found.")
            return None
        
        pairs_to_analyze = self.cointegrated_pairs
        if max_pairs is not None:
            pairs_to_analyze = pairs_to_analyze[:max_pairs]
        
        print(f"Analyzing {len(pairs_to_analyze)} cointegrated pairs...")
        
        for i, (stock1, stock2, pvalue) in enumerate(pairs_to_analyze):
            print(f"\nAnalyzing pair {i+1}/{len(pairs_to_analyze)}: {stock1} - {stock2} (p-value: {pvalue:.4f})")
            self.backtest_strategy(stock1, stock2)
        
        if self.trade_performance:
            best_pair = max(self.trade_performance.items(), 
                           key=lambda x: x[1]['total_return'] if x[1] is not None else float('-inf'))
            
            stock1, stock2 = best_pair[0]
            performance = best_pair[1]
            
            print("\n" + "="*50)
            print(f"Best performing pair: {stock1} - {stock2}")
            print(f"Total Return: {performance['total_return']:.2%}")
            print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
            print(f"Win Rate: {performance['win_rate']:.2%}")
            print("="*50)
            
            return best_pair
        else:
            print("No pairs could be successfully analyzed.")
            return None
    
    def cross_asset_analysis(self):
        if self.data is None:
            self.fetch_data()
        
        returns = self.data.pct_change().dropna()
        corr_matrix = returns.corr()        
        distance_matrix = 1 - abs(corr_matrix)
        
        from scipy.cluster.hierarchy import linkage, dendrogram
        Z = linkage(distance_matrix, 'ward')
        
        plt.figure(figsize=(16, 10))
        dendrogram(Z, labels=corr_matrix.columns, leaf_rotation=90)
        plt.title('Hierarchical Clustering of Assets')
        plt.xlabel('Assets')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()

        print("Cross-Asset Cointegration Analysis:")
        self.find_cointegrated_pairs()        
        for i, (stock1, stock2, pvalue) in enumerate(self.cointegrated_pairs[:5]):
            print(f"{i+1}. {stock1} - {stock2}: p-value = {pvalue:.4f}")
        
        return self.analyze_all_pairs()
    
    def analyze_paper_pair(self):
        print("Analyzing GDXJ-UNG pair from the paper...")
        if 'GDXJ' not in self.assets or 'UNG' not in self.assets:
            print("GDXJ or UNG not in asset list. Unable to replicate paper analysis.")
            return
        
        # 1. Fit ARIMA-GARCH model
        self.fit_arima_garch('GDXJ', 'UNG', arima_order=(2,0,1), garch_order=(1,1))
        
        # 2. Evaluate forecast performance
        forecast_perf = self.evaluate_forecast_performance('GDXJ', 'UNG', horizons=[1, 5, 20])
        
        # 3. Generate trading signals
        self.generate_signals('GDXJ', 'UNG')
        
        # 4. Backtest the strategy
        performance = self.backtest_strategy('GDXJ', 'UNG', adaptive=True)
        
        return {
            'forecast_performance': forecast_perf,
            'trading_performance': performance
        }

def main():
    assets = [
        # US Treasury bonds of different durations
        'TLT',   # 20+ Year Treasury
        'IEF',   # 7-10 Year Treasury
        'SHY',   # 1-3 Year Treasury
        'SPTL',  # Long-Term Treasury
        
        # Corporate bonds
        'LQD',   # Investment Grade Corporate Bonds
        'HYG',   # High Yield Corporate Bonds
        
        # US Equity indices
        'SPY',   # S&P 500
        'VOO',   # Vanguard S&P 500
        'IVV',   # iShares Core S&P 500
        'QQQ',   # Nasdaq 100
        'DIA',   # Dow Jones Industrial Average
        
        # Sector ETFs
        'XOP',   # Oil & Gas Exploration
        'OIH',   # Oil Services
        
        # Gold and precious metals
        'GLD',   # Gold ETF
        'IAU',   # iShares Gold Trust
        'GDX',   # Gold Miners
        'GDXJ',  # Junior Gold Miners
        'SLV',   # Silver
        
        # Energy commodities
        'USO',   # US Oil Fund
        'UNG',   # US Natural Gas Fund
        
        # Currency pairs
        'UUP',   # US Dollar Bullish
        'FXE',   # Euro
        
        # International markets
        'EWJ',   # Japan
        'EWG',   # Germany
        'EWU',   # United Kingdom
        'EWC'    # Canada
    ]    
    stat_arb = StatisticalArbitrage(assets, start_date='2015-01-01', end_date='2023-01-01')  # Using data range from paper
    data = stat_arb.fetch_data()
    
    # Check if data is valid and has content
    if data is not None and not data.empty:
        # Filter out assets with missing data
        valid_assets = []
        for asset in assets:
            if asset in data.columns and not data[asset].empty:
                valid_assets.append(asset)
        
        if valid_assets:
            plt.figure(figsize=(15, 10))
            for asset in valid_assets:
                # Check if the asset column has data before normalizing
                if len(data[asset]) > 0:
                    plt.plot(data[asset] / data[asset].iloc[0], label=asset)
            plt.title('Normalized Asset Prices')
            plt.legend()
            plt.grid(True)
            plt.show()    
            stat_arb.plot_correlation_heatmap()
            
            pairs = stat_arb.find_cointegrated_pairs(significance_level=0.05)
            print("Cointegrated Pairs:")
            for i, (stock1, stock2, pvalue) in enumerate(pairs):
                print(f"{i+1}. {stock1} - {stock2}: p-value = {pvalue:.4f}")
            
            # First analyze the GDXJ-UNG pair from the paper to replicate section 3.2.4 results
            paper_results = stat_arb.analyze_paper_pair()
            
            # Then continue with the rest of the analysis
            best_pair = stat_arb.analyze_all_pairs(max_pairs=3)
            if best_pair:
                stock1, stock2 = best_pair[0]
                print(f"\nDetailed analysis of best pair: {stock1} - {stock2}")
            
            stat_arb.cross_asset_analysis()
        else:
            print("No valid assets with data found.")
    else:
        print("Failed to fetch data or the data is empty.")

if __name__ == "__main__":
    main()
