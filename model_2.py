import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
    def __init__(self, assets, start_date='2018-01-01', end_date='2023-01-01', output_dir='output_charts'):
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.pairs = []
        self.cointegrated_pairs = []
        self.models = {}
        self.trade_signals = {}
        self.trade_performance = {}
        self.hedge_instruments = ['GLD', 'SLV', 'USO', 'UNG', 'DBC']
        self.max_drawdown_threshold = 0.05
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fetch_data(self):
        all_assets = list(set(self.assets + self.hedge_instruments))
        self.data = yf.download(all_assets, start=self.start_date, end=self.end_date)['Close']
        self.data.dropna(inplace=True)
        return self.data
    
    def save_figure(self, name, fig=None):
        if fig is None:
            plt.savefig(os.path.join(self.output_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
        else:
            fig.savefig(os.path.join(self.output_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
    
    def find_cointegrated_pairs(self, significance_level=0.05):
        n = len(self.assets)
        pvalues = np.zeros((n, n))
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                stock1 = self.assets[i]
                stock2 = self.assets[j]
                if stock1 not in self.data.columns or stock2 not in self.data.columns:
                    continue
                try:
                    result = coint(self.data[stock1], self.data[stock2])
                    pvalue = result[1]
                    pvalues[i, j] = pvalue
                    if pvalue < significance_level:
                        pairs.append((stock1, stock2, pvalue))
                except:
                    continue
        self.cointegrated_pairs = sorted(pairs, key=lambda x: x[2])
        return self.cointegrated_pairs
    
    def plot_correlation_heatmap(self):
        if self.data is None or self.data.empty:
            print("No data available for correlation analysis.")
            return None
        returns = self.data.pct_change().dropna()
        corr = returns.corr()
        valid_assets = [asset for asset in self.assets if asset in corr.columns]
        if not valid_assets:
            print("No valid assets for correlation analysis.")
            return None
        corr = corr.loc[valid_assets, valid_assets]
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', annot_kws={"size": 8})
        plt.title('Correlation Heatmap of Asset Returns')
        plt.tight_layout()
        self.save_figure('correlation_heatmap')
        plt.close()
        return corr
    
    def find_optimal_hedge(self, returns_series):
        if returns_series.empty:
            return None
        hedge_correlations = {}
        for hedge in self.hedge_instruments:
            if hedge in self.data.columns:
                hedge_returns = self.data[hedge].pct_change().dropna()
                common_idx = returns_series.index.intersection(hedge_returns.index)
                if len(common_idx) > 0:
                    strategy_returns = returns_series.loc[common_idx]
                    hedge_ret = hedge_returns.loc[common_idx]
                    correlation = strategy_returns.corr(hedge_ret)
                    hedge_correlations[hedge] = correlation
        if not hedge_correlations:
            return None
        best_hedge = min(hedge_correlations.items(), key=lambda x: x[1])
        return best_hedge[0]
    
    def analyze_pair(self, stock1, stock2):
        if self.data is None or self.data.empty:
            print("No data available. Please fetch data first.")
            return None
        if stock1 not in self.data.columns or stock2 not in self.data.columns:
            print(f"One or both assets ({stock1}, {stock2}) not found in data.")
            return None
        if len(self.cointegrated_pairs) == 0:
            self.find_cointegrated_pairs()
        S1 = self.data[stock1]
        S2 = self.data[stock2]
        from statsmodels.regression.linear_model import OLS
        model = OLS(S1, S2).fit()
        hedge_ratio = model.params[0]
        spread = S1 - hedge_ratio * S2
        if spread.std() < 1e-10:
            print(f"Warning: Spread between {stock1} and {stock2} has near-zero variance.")
            spread = spread + np.random.normal(0, 1e-8, len(spread))
        try:
            adf_result = adfuller(spread)
            adf_pvalue = adf_result[1]
        except ValueError as e:
            print(f"ADF Test failed: {e}")
            adf_pvalue = 1.0
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        if not S1.empty and not S2.empty:
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
            self.save_figure(f"pair_analysis_{stock1}_{stock2}")
            plt.close(fig)
        return spread
    
    def fit_arima_garch(self, stock1, stock2, arima_order=(1,1,1), garch_order=(1,1)):
        try:
            spread = self.analyze_pair(stock1, stock2)
            if spread is None or spread.empty:
                print(f"Could not generate spread for {stock1}-{stock2}")
                return None
            arima_model = ARIMA(spread, order=arima_order)
            arima_results = arima_model.fit()
            garch_model = arch_model(arima_results.resid, vol='GARCH', p=garch_order[0], q=garch_order[1])
            garch_results = garch_model.fit(disp='off')
            self.models[(stock1, stock2)] = {
                'spread': spread,
                'arima': arima_results,
                'garch': garch_results,
                'hedge_ratio': arima_results.params[0] if len(arima_results.params) > 0 else 1.0
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
            self.save_figure(f"arima_garch_{stock1}_{stock2}")
            plt.close(fig)
            f, Pxx = periodogram(spread, detrend='linear')
            plt.figure(figsize=(12, 6))
            plt.semilogy(f, Pxx)
            plt.title('Periodogram for Seasonality Detection')
            plt.xlabel('Frequency')
            plt.ylabel('Power Spectral Density')
            plt.grid(True)
            self.save_figure(f"periodogram_{stock1}_{stock2}")
            plt.close()
            return self.models[(stock1, stock2)]
        except Exception as e:
            print(f"Error fitting ARIMA-GARCH for {stock1}-{stock2}: {str(e)}")
            return None
    
    def generate_signals(self, stock1, stock2, entry_threshold=2.0, exit_threshold=0.5, mean_reversion_weight=0.6, trend_weight=0.2, vol_weight=0.2):
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
            mr_signals = pd.Series(index=spread.index)
            mr_signals[z_score < -entry_threshold] = 1
            mr_signals[z_score > entry_threshold] = -1
            mr_signals[(z_score > -exit_threshold) & (z_score < exit_threshold)] = 0
            mr_signals = mr_signals.ffill().fillna(0)
            short_ma = spread.rolling(window=5).mean()
            long_ma = spread.rolling(window=20).mean()
            trend_signals = pd.Series(index=spread.index)
            trend_signals[short_ma > long_ma] = 1
            trend_signals[short_ma < long_ma] = -1
            trend_signals = trend_signals.fillna(0)
            volatility_forecast = garch_model.conditional_volatility
            volatility_adjusted_threshold = entry_threshold * (volatility_forecast / volatility_forecast.mean())
            vol_signals = pd.Series(index=spread.index)
            vol_ratio = volatility_forecast / volatility_forecast.rolling(window=20).mean()
            vol_signals[vol_ratio > 1.5] = -1
            vol_signals[vol_ratio < 0.8] = 1
            vol_signals = vol_signals.fillna(0)
            combined_signals = (mean_reversion_weight * mr_signals + 
                               trend_weight * trend_signals + 
                               vol_weight * vol_signals)
            final_signals = pd.Series(index=spread.index)
            final_signals[combined_signals > 0.5] = 1
            final_signals[combined_signals < -0.5] = -1
            final_signals = final_signals.fillna(0).ffill()
            signal_changes = final_signals.diff().abs()
            whipsaw_filter = signal_changes.rolling(window=3).sum()
            filtered_signals = final_signals.copy()
            filtered_signals[whipsaw_filter > 2] = 0
            self.trade_signals[(stock1, stock2)] = {
                'z_score': z_score,
                'standard_signals': mr_signals,
                'trend_signals': trend_signals,
                'volatility_signals': vol_signals,
                'final_signals': filtered_signals,
                'volatility_forecast': volatility_forecast
            }
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            ax1.plot(z_score, label='Z-Score', alpha=0.7)
            ax1.plot(mr_signals, label='Mean-Reversion Signals', drawstyle='steps-post', linewidth=1.5)
            ax1.plot(trend_signals, label='Trend Signals', drawstyle='steps-post', linewidth=1.5, alpha=0.6)
            ax1.axhline(y=entry_threshold, color='r', linestyle='--', label=f'+{entry_threshold} Std')
            ax1.axhline(y=-entry_threshold, color='g', linestyle='--', label=f'-{entry_threshold} Std')
            ax1.set_title('Signal Components')
            ax1.legend()
            ax2.plot(z_score, label='Z-Score', alpha=0.5)
            ax2.plot(filtered_signals, label='Final Filtered Signals', drawstyle='steps-post', linewidth=2)
            ax2.set_title('Final Trading Signals')
            ax2.legend()
            plt.tight_layout()
            self.save_figure(f"signals_{stock1}_{stock2}")
            plt.close(fig)
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
            signals = signal_data['final_signals']
            S1 = self.data[stock1]
            S2 = self.data[stock2]
            S1_returns = S1.pct_change().dropna()
            S2_returns = S2.pct_change().dropna()
            signals = signals.shift(1).dropna()
            common_index = signals.index.intersection(S1_returns.index)
            signals = signals.loc[common_index]
            S1_returns = S1_returns.loc[common_index]
            S2_returns = S2_returns.loc[common_index]
            pair_returns = signals * (S1_returns - S2_returns)
            signal_changes = signals.diff().fillna(0)
            transaction_costs = abs(signal_changes) * transaction_cost * 2
            net_returns = pair_returns - transaction_costs
            cumulative_returns = (1 + net_returns).cumprod() - 1
            drawdowns = cumulative_returns - cumulative_returns.cummax()
            stop_loss_triggered = pd.Series(False, index=drawdowns.index)
            stop_loss_triggered[drawdowns < -self.max_drawdown_threshold] = True
            in_recovery = pd.Series(False, index=drawdowns.index)
            for i in range(1, len(stop_loss_triggered)):
                if stop_loss_triggered.iloc[i-1] and drawdowns.iloc[i] < -0.02:
                    stop_loss_triggered.iloc[i] = True
            if stop_loss_triggered.any():
                best_hedge = self.find_optimal_hedge(net_returns)
                if best_hedge and best_hedge in self.data.columns:
                    hedge_returns = self.data[best_hedge].pct_change().dropna()
                    hedge_returns = hedge_returns.loc[common_index]
                    for i in range(len(net_returns)):
                        if stop_loss_triggered.iloc[i] and i < len(hedge_returns):
                            net_returns.iloc[i] = hedge_returns.iloc[i]
            hedged_cumulative_returns = (1 + net_returns).cumprod() - 1
            sharpe_ratio = net_returns.mean() / net_returns.std() * np.sqrt(252)
            max_drawdown = (hedged_cumulative_returns.cummax() - hedged_cumulative_returns).max()
            total_return = hedged_cumulative_returns.iloc[-1]
            win_rate = (net_returns > 0).mean()
            self.trade_performance[(stock1, stock2)] = {
                'returns': net_returns,
                'cumulative_returns': hedged_cumulative_returns,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': total_return,
                'win_rate': win_rate,
                'stop_loss_events': stop_loss_triggered.sum()
            }
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16))
            ax1.plot(cumulative_returns, label='Strategy Returns (No Hedge)', alpha=0.7)
            ax1.plot(hedged_cumulative_returns, label='Strategy Returns (With Hedge)', linewidth=2)
            ax1.set_title(f'Cumulative Returns (Sharpe: {sharpe_ratio:.2f}, Return: {total_return:.2%})')
            ax1.legend()
            ax1.grid(True)
            ax2.plot(cumulative_returns, label='Cumulative Returns', alpha=0.7)
            ax2.scatter(stop_loss_triggered[stop_loss_triggered].index, 
                       cumulative_returns[stop_loss_triggered], 
                       color='red', marker='v', s=100, label='Stop Loss Triggered')
            ax2.axhline(y=-self.max_drawdown_threshold, color='r', linestyle='--', 
                       label=f'Stop Loss Level (-{self.max_drawdown_threshold:.0%})')
            ax2.set_title(f'Stop Loss Events (Total: {stop_loss_triggered.sum()})')
            ax2.legend()
            ax2.grid(True)
            underwater = hedged_cumulative_returns - hedged_cumulative_returns.cummax()
            ax3.fill_between(underwater.index, underwater.values, 0, color='r', alpha=0.3)
            ax3.set_title(f'Drawdowns (Max: {max_drawdown:.2%})')
            ax3.grid(True)
            plt.tight_layout()
            self.save_figure(f"backtest_{stock1}_{stock2}")
            plt.close(fig)
            perf = self.trade_performance[(stock1, stock2)]
            summary = pd.DataFrame({
                'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Stop Loss Events'],
                'Value': [f"{perf['total_return']:.2%}", 
                         f"{perf['sharpe_ratio']:.2f}", 
                         f"{perf['max_drawdown']:.2%}", 
                         f"{perf['win_rate']:.2%}",
                         f"{perf['stop_loss_events']}"]
            })
            summary.to_csv(os.path.join(self.output_dir, f"performance_{stock1}_{stock2}.csv"), index=False)
            print(f"Performance Summary for {stock1} - {stock2}:")
            print(f"Total Return: {total_return:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Stop Loss Events: {stop_loss_triggered.sum()}")
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
        performance_summary = []
        for i, (stock1, stock2, pvalue) in enumerate(pairs_to_analyze):
            print(f"\nAnalyzing pair {i+1}/{len(pairs_to_analyze)}: {stock1} - {stock2} (p-value: {pvalue:.4f})")
            perf = self.backtest_strategy(stock1, stock2)
            if perf:
                performance_summary.append({
                    'Pair': f"{stock1}-{stock2}",
                    'P-Value': pvalue,
                    'Total Return': perf['total_return'],
                    'Sharpe Ratio': perf['sharpe_ratio'],
                    'Max Drawdown': perf['max_drawdown'],
                    'Win Rate': perf['win_rate'],
                    'Stop Loss Events': perf.get('stop_loss_events', 0)
                })
        if performance_summary:
            perf_df = pd.DataFrame(performance_summary)
            perf_df = perf_df.sort_values('Sharpe Ratio', ascending=False)
            perf_df.to_csv(os.path.join(self.output_dir, "all_pairs_performance.csv"), index=False)
            best_pair = max(self.trade_performance.items(), 
                           key=lambda x: x[1]['sharpe_ratio'] if x[1] is not None else float('-inf'))
            stock1, stock2 = best_pair[0]
            performance = best_pair[1]
            print("\n" + "="*50)
            print(f"Best performing pair: {stock1} - {stock2}")
            print(f"Total Return: {performance['total_return']:.2%}")
            print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
            print(f"Win Rate: {performance['win_rate']:.2%}")
            print(f"Stop Loss Events: {performance.get('stop_loss_events', 0)}")
            print("="*50)
            return best_pair
        else:
            print("No pairs could be successfully analyzed.")
            return None
    
    def cross_asset_analysis(self):
        if self.data is None:
            self.fetch_data()
        if self.data is None or self.data.empty:
            print("Failed to fetch data for cross-asset analysis.")
            return None
        valid_assets = [asset for asset in self.assets if asset in self.data.columns]
        if not valid_assets:
            print("No valid assets for cross-asset analysis.")
            return None
        returns = self.data[valid_assets].pct_change().dropna()
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
        self.save_figure("hierarchical_clustering")
        plt.close()
        corr_matrix.to_csv(os.path.join(self.output_dir, "correlation_matrix.csv"))
        print("Cross-Asset Cointegration Analysis:")
        self.find_cointegrated_pairs()
        if self.cointegrated_pairs:
            for i, (stock1, stock2, pvalue) in enumerate(self.cointegrated_pairs[:5]):
                print(f"{i+1}. {stock1} - {stock2}: p-value = {pvalue:.4f}")
            coint_df = pd.DataFrame(self.cointegrated_pairs, columns=['Asset1', 'Asset2', 'P-Value'])
            coint_df.to_csv(os.path.join(self.output_dir, "cointegrated_pairs.csv"), index=False)
        else:
            print("No cointegrated pairs found.")
        return self.analyze_all_pairs()

def main():
    # Define the asset list
    assets = [
        # US Treasury bonds of different durations
        'TLT',   # 20+ Year Treasury
        'IEF',   # 7-10 Year Treasury
        'SHY',   # 1-3 Year Treasury
        'SPTL',  # Long-Term Treasury
        'VGIT',  # Intermediate-Term Treasury
        
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
        'XLE',   # Energy
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
        'BNO',   # United States Brent Oil Fund
        
        # Currency pairs
        'UUP',   # US Dollar Bullish
        'FXE',   # Euro
        'FXY',   # Japanese Yen
        
        # International markets
        'EWJ',   # Japan
        'EWG',   # Germany
        'EWU',   # United Kingdom
        'EWC'    # Canada
    ]
    
    # Create output directory
    output_dir = 'stat_arb_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize statistical arbitrage with output directory
    stat_arb = StatisticalArbitrage(assets, output_dir=output_dir)
    
    print("Fetching data...")
    data = stat_arb.fetch_data()
    
    # Check if data is valid and has content
    if data is not None and not data.empty:
        valid_assets = []
        for asset in assets:
            if asset in data.columns and not data[asset].empty and len(data[asset]) > 0:
                valid_assets.append(asset)
        
        if valid_assets:
            print(f"Found {len(valid_assets)} valid assets out of {len(assets)}")
            
            plt.figure(figsize=(15, 10))
            for asset in valid_assets:
                plt.plot(data[asset] / data[asset].iloc[0], label=asset)
            plt.title('Normalized Asset Prices')
            plt.legend(loc='upper left', fontsize='small')
            plt.grid(True)
            stat_arb.save_figure("normalized_prices")
            plt.close()
            
            print("Generating correlation heatmap...")
            stat_arb.plot_correlation_heatmap()
            
            print("Finding cointegrated pairs...")
            pairs = stat_arb.find_cointegrated_pairs(significance_level=0.05)
            
            if pairs:
                print("Top Cointegrated Pairs:")
                for i, (stock1, stock2, pvalue) in enumerate(pairs[:10]):
                    print(f"{i+1}. {stock1} - {stock2}: p-value = {pvalue:.4f}")
                
                pairs_df = pd.DataFrame(pairs, columns=['Asset1', 'Asset2', 'P-Value'])
                pairs_df.to_csv(os.path.join(output_dir, "cointegrated_pairs.csv"), index=False)
                
                print("\nAnalyzing top pairs...")
                best_pair = stat_arb.analyze_all_pairs(max_pairs=5)
                
                if best_pair:
                    stock1, stock2 = best_pair[0]
                    print(f"\nPerforming detailed analysis of best pair: {stock1} - {stock2}")
                    stat_arb.generate_signals(stock1, stock2, entry_threshold=1.5, exit_threshold=0.5)
                
                print("\nPerforming cross-asset analysis...")
                stat_arb.cross_asset_analysis()
                
                print(f"\nAll analysis complete. Results saved to '{output_dir}' directory.")
            else:
                print("No cointegrated pairs found.")
        else:
            print("No valid assets with data found.")
    else:
        print("Failed to fetch data or the data is empty.")

if __name__ == "__main__":
    main()


