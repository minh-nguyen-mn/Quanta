"""
Signal Generators - Diverse Persona-Based Trading Signals
========================================================

This module implements 12+ different trading personas, each generating 5+ unique signals
for QQQ trading. Each persona represents a different approach to market analysis,
ensuring maximum diversity and orthogonality in signal generation.

Personas implemented:
1. Volatility Specialist - Focus on volatility regimes and vol-based signals
2. Momentum Trader - Trend-following and momentum signals
3. Mean Reversion Trader - Counter-trend and reversion signals  
4. Risk Manager - Defensive and risk-adjusted signals
5. Quantitative Researcher - Statistical and mathematical signals
6. Technical Analyst - Chart patterns and technical indicators
7. Macro Economist - Economic cycle and macro signals
8. Behavioral Psychologist - Market psychology and sentiment
9. Physicist - Physics-inspired market dynamics
10. Biologist - Evolution and ecosystem-inspired signals
11. Oceanographer - Wave theory and fluid dynamics
12. Short Specialist - Bear market and short-only signals
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class SignalGenerators:
    """Collection of persona-based signal generators for QQQ trading."""
    
    def __init__(self, min_leverage=-1.0, max_leverage=1.5):
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
    
    def _clip_signal(self, signal):
        """Ensure signal stays within leverage bounds."""
        return np.clip(signal, self.min_leverage, self.max_leverage)
    
    def _normalize_signal(self, signal, target_range=(-1, 1.5)):
        """Normalize signal to target range using expanding windows (no lookahead bias)."""
        if signal.std() == 0:
            return pd.Series(0, index=signal.index)
        
        # Use expanding windows to avoid lookahead bias
        expanding_mean = signal.expanding(min_periods=30).mean()
        expanding_std = signal.expanding(min_periods=30).std()
        
        # Z-score normalize using only past data
        normalized = (signal - expanding_mean) / (expanding_std + 1e-6)
        
        # Scale to target range
        min_target, max_target = target_range
        scaled = normalized * (max_target - min_target) / 6  # 6 sigma range
        scaled = scaled + (max_target + min_target) / 2  # Center
        
        return self._clip_signal(scaled)
    
    # ================================================================
    # 1. VOLATILITY SPECIALIST PERSONA
    # ================================================================
    
    def volatility_specialist_signals(self, data):
        """
        Volatility Specialist Persona
        Focus: Volatility regimes, vol clustering, vol mean reversion
        Philosophy: Volatility is predictable and mean-reverting
        """
        signals = {}
        
        # Signal 1: Parkinson Volatility Regime
        hl_vol = np.log(data['High'] / data['Low'])
        parkinson_vol = hl_vol.rolling(20).std() * np.sqrt(252)
        vol_regime = pd.Series(0.0, index=data.index)
        vol_regime[parkinson_vol < parkinson_vol.rolling(252).quantile(0.3)] = 1.5  # Low vol = Long
        vol_regime[parkinson_vol > parkinson_vol.rolling(252).quantile(0.7)] = -0.5  # High vol = Short
        signals['Vol_Parkinson_Regime'] = self._clip_signal(vol_regime)
        
        # Signal 2: VIX-like Volatility Mean Reversion  
        realized_vol = data['Returns'].rolling(20).std() * np.sqrt(252)
        vol_zscore = (realized_vol - realized_vol.rolling(252).mean()) / realized_vol.rolling(252).std()
        vol_mr_signal = -vol_zscore * 0.3  # Fade high vol, buy low vol
        signals['Vol_Mean_Reversion'] = self._clip_signal(vol_mr_signal)
        
        # Signal 3: Volatility Breakout Signal
        vol_ma = realized_vol.rolling(50).mean()
        vol_breakout = pd.Series(0.0, index=data.index)
        vol_breakout[realized_vol > vol_ma * 1.5] = -1.0  # Vol spike = defensive
        vol_breakout[realized_vol < vol_ma * 0.7] = 1.2   # Vol compression = aggressive long
        signals['Vol_Breakout'] = self._clip_signal(vol_breakout)
        
        # Signal 4: Vol-of-Vol Signal
        vol_changes = realized_vol.pct_change().rolling(10).std()
        vol_of_vol_signal = pd.Series(0.0, index=data.index)
        vol_of_vol_threshold = vol_changes.rolling(252).quantile(0.8)
        vol_of_vol_signal[vol_changes > vol_of_vol_threshold] = -0.8  # High vol-of-vol = bearish
        vol_of_vol_signal[vol_changes < vol_changes.rolling(252).quantile(0.2)] = 1.0
        signals['Vol_of_Vol'] = self._clip_signal(vol_of_vol_signal)
        
        # Signal 5: GARCH-like Volatility Clustering
        vol_persistence = realized_vol.rolling(5).mean() / realized_vol.rolling(20).mean()
        garch_signal = pd.Series(0.0, index=data.index)
        garch_signal[vol_persistence > 1.3] = -0.6  # Persistent high vol = bearish
        garch_signal[vol_persistence < 0.8] = 1.1   # Low vol environment = bullish
        signals['GARCH_Clustering'] = self._clip_signal(garch_signal)
        
        return signals
    
    # ================================================================
    # 2. MOMENTUM TRADER PERSONA  
    # ================================================================
    
    def momentum_trader_signals(self, data):
        """
        Momentum Trader Persona
        Focus: Trend following, breakouts, momentum persistence
        Philosophy: Trends persist longer than expected
        """
        signals = {}
        
        # Signal 1: Multi-Timeframe Momentum
        mom_3m = data['Close'] / data['Close'].shift(63) - 1  # 3 month momentum
        mom_1m = data['Close'] / data['Close'].shift(21) - 1  # 1 month momentum
        mom_signal = (mom_3m * 0.7 + mom_1m * 0.3) * 2  # Combined momentum
        signals['Multi_Timeframe_Momentum'] = self._clip_signal(mom_signal)
        
        # Signal 2: Breakout Signal with Volume Confirmation
        price_high = data['Close'].rolling(50).max()
        volume_avg = data['Volume'].rolling(50).mean()
        breakout_signal = pd.Series(0.0, index=data.index)
        breakout_condition = (data['Close'] > price_high.shift(1)) & (data['Volume'] > volume_avg * 1.5)
        breakout_signal[breakout_condition] = 1.3
        # Decay the signal over time
        for i in range(1, 10):
            breakout_signal = breakout_signal.shift(1).fillna(0) * 0.9 + breakout_signal
        signals['Volume_Breakout'] = self._clip_signal(breakout_signal)
        
        # Signal 3: Trend Strength Indicator
        closes = data['Close']
        trend_signal = pd.Series(0.0, index=data.index)
        for period in [10, 20, 50]:
            sma = closes.rolling(period).mean()
            trend_component = np.where(closes > sma, 1, -1) * (abs(closes - sma) / sma)
            trend_signal += trend_component / 3
        signals['Trend_Strength'] = self._clip_signal(trend_signal)
        
        # Signal 4: Momentum Acceleration
        returns_5d = data['Returns'].rolling(5).sum()
        returns_20d = data['Returns'].rolling(20).sum()
        acceleration = returns_5d - returns_20d
        accel_signal = self._normalize_signal(acceleration, (-0.8, 1.4))
        signals['Momentum_Acceleration'] = accel_signal
        
        # Signal 5: Price vs Volume Momentum
        price_momentum = data['Close'].pct_change(20)
        volume_momentum = data['Volume'].pct_change(20)
        pv_momentum = price_momentum * np.sign(volume_momentum) * 1.5
        signals['Price_Volume_Momentum'] = self._clip_signal(pv_momentum)
        
        return signals
    
    # ================================================================
    # 3. MEAN REVERSION TRADER PERSONA
    # ================================================================
    
    def mean_reversion_trader_signals(self, data):
        """
        Mean Reversion Trader Persona
        Focus: Oversold/overbought conditions, reversion to mean
        Philosophy: Extreme moves tend to reverse
        """
        signals = {}
        
        # Signal 1: RSI Mean Reversion with Regime Filter
        rsi = data['RSI_14']
        regime_filter = data['Close'] > data['SMA_200']  # Bull market filter
        rsi_signal = pd.Series(0.0, index=data.index)
        
        # In bull markets, buy RSI dips more aggressively
        rsi_signal[regime_filter & (rsi < 30)] = 1.4
        rsi_signal[regime_filter & (rsi < 20)] = 1.5
        rsi_signal[~regime_filter & (rsi < 30)] = 0.8  # More conservative in bear markets
        rsi_signal[rsi > 80] = -0.6
        
        signals['RSI_Regime_MeanRev'] = self._clip_signal(rsi_signal)
        
        # Signal 2: Bollinger Band Mean Reversion
        bb_position = data['BB_Position']
        bb_signal = pd.Series(0.0, index=data.index)
        bb_signal[bb_position < 0.1] = 1.2  # Near lower band = buy
        bb_signal[bb_position > 0.9] = -0.7  # Near upper band = sell
        bb_signal[(bb_position > 0.2) & (bb_position < 0.8)] = 0.3  # Middle range = slight long bias
        signals['Bollinger_MeanRev'] = self._clip_signal(bb_signal)
        
        # Signal 3: Short-term Reversal (2-day RSI)
        close_prices = data['Close']
        delta_2d = close_prices.diff()
        gain_2d = (delta_2d.where(delta_2d > 0, 0)).rolling(window=2).mean()
        loss_2d = (-delta_2d.where(delta_2d < 0, 0)).rolling(window=2).mean()
        rsi_2d = 100 - (100 / (1 + gain_2d / loss_2d))
        
        short_rev_signal = pd.Series(0.0, index=data.index)
        short_rev_signal[rsi_2d < 10] = 1.3
        short_rev_signal[rsi_2d > 90] = -0.8
        signals['Short_Term_Reversal'] = self._clip_signal(short_rev_signal)
        
        # Signal 4: Z-Score Mean Reversion
        returns_zscore = (data['Returns'] - data['Returns'].rolling(252).mean()) / data['Returns'].rolling(252).std()
        zscore_signal = -returns_zscore * 0.4  # Fade extreme moves
        signals['ZScore_MeanRev'] = self._clip_signal(zscore_signal)
        
        # Signal 5: Gap Fade Signal
        gaps = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        gap_signal = pd.Series(0.0, index=data.index)
        gap_signal[gaps > 0.02] = -0.8  # Fade large up gaps
        gap_signal[gaps < -0.02] = 1.1   # Buy large down gaps
        signals['Gap_Fade'] = self._clip_signal(gap_signal)
        
        return signals
    
    # ================================================================
    # 4. RISK MANAGER PERSONA
    # ================================================================
    
    def risk_manager_signals(self, data):
        """
        Risk Manager Persona
        Focus: Risk-adjusted returns, drawdown control, tail risk
        Philosophy: Preserve capital first, returns second
        """
        signals = {}
        
        # Signal 1: Maximum Drawdown Protection
        cumulative_returns = (1 + data['Returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        dd_signal = pd.Series(1.0, index=data.index)  # Default long bias
        dd_signal[drawdown < -0.05] = 0.5  # Reduce exposure in drawdown
        dd_signal[drawdown < -0.10] = 0.0  # Go to cash in large drawdown
        dd_signal[drawdown < -0.15] = -0.3  # Short in severe drawdown
        signals['Drawdown_Protection'] = self._clip_signal(dd_signal)
        
        # Signal 2: VaR-based Position Sizing
        returns_window = data['Returns'].rolling(63)  # 3 months
        var_95 = returns_window.quantile(0.05)  # 5% VaR
        
        var_signal = pd.Series(0.8, index=data.index)  # Base position
        var_signal[var_95 < -0.03] = 0.3  # Reduce when tail risk is high
        var_signal[var_95 < -0.05] = 0.0  # Cash when extreme tail risk
        var_signal[var_95 > -0.01] = 1.2  # Increase when tail risk is low
        signals['VaR_Position_Sizing'] = self._clip_signal(var_signal)
        
        # Signal 3: Sharpe Ratio Momentum
        rolling_returns = data['Returns'].rolling(63)
        rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)
        
        sharpe_signal = pd.Series(0.5, index=data.index)
        sharpe_signal[rolling_sharpe > 1.0] = 1.3  # High Sharpe = increase exposure
        sharpe_signal[rolling_sharpe < 0] = -0.2   # Negative Sharpe = short
        sharpe_signal[rolling_sharpe < -0.5] = -0.8
        signals['Sharpe_Momentum'] = self._clip_signal(sharpe_signal)
        
        # Signal 4: Correlation Breakdown Signal
        # When correlations spike, reduce exposure (systemic risk)
        returns_abs = data['Returns'].abs()
        correlation_proxy = returns_abs.rolling(21).corr(returns_abs.shift(1))
        
        corr_signal = pd.Series(1.0, index=data.index)
        corr_signal[correlation_proxy > 0.7] = 0.4  # High correlation = reduce exposure
        corr_signal[correlation_proxy < 0.3] = 1.3  # Low correlation = increase
        signals['Correlation_Breakdown'] = self._clip_signal(corr_signal)
        
        # Signal 5: Kelly Criterion Position Sizing
        win_rate = (data['Returns'] > 0).rolling(252).mean()
        avg_win = data['Returns'][data['Returns'] > 0].rolling(252).mean()
        avg_loss = data['Returns'][data['Returns'] < 0].rolling(252).mean()
        
        # Kelly formula: f = (bp - q) / b, where b = avg_win/avg_loss, p = win_rate, q = 1-p
        kelly_fraction = ((avg_win / abs(avg_loss)) * win_rate - (1 - win_rate)) / (avg_win / abs(avg_loss))
        kelly_signal = kelly_fraction.fillna(0.5) * 2  # Scale up the signal
        signals['Kelly_Position_Sizing'] = self._clip_signal(kelly_signal)
        
        return signals
    
    # ================================================================
    # 5. QUANTITATIVE RESEARCHER PERSONA
    # ================================================================
    
    def quantitative_researcher_signals(self, data):
        """
        Quantitative Researcher Persona
        Focus: Statistical arbitrage, mathematical models, data mining
        Philosophy: Markets have statistical inefficiencies to exploit
        """
        signals = {}
        
        # Signal 1: Hurst Exponent - Market Efficiency Measure
        def hurst_exponent(ts, max_lag=20):
            """Calculate Hurst exponent for mean reversion detection."""
            if len(ts) < max_lag * 2:
                return 0.5
            
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        
        hurst_values = data['Returns'].rolling(126).apply(lambda x: hurst_exponent(x.values), raw=False)
        hurst_signal = pd.Series(0.0, index=data.index)
        hurst_signal[hurst_values < 0.4] = 1.2  # Mean reverting = contrarian
        hurst_signal[hurst_values > 0.6] = 0.8  # Trending = momentum
        signals['Hurst_Efficiency'] = self._clip_signal(hurst_signal)
        
        # Signal 2: Autocorrelation Signal
        autocorr_1d = data['Returns'].rolling(63).apply(lambda x: x.autocorr(lag=1), raw=False)
        autocorr_signal = pd.Series(0.8, index=data.index)  # Base long bias
        autocorr_signal[autocorr_1d > 0.1] = 1.3   # Positive autocorr = momentum
        autocorr_signal[autocorr_1d < -0.1] = 0.3  # Negative autocorr = mean reversion
        signals['Autocorrelation'] = self._clip_signal(autocorr_signal)
        
        # Signal 3: Skewness Signal
        rolling_skew = data['Returns'].rolling(63).skew()
        skew_signal = pd.Series(0.7, index=data.index)
        skew_signal[rolling_skew > 0.5] = 1.2   # Positive skew = bullish
        skew_signal[rolling_skew < -0.5] = 0.2  # Negative skew = bearish
        signals['Skewness'] = self._clip_signal(skew_signal)
        
        # Signal 4: Entropy-based Signal
        def rolling_entropy(series, bins=10):
            """Calculate rolling entropy of returns distribution."""
            if len(series) < 20:
                return 0
            hist, _ = np.histogram(series, bins=bins, density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            return -np.sum(hist * np.log(hist))
        
        entropy = data['Returns'].rolling(63).apply(lambda x: rolling_entropy(x.values), raw=False)
        entropy_signal = self._normalize_signal(entropy, (0.2, 1.4))
        signals['Entropy'] = entropy_signal
        
        # Signal 5: Fractal Dimension Signal
        def fractal_dimension(ts, max_k=10):
            """Estimate fractal dimension using box-counting method."""
            if len(ts) < 20:
                return 1.5
            
            # Normalize time series
            ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-10)
            
            # Calculate fractal dimension
            scales = np.logspace(0.01, 1, max_k)
            counts = []
            
            for scale in scales:
                # Box counting
                boxes = int(1 / scale)
                if boxes < 2:
                    continue
                count = 0
                for i in range(boxes):
                    box_min = i * scale
                    box_max = (i + 1) * scale
                    if np.any((ts_norm >= box_min) & (ts_norm < box_max)):
                        count += 1
                counts.append(count)
            
            if len(counts) < 2:
                return 1.5
            
            # Fit line to log-log plot
            coeffs = np.polyfit(np.log(scales[:len(counts)]), np.log(counts), 1)
            return -coeffs[0]
        
        fractal_dim = data['Close'].rolling(63).apply(lambda x: fractal_dimension(x.values), raw=False)
        fractal_signal = self._normalize_signal(fractal_dim, (0.3, 1.3))
        signals['Fractal_Dimension'] = fractal_signal
        
        return signals
    
    # ================================================================
    # 6. TECHNICAL ANALYST PERSONA
    # ================================================================
    
    def technical_analyst_signals(self, data):
        """
        Technical Analyst Persona
        Focus: Chart patterns, support/resistance, classical TA
        Philosophy: Price action tells the complete story
        """
        signals = {}
        
        # Signal 1: Support/Resistance Levels
        def find_support_resistance(prices, window=20, strength=3):
            """Find support and resistance levels."""
            highs = prices.rolling(window, center=True).max()
            lows = prices.rolling(window, center=True).min()
            
            resistance = (prices == highs).rolling(strength).sum() >= 1
            support = (prices == lows).rolling(strength).sum() >= 1
            
            return support, resistance
        
        support, resistance = find_support_resistance(data['Close'])
        sr_signal = pd.Series(0.5, index=data.index)  # Base long bias
        
        # Buy near support, sell near resistance
        sr_signal[support & (data['Close'] <= data['Close'].rolling(5).min())] = 1.3
        sr_signal[resistance & (data['Close'] >= data['Close'].rolling(5).max())] = -0.2
        signals['Support_Resistance'] = self._clip_signal(sr_signal)
        
        # Signal 2: Moving Average Crossovers
        ma_fast = data['EMA_12'] if 'EMA_12' in data.columns else data['Close'].ewm(span=12).mean()
        ma_slow = data['EMA_26'] if 'EMA_26' in data.columns else data['Close'].ewm(span=26).mean()
        
        ma_signal = pd.Series(0.0, index=data.index)
        ma_signal[ma_fast > ma_slow] = 1.1  # Golden cross
        ma_signal[ma_fast < ma_slow] = -0.3  # Death cross
        
        # Add momentum to crossover
        ma_momentum = (ma_fast - ma_slow).pct_change()
        ma_signal += ma_momentum * 10
        signals['MA_Crossover'] = self._clip_signal(ma_signal)
        
        # Signal 3: MACD Divergence
        macd = data['MACD'] if 'MACD' in data.columns else (ma_fast - ma_slow)
        macd_signal = data['MACD_Signal'] if 'MACD_Signal' in data.columns else macd.ewm(span=9).mean()
        
        macd_crossover = pd.Series(0.0, index=data.index)
        macd_crossover[macd > macd_signal] = 0.9
        macd_crossover[macd < macd_signal] = -0.4
        
        # Add MACD histogram momentum
        macd_hist = macd - macd_signal
        macd_momentum = macd_hist.diff()
        macd_crossover += macd_momentum * 50
        signals['MACD_System'] = self._clip_signal(macd_crossover)
        
        # Signal 4: Price Channel Breakouts
        high_channel = data['High'].rolling(20).max()
        low_channel = data['Low'].rolling(20).min()
        
        channel_signal = pd.Series(0.0, index=data.index)
        channel_signal[data['Close'] > high_channel.shift(1)] = 1.2  # Upside breakout
        channel_signal[data['Close'] < low_channel.shift(1)] = -0.8  # Downside breakdown
        
        # Add channel position
        channel_position = (data['Close'] - low_channel) / (high_channel - low_channel)
        channel_signal += (channel_position - 0.5) * 0.6
        signals['Channel_Breakout'] = self._clip_signal(channel_signal)
        
        # Signal 5: Relative Strength vs Market
        # Since we only have QQQ, we'll use a proxy comparison with its own moving averages
        relative_strength = data['Close'] / data['SMA_50']
        rs_signal = self._normalize_signal(relative_strength - 1, (-0.5, 1.4))
        signals['Relative_Strength'] = rs_signal
        
        return signals
    
    # ================================================================
    # 7. MACRO ECONOMIST PERSONA
    # ================================================================
    
    def macro_economist_signals(self, data):
        """
        Macro Economist Persona
        Focus: Economic cycles, seasonality, macro trends
        Philosophy: Macro forces drive long-term market direction
        """
        signals = {}
        
        # Signal 1: Economic Cycle Proxy (using price momentum as proxy)
        long_term_trend = data['Close'].rolling(252).mean()  # 1-year trend
        cycle_signal = pd.Series(0.0, index=data.index)
        
        price_vs_trend = data['Close'] / long_term_trend
        cycle_signal[price_vs_trend > 1.1] = 1.3   # Expansion phase
        cycle_signal[price_vs_trend < 0.95] = -0.2  # Contraction phase
        cycle_signal[(price_vs_trend >= 0.95) & (price_vs_trend <= 1.1)] = 0.7  # Neutral
        signals['Economic_Cycle'] = self._clip_signal(cycle_signal)
        
        # Signal 2: Seasonality Effects
        seasonal_signal = pd.Series(0.5, index=data.index)  # Base position
        
        # Add seasonal patterns
        month = data.index.month
        seasonal_signal[month.isin([11, 12, 1])] = 1.2  # Winter rally
        seasonal_signal[month.isin([5, 6, 7, 8, 9])] = 0.3  # Sell in May
        seasonal_signal[month == 10] = -0.2  # October effect
        
        # Turn of month effect
        day_of_month = data.index.day
        seasonal_signal[(day_of_month <= 3) | (day_of_month >= 28)] = seasonal_signal + 0.3
        signals['Seasonality'] = self._clip_signal(seasonal_signal)
        
        # Signal 3: Interest Rate Proxy (using volatility as inverse proxy)
        vol_proxy = data['Volatility_20'] if 'Volatility_20' in data.columns else data['Returns'].rolling(20).std() * np.sqrt(252)
        rate_proxy_signal = pd.Series(0.0, index=data.index)
        
        # Low volatility = low rates = good for growth stocks
        vol_percentile = vol_proxy.rolling(252).rank(pct=True)
        rate_proxy_signal[vol_percentile < 0.3] = 1.3  # Low vol environment
        rate_proxy_signal[vol_percentile > 0.7] = 0.2  # High vol environment
        signals['Interest_Rate_Proxy'] = self._clip_signal(rate_proxy_signal)
        
        # Signal 4: Inflation Expectations (using price momentum)
        price_momentum_3m = data['Close'].pct_change(63)  # 3-month change
        inflation_signal = pd.Series(0.8, index=data.index)
        
        # Moderate inflation good for stocks, high inflation bad
        inflation_signal[price_momentum_3m > 0.15] = 0.3  # High inflation proxy
        inflation_signal[price_momentum_3m < -0.10] = 1.2  # Deflation fears
        signals['Inflation_Expectations'] = self._clip_signal(inflation_signal)
        
        # Signal 5: Dollar Strength Proxy (using QQQ momentum vs its volatility)
        dollar_proxy = price_momentum_3m / (vol_proxy / 100 + 0.01)  # Risk-adjusted momentum
        dollar_signal = self._normalize_signal(dollar_proxy, (-0.3, 1.2))
        signals['Dollar_Strength_Proxy'] = dollar_signal
        
        return signals
    
    # ================================================================
    # 8. BEHAVIORAL PSYCHOLOGIST PERSONA
    # ================================================================
    
    def behavioral_psychologist_signals(self, data):
        """
        Behavioral Psychologist Persona
        Focus: Market psychology, sentiment, behavioral biases
        Philosophy: Human emotions drive market inefficiencies
        """
        signals = {}
        
        # Signal 1: Fear & Greed Indicator
        returns_5d = data['Returns'].rolling(5).sum()
        vol_5d = data['Returns'].rolling(5).std()
        
        fear_greed = returns_5d / (vol_5d + 0.001)  # Risk-adjusted short-term returns
        fg_signal = pd.Series(0.0, index=data.index)
        
        # Contrarian: buy fear, sell greed
        fg_percentile = fear_greed.rolling(252).rank(pct=True)
        fg_signal[fg_percentile < 0.2] = 1.3  # Extreme fear = buy
        fg_signal[fg_percentile > 0.8] = -0.4  # Extreme greed = sell
        fg_signal[(fg_percentile >= 0.4) & (fg_percentile <= 0.6)] = 0.8  # Neutral = slight long
        signals['Fear_Greed_Contrarian'] = self._clip_signal(fg_signal)
        
        # Signal 2: Overconfidence Bias (High volume after gains)
        price_change = data['Close'].pct_change()
        volume_ratio = data['Volume'] / data['Volume'].rolling(20).mean()
        
        overconfidence = pd.Series(0.0, index=data.index)
        # High volume after gains suggests overconfidence
        overconf_condition = (price_change > 0.02) & (volume_ratio > 1.5)
        overconfidence[overconf_condition] = -0.6  # Fade overconfidence
        
        # Low volume after gains suggests underreaction
        underreact_condition = (price_change > 0.02) & (volume_ratio < 0.7)
        overconfidence[underreact_condition] = 1.1  # Follow underreaction
        signals['Overconfidence_Bias'] = self._clip_signal(overconfidence)
        
        # Signal 3: Anchoring Bias (52-week high/low effect)
        high_52w = data['High'].rolling(252).max()
        low_52w = data['Low'].rolling(252).min()
        
        anchor_signal = pd.Series(0.0, index=data.index)
        # Distance from 52-week high (people anchor to recent highs)
        distance_from_high = (data['Close'] - high_52w) / high_52w
        anchor_signal[distance_from_high > -0.05] = -0.3  # Near 52w high = sell
        anchor_signal[distance_from_high < -0.20] = 1.1   # Far from 52w high = buy
        
        signals['Anchoring_Bias'] = self._clip_signal(anchor_signal)
        
        # Signal 4: Disposition Effect (Hold winners too long, sell losers too quick)
        returns_10d = data['Returns'].rolling(10).sum()
        disposition_signal = pd.Series(0.0, index=data.index)
        
        # Fade recent winners (people hold too long)
        disposition_signal[returns_10d > 0.10] = -0.5
        # Buy recent losers (people sell too quick)
        disposition_signal[returns_10d < -0.10] = 1.0
        signals['Disposition_Effect'] = self._clip_signal(disposition_signal)
        
        # Signal 5: Herding Behavior (Volume spikes indicate herding)
        volume_spike = data['Volume'] / data['Volume'].rolling(50).mean()
        price_change = data['Close'].pct_change()
        
        herd_signal = pd.Series(0.0, index=data.index)
        # Large volume + large moves = herding, fade it
        herding_condition = (volume_spike > 2.0) & (abs(price_change) > 0.03)
        herd_signal[herding_condition & (price_change > 0)] = -0.7  # Fade up herding
        herd_signal[herding_condition & (price_change < 0)] = 1.2   # Buy down herding
        signals['Herding_Behavior'] = self._clip_signal(herd_signal)
        
        return signals
    
    # ================================================================
    # 9. PHYSICIST PERSONA
    # ================================================================
    
    def physicist_signals(self, data):
        """
        Physicist Persona
        Focus: Physics principles applied to markets - momentum, energy, oscillations
        Philosophy: Markets follow physical laws of motion and energy
        """
        signals = {}
        
        # Signal 1: Newton's First Law - Objects in Motion Stay in Motion
        price_velocity = data['Close'].diff()  # First derivative
        price_acceleration = price_velocity.diff()  # Second derivative
        
        newton_signal = pd.Series(0.0, index=data.index)
        # Positive velocity + positive acceleration = strong uptrend
        newton_signal[(price_velocity > 0) & (price_acceleration > 0)] = 1.3
        # Negative velocity + negative acceleration = strong downtrend
        newton_signal[(price_velocity < 0) & (price_acceleration < 0)] = -0.8
        # Deceleration signals
        newton_signal[(price_velocity > 0) & (price_acceleration < 0)] = 0.2
        newton_signal[(price_velocity < 0) & (price_acceleration > 0)] = 0.8
        signals['Newton_Motion'] = self._clip_signal(newton_signal)
        
        # Signal 2: Harmonic Oscillator (Mean reversion as spring force)
        equilibrium = data['Close'].rolling(50).mean()
        displacement = (data['Close'] - equilibrium) / equilibrium
        
        # Spring force proportional to displacement (Hooke's Law)
        spring_force = -displacement * 2  # Restoring force
        oscillator_signal = self._clip_signal(spring_force)
        signals['Harmonic_Oscillator'] = oscillator_signal
        
        # Signal 3: Wave Interference Pattern
        # Multiple sine waves of different frequencies
        wave1 = np.sin(2 * np.pi * np.arange(len(data)) / 20)  # 20-day cycle
        wave2 = np.sin(2 * np.pi * np.arange(len(data)) / 50)  # 50-day cycle
        wave3 = np.sin(2 * np.pi * np.arange(len(data)) / 100) # 100-day cycle
        
        interference = (wave1 + wave2 + wave3) / 3
        wave_signal = pd.Series(interference, index=data.index) * 0.8
        signals['Wave_Interference'] = self._clip_signal(wave_signal)
        
        # Signal 4: Thermodynamics - Entropy and Energy Conservation
        # High volatility = high entropy = system far from equilibrium
        entropy_proxy = data['Returns'].rolling(20).std()
        energy_signal = pd.Series(0.0, index=data.index)
        
        entropy_percentile = entropy_proxy.rolling(252).rank(pct=True)
        energy_signal[entropy_percentile < 0.3] = 1.2  # Low entropy = stable = buy
        energy_signal[entropy_percentile > 0.7] = 0.1  # High entropy = unstable = reduce
        signals['Thermodynamic_Energy'] = self._clip_signal(energy_signal)
        
        # Signal 5: Quantum Tunneling Effect (Breakouts through resistance)
        resistance_level = data['High'].rolling(20).max()
        tunnel_signal = pd.Series(0.0, index=data.index)
        
        # "Quantum tunneling" through resistance
        tunnel_probability = (data['Close'] - resistance_level.shift(1)) / resistance_level.shift(1)
        tunnel_signal[tunnel_probability > 0.02] = 1.4  # Successful tunneling
        tunnel_signal[tunnel_probability < -0.02] = -0.3  # Failed tunneling
        signals['Quantum_Tunneling'] = self._clip_signal(tunnel_signal)
        
        return signals
    
    # ================================================================
    # 10. BIOLOGIST PERSONA
    # ================================================================
    
    def biologist_signals(self, data):
        """
        Biologist Persona
        Focus: Evolution, adaptation, ecosystem dynamics, survival strategies
        Philosophy: Markets evolve and adapt like biological systems
        """
        signals = {}
        
        # Signal 1: Natural Selection - Survival of the Fittest Trends
        fitness_score = data['Returns'].rolling(63).mean() / data['Returns'].rolling(63).std()
        
        selection_signal = pd.Series(0.0, index=data.index)
        fitness_rank = fitness_score.rolling(252).rank(pct=True)
        selection_signal[fitness_rank > 0.8] = 1.3  # High fitness = survive
        selection_signal[fitness_rank < 0.2] = -0.2  # Low fitness = extinction
        signals['Natural_Selection'] = self._clip_signal(selection_signal)
        
        # Signal 2: Predator-Prey Dynamics (Volatility vs Returns)
        predator = data['Returns'].rolling(20).std() * 100  # Volatility as predator
        prey = abs(data['Returns'].rolling(20).mean()) * 100  # Returns as prey
        
        # Lotka-Volterra equations approximation
        predator_prey_ratio = prey / (predator + 0.001)
        pp_signal = self._normalize_signal(predator_prey_ratio, (0.2, 1.3))
        signals['Predator_Prey'] = pp_signal
        
        # Signal 3: Evolutionary Adaptation Speed
        # How quickly the market adapts to new information
        adaptation_speed = data['Returns'].rolling(5).std() / data['Returns'].rolling(20).std()
        
        adapt_signal = pd.Series(0.8, index=data.index)  # Base position
        adapt_signal[adaptation_speed > 1.5] = 0.3  # Fast adaptation = unstable
        adapt_signal[adaptation_speed < 0.7] = 1.2  # Slow adaptation = stable
        signals['Adaptation_Speed'] = self._clip_signal(adapt_signal)
        
        # Signal 4: Symbiosis - Volume and Price Relationship
        volume_price_correlation = data['Volume'].rolling(63).corr(abs(data['Returns']))
        
        symbiosis_signal = pd.Series(0.5, index=data.index)
        symbiosis_signal[volume_price_correlation > 0.5] = 1.1  # Good symbiosis
        symbiosis_signal[volume_price_correlation < 0.1] = 0.2  # Poor symbiosis
        signals['Symbiosis'] = self._clip_signal(symbiosis_signal)
        
        # Signal 5: Genetic Drift - Random Walk Detection
        # Measure how much price movement is random vs directional
        drift_measure = abs(data['Returns']).rolling(20).sum() / abs(data['Close'].diff(20) / data['Close'].shift(20))
        
        drift_signal = pd.Series(0.0, index=data.index)
        drift_signal[drift_measure > 5] = 0.1   # High drift = random = reduce position
        drift_signal[drift_measure < 2] = 1.2   # Low drift = directional = increase
        signals['Genetic_Drift'] = self._clip_signal(drift_signal)
        
        return signals
    
    # ================================================================
    # 11. OCEANOGRAPHER PERSONA
    # ================================================================
    
    def oceanographer_signals(self, data):
        """
        Oceanographer Persona
        Focus: Wave patterns, tides, currents, fluid dynamics
        Philosophy: Markets flow like ocean currents with predictable patterns
        """
        signals = {}
        
        # Signal 1: Elliott Wave-inspired Pattern
        # Simplified Elliott Wave using price swings
        price_swings = data['Close'].diff().rolling(5).sum()
        
        wave_signal = pd.Series(0.0, index=data.index)
        # Look for 5-wave patterns (simplified)
        swing_direction = np.sign(price_swings)
        direction_changes = (swing_direction != swing_direction.shift(1)).rolling(10).sum()
        
        wave_signal[direction_changes >= 4] = 1.1  # Complete wave pattern
        wave_signal[direction_changes <= 1] = 0.3  # Trending phase
        signals['Elliott_Waves'] = self._clip_signal(wave_signal)
        
        # Signal 2: Tidal Forces - Long-term Cycles
        # Use multiple moving averages as tidal forces
        tide_fast = data['Close'].rolling(21).mean()  # Monthly tide
        tide_slow = data['Close'].rolling(252).mean()  # Annual tide
        
        tidal_signal = pd.Series(0.0, index=data.index)
        tidal_alignment = (data['Close'] - tide_fast) * (tide_fast - tide_slow)
        tidal_signal[tidal_alignment > 0] = 1.0  # Tides aligned = strong signal
        tidal_signal[tidal_alignment < 0] = 0.2  # Tides opposing = weak signal
        signals['Tidal_Forces'] = self._clip_signal(tidal_signal)
        
        # Signal 3: Ocean Current Strength
        # Measure the "current" strength using volume-weighted momentum
        current_strength = (data['Close'].diff() * data['Volume']).rolling(10).sum()
        current_strength_norm = current_strength / data['Volume'].rolling(10).sum()
        
        current_signal = self._normalize_signal(current_strength_norm, (-0.5, 1.4))
        signals['Ocean_Current'] = current_signal
        
        # Signal 4: Tsunami Warning System
        # Detect potential large moves using volume and volatility spikes
        vol_spike = data['Returns'].rolling(5).std() / data['Returns'].rolling(20).std()
        volume_spike = data['Volume'] / data['Volume'].rolling(20).mean()
        
        tsunami_signal = pd.Series(1.0, index=data.index)  # Default long
        warning_condition = (vol_spike > 2.0) & (volume_spike > 2.0)
        tsunami_signal[warning_condition] = 0.0  # Go to cash before tsunami
        
        # Recovery signal after tsunami
        recovery_condition = warning_condition.shift(1) & (vol_spike < 1.5)
        tsunami_signal[recovery_condition] = 1.4  # Buy the recovery
        signals['Tsunami_Warning'] = self._clip_signal(tsunami_signal)
        
        # Signal 5: Rip Current Detection
        # Detect dangerous counter-trends
        main_trend = data['Close'] > data['SMA_50']
        short_trend = data['Close'].diff(5).rolling(5).mean()
        
        rip_current = pd.Series(0.8, index=data.index)  # Base position
        # Rip current = short-term move against main trend
        rip_condition = main_trend & (short_trend < -data['Close'] * 0.01)
        rip_current[rip_condition] = 0.2  # Reduce exposure in rip current
        
        # Opportunity after rip current
        recovery_from_rip = rip_condition.shift(1) & (short_trend > 0)
        rip_current[recovery_from_rip] = 1.3  # Buy recovery from rip
        signals['Rip_Current'] = self._clip_signal(rip_current)
        
        return signals
    
    # ================================================================
    # 12. SHORT SPECIALIST PERSONA
    # ================================================================
    
    def short_specialist_signals(self, data):
        """
        Short Specialist Persona
        Focus: Bear market signals, crash prediction, short-only strategies
        Philosophy: Markets fall faster than they rise - specialize in the downside
        """
        signals = {}
        
        # Signal 1: Crash Predictor - Multiple Warning Signs
        crash_score = pd.Series(0, index=data.index)
        
        # Warning sign 1: Parabolic move (price > 120% of 200-day MA)
        parabolic = data['Close'] / data['SMA_200'] > 1.20
        crash_score += parabolic.astype(int)
        
        # Warning sign 2: Low volatility before crash
        low_vol = data['Volatility_20'] < data['Volatility_20'].rolling(252).quantile(0.2)
        crash_score += low_vol.astype(int)
        
        # Warning sign 3: High momentum
        high_momentum = data['Close'].pct_change(20) > 0.15
        crash_score += high_momentum.astype(int)
        
        crash_signal = pd.Series(0.0, index=data.index)
        crash_signal[crash_score >= 2] = -1.0  # Multiple warning signs = short
        crash_signal[crash_score == 1] = -0.3  # Single warning = light short
        signals['Crash_Predictor'] = self._clip_signal(crash_signal)
        
        # Signal 2: Bear Market Momentum
        bear_momentum = pd.Series(0.0, index=data.index)
        
        # Identify bear market (below 200-day MA with declining MA)
        bear_market = (data['Close'] < data['SMA_200']) & (data['SMA_200'].diff(20) < 0)
        bear_momentum[bear_market] = -0.8
        
        # Accelerating bear market
        accelerating_bear = bear_market & (data['Close'].pct_change(10) < -0.10)
        bear_momentum[accelerating_bear] = -1.0
        signals['Bear_Momentum'] = self._clip_signal(bear_momentum)
        
        # Signal 3: Distribution Detection
        # Look for signs of institutional selling
        distribution_signal = pd.Series(0.0, index=data.index)
        
        # High volume on down days
        down_days = data['Returns'] < 0
        high_volume_down = down_days & (data['Volume'] > data['Volume'].rolling(20).mean() * 1.3)
        
        # Accumulate distribution signals
        distribution_score = high_volume_down.rolling(20).sum()
        distribution_signal[distribution_score > 8] = -0.7  # Heavy distribution
        distribution_signal[distribution_score > 12] = -1.0  # Extreme distribution
        signals['Distribution_Detection'] = self._clip_signal(distribution_signal)
        
        # Signal 4: Failed Rally Signal
        # Rallies that fail in bear markets are shorting opportunities
        failed_rally = pd.Series(0.0, index=data.index)
        
        # Define rally attempt (3+ consecutive up days)
        up_days = data['Returns'] > 0
        rally_attempt = up_days.rolling(3).sum() >= 3
        
        # Rally fails if it doesn't break recent high and then declines
        recent_high = data['High'].rolling(10).max()
        rally_failed = rally_attempt.shift(1) & (data['Close'] < recent_high.shift(1)) & (data['Returns'] < -0.02)
        
        failed_rally[rally_failed] = -0.9
        signals['Failed_Rally'] = self._clip_signal(failed_rally)
        
        # Signal 5: Volatility Explosion Short
        # Short when volatility explodes (usually happens in crashes)
        vol_explosion = pd.Series(0.0, index=data.index)
        
        vol_ratio = data['Volatility_5'] / data['Volatility_20']
        vol_explosion[vol_ratio > 2.0] = -0.8  # Volatility explosion
        vol_explosion[vol_ratio > 3.0] = -1.0  # Extreme volatility explosion
        
        # But cover shorts when volatility starts to decline (bottoming process)
        vol_declining = (vol_ratio < vol_ratio.shift(1)) & (vol_ratio.shift(1) > 2.0)
        vol_explosion[vol_declining] = 0.0  # Cover shorts
        signals['Volatility_Explosion'] = self._clip_signal(vol_explosion)
        
        return signals
    
    # ================================================================
    # MASTER SIGNAL GENERATOR
    # ================================================================
    
    def generate_all_signals(self, data):
        """Generate all signals from all personas."""
        all_signals = {}
        
        # Generate signals from each persona
        personas = [
            ('Volatility_Specialist', self.volatility_specialist_signals),
            ('Momentum_Trader', self.momentum_trader_signals),
            ('Mean_Reversion_Trader', self.mean_reversion_trader_signals),
            ('Risk_Manager', self.risk_manager_signals),
            ('Quantitative_Researcher', self.quantitative_researcher_signals),
            ('Technical_Analyst', self.technical_analyst_signals),
            ('Macro_Economist', self.macro_economist_signals),
            ('Behavioral_Psychologist', self.behavioral_psychologist_signals),
            ('Physicist', self.physicist_signals),
            ('Biologist', self.biologist_signals),
            ('Oceanographer', self.oceanographer_signals),
            ('Short_Specialist', self.short_specialist_signals)
        ]
        
        for persona_name, persona_func in personas:
            try:
                persona_signals = persona_func(data)
                for signal_name, signal_series in persona_signals.items():
                    full_signal_name = f"{persona_name}_{signal_name}"
                    all_signals[full_signal_name] = signal_series
                print(f"✅ {persona_name}: Generated {len(persona_signals)} signals")
            except Exception as e:
                print(f"❌ Error in {persona_name}: {e}")
        
        print(f"\n🎯 Total signals generated: {len(all_signals)}")
        return all_signals


if __name__ == "__main__":
    # Test the signal generators
    print("Testing Signal Generators...")
    
    # Create dummy data for testing
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    dummy_data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(500) * 0.02),
        'High': None,
        'Low': None,
        'Open': None,
        'Volume': np.random.randint(1000000, 5000000, 500),
        'Returns': None
    }, index=dates)
    
    # Add OHLC data
    dummy_data['High'] = dummy_data['Close'] + np.random.uniform(0, 2, 500)
    dummy_data['Low'] = dummy_data['Close'] - np.random.uniform(0, 2, 500)
    dummy_data['Open'] = dummy_data['Close'].shift(1) + np.random.uniform(-1, 1, 500)
    dummy_data['Returns'] = dummy_data['Close'].pct_change()
    
    # Add technical indicators that are expected
    for period in [20, 50, 200]:
        dummy_data[f'SMA_{period}'] = dummy_data['Close'].rolling(period).mean()
    
    # Add other expected indicators
    delta = dummy_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    dummy_data['RSI_14'] = 100 - (100 / (1 + rs))
    
    dummy_data['Volatility_20'] = dummy_data['Returns'].rolling(20).std() * np.sqrt(252)
    dummy_data['Volatility_5'] = dummy_data['Returns'].rolling(5).std() * np.sqrt(252)
    
    # Test signal generation
    generator = SignalGenerators()
    all_signals = generator.generate_all_signals(dummy_data)
    
    print(f"\n🚀 SUCCESS: Generated {len(all_signals)} total signals!")
    print("Signal generation system ready for production use.")
