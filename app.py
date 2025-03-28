from flask import Flask, render_template, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from datetime import datetime
import time
import joblib
import numpy as np
import mplfinance as mpf
import yfinance as yf
import warnings

app = Flask(__name__)

# Load ML model
model = joblib.load("ml_model.pkl")

def initialize_mt5():
    if not mt5.initialize(server="Exness-MT5Trial8", login=79700174, password="Botmudra.com@01"):
        print("Failed to connect:", mt5.last_error())
        return False
    return True

def get_live_data():
    if not initialize_mt5():
        return None
    
    # Fetch latest 1000 candles of XAUUSD
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M5
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
    
    if rates is None:
        print("Failed to get rates:", mt5.last_error())
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Get current price
    current_price = df.iloc[-1]['close']
    price_change = df.iloc[-1]['close'] - df.iloc[-2]['close']
    price_change_pct = (price_change / df.iloc[-2]['close']) * 100
    
    # Shutdown MT5 connection
    mt5.shutdown()
    return df, current_price, price_change, price_change_pct

def get_prediction(df):
    latest_data = df.iloc[-1][['open', 'high', 'low', 'close', 'tick_volume']].values.reshape(1, -1)
    prediction = model.predict_proba(latest_data)[0]
    bullish_prob = round(prediction[1] * 100, 2)  # Probability of bullish candle
    bearish_prob = round(prediction[0] * 100, 2)  # Probability of bearish candle
    return bullish_prob, bearish_prob

def get_correlation_data():
    # Get correlation with USD, S&P 500, and US Treasury Yields
    symbols = ['GC=F', 'DX-Y.NYB', '^GSPC', '^TNX']  # Gold, USD Index, S&P 500, 10Y Treasury
    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1mo')
        data[symbol] = hist['Close']
    
    # Calculate correlations
    df_corr = pd.DataFrame(data)
    correlations = df_corr.corr()['GC=F'].drop('GC=F')
    return correlations

def calculate_sentiment(df):
    """Calculate market sentiment based on price action"""
    try:
        # Calculate sentiment based on recent price movements
        recent_prices = df['close'].tail(20)  # Last 20 periods
        price_changes = recent_prices.pct_change().dropna()
        
        # Calculate weighted sentiment
        weights = np.linspace(0.1, 1, len(price_changes))  # More weight to recent changes
        weighted_changes = price_changes * weights
        sentiment = np.mean(weighted_changes) * 100
        
        return round(sentiment, 2)
    except Exception as e:
        print(f"Error calculating sentiment: {str(e)}")
        return 0

def generate_chart(df):
    # Rename tick_volume to volume for mplfinance
    df = df.rename(columns={'tick_volume': 'volume'})
    
    # Set style for better visibility
    style = mpf.make_mpf_style(
        marketcolors=mpf.make_marketcolors(
            up='green',
            down='red',
            edge='inherit',
            wick='inherit',
            volume='inherit'
        ),
        gridstyle='dotted'
    )
    
    # Create the candlestick chart
    fig, axes = mpf.plot(df, type='candle', style=style,
                        title='XAUUSD Price Chart (5-Minute Interval)',
                        ylabel='Price (USD)',
                        volume=True,
                        returnfig=True,
                        figsize=(12, 8))
    
    # Adjust layout
    plt.subplots_adjust(hspace=0.3)
    
    # Save the chart as an image
    plt.savefig("static/xauusd_chart.png", bbox_inches='tight', dpi=100)
    plt.close(fig)

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1.+rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_ma(prices, period=20):
    """Calculate Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_technical_indicators(df):
    """Calculate technical indicators for the latest data"""
    try:
        # Calculate RSI
        rsi = calculate_rsi(df['close'])
        
        # Calculate MACD
        macd, signal, hist = calculate_macd(df['close'])
        
        # Calculate Moving Average
        ma20 = calculate_ma(df['close'], 20)
        
        # Calculate Stochastic
        k, d = calculate_stochastic(df['high'], df['low'], df['close'])
        
        # Get latest values
        latest_rsi = round(float(rsi.iloc[-1]), 1)
        latest_macd = round(float(hist.iloc[-1]), 2)
        latest_ma20 = round(float(ma20.iloc[-1]), 2)
        latest_stoch = round(float(k.iloc[-1]), 1)
        
        # Determine signals
        rsi_signal = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
        macd_signal = "Bullish" if latest_macd > 0 else "Bearish"
        ma_signal = "Above Price" if latest_ma20 > df['close'].iloc[-1] else "Below Price"
        stoch_signal = "Overbought" if latest_stoch > 80 else "Oversold" if latest_stoch < 20 else "Neutral"
        
        return {
            "rsi": {"value": latest_rsi, "signal": rsi_signal},
            "macd": {"value": latest_macd, "signal": macd_signal},
            "ma20": {"value": latest_ma20, "signal": ma_signal},
            "stoch": {"value": latest_stoch, "signal": stoch_signal}
        }
    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        return None

def calculate_support_resistance(df):
    """Calculate support and resistance levels"""
    try:
        # Get recent high and low prices
        recent_data = df.tail(100)
        
        # Calculate pivot points
        pivot = (recent_data['high'].iloc[-1] + recent_data['low'].iloc[-1] + recent_data['close'].iloc[-1]) / 3
        
        r1 = 2 * pivot - recent_data['low'].iloc[-1]
        r2 = pivot + (recent_data['high'].iloc[-1] - recent_data['low'].iloc[-1])
        s1 = 2 * pivot - recent_data['high'].iloc[-1]
        s2 = pivot - (recent_data['high'].iloc[-1] - recent_data['low'].iloc[-1])
        
        return {
            "strong_resistance": round(r2, 2),
            "resistance": round(r1, 2),
            "support": round(s1, 2),
            "strong_support": round(s2, 2)
        }
    except Exception as e:
        print(f"Error calculating support/resistance: {str(e)}")
        return None

def calculate_market_insights(df):
    """Calculate market insights including trend strength and volatility"""
    try:
        # Calculate trend strength using price momentum
        returns = df['close'].pct_change()
        momentum = returns.rolling(window=20).mean() * 100
        trend_strength = min(abs(float(momentum.iloc[-1] * 10)), 100)  # Scale and cap at 100
        
        # Calculate volatility using standard deviation
        volatility = min(float(returns.rolling(window=20).std() * 100 * 10), 100)  # Scale and cap at 100
        
        # Determine trend direction using simple moving averages
        sma20 = df['close'].rolling(window=20).mean()
        sma50 = df['close'].rolling(window=50).mean()
        trend_direction = "Uptrend" if float(sma20.iloc[-1]) > float(sma50.iloc[-1]) else "Downtrend"
        
        # Create market summary
        if trend_strength > 25:
            strength_desc = "Strong" if trend_strength > 50 else "Moderate"
            trend_desc = f"{strength_desc} {trend_direction}"
        else:
            trend_desc = "Ranging Market"
            
        volatility_desc = "High" if volatility > 70 else "Moderate" if volatility > 30 else "Low"
        
        return {
            "trend_strength": trend_strength,
            "trend_description": trend_desc,
            "volatility": volatility,
            "volatility_description": volatility_desc
        }
    except Exception as e:
        print(f"Error calculating market insights: {str(e)}")
        return None

@app.route("/")
def home():
    data = get_live_data()
    if data is not None:
        df, current_price, price_change, price_change_pct = data
        generate_chart(df)
        bullish, bearish = get_prediction(df)
        
        # Get additional data
        correlations = get_correlation_data()
        sentiment = calculate_sentiment(df)
        
        # Calculate technical analysis data
        technical_indicators = calculate_technical_indicators(df)
        support_resistance = calculate_support_resistance(df)
        market_insights = calculate_market_insights(df)
        
        return render_template("index.html", 
                             bullish=bullish, 
                             bearish=bearish,
                             current_price=round(current_price, 2),
                             price_change=round(price_change, 2),
                             price_change_pct=round(price_change_pct, 2),
                             correlations=correlations,
                             sentiment=sentiment,
                             technical_indicators=technical_indicators,
                             support_resistance=support_resistance,
                             market_insights=market_insights)
    return render_template("index.html")

@app.route("/update_chart")
def update_chart():
    data = get_live_data()
    if data is not None:
        df, current_price, price_change, price_change_pct = data
        generate_chart(df)
        bullish, bearish = get_prediction(df)
        
        # Get additional data
        correlations = get_correlation_data()
        sentiment = calculate_sentiment(df)
        
        # Calculate technical analysis data
        technical_indicators = calculate_technical_indicators(df)
        support_resistance = calculate_support_resistance(df)
        market_insights = calculate_market_insights(df)
        
        return jsonify({
            "status": "success",
            "bullish": bullish,
            "bearish": bearish,
            "current_price": round(current_price, 2),
            "price_change": round(price_change, 2),
            "price_change_pct": round(price_change_pct, 2),
            "correlations": correlations.to_dict(),
            "sentiment": sentiment,
            "technical_indicators": technical_indicators,
            "support_resistance": support_resistance,
            "market_insights": market_insights
        })
    return jsonify({"status": "error"})

if __name__ == "__main__":
    app.run(debug=True)
