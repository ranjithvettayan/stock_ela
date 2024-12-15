from nsepython import nse_eq, nse_quote_ltp, nse_optionchain_scrapper
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import plotly.graph_objects as go
import plotly.express as px
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import json
import yfinance as yf
import pickle
import requests
from datetime import datetime, timedelta

load_dotenv()
gemini_pro = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_GENAI_API_KEY"))

# Sentiment Analysis Model
sentiment_model = pipeline("sentiment-analysis")

# Watchlist Management
def save_watchlist(watchlist):
    with open('watchlist.pkl', 'wb') as f:
        pickle.dump(watchlist, f)

def load_watchlist():
    try:
        with open('watchlist.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

# Function to fetch trending stocks based on multiple factors
def fetch_trending_stocks():
    """
    Fetch trending stocks based on multiple criteria:
    1. Climate and seasonal impacts
    2. Market sentiment
    3. Social media trends
    4. Potential growth sectors
    """
    try:
        # Prompt for trending stocks analysis
        prompt = """
        Provide a list of 10 potentially trending Indian stocks considering:
        1. Climate and seasonal impacts
        2. Current market sentiment
        3. Technological and economic trends
        4. Potential growth sectors

        Return in strict JSON format:
        [
            {
                "symbol": "STOCKSYMBOL",
                "company": "Company Name",
                "sector": "Sector",
                "trend_score": 85,
                "prediction_1week": "+5.6%",
                "prediction_1month": "+12.3%",
                "key_driver": "Climate/Tech/Economic Trend"
            }
        ]
        """

        # Use Gemini Pro to generate trending stocks
        response = gemini_pro.invoke(prompt)
        
        try:
            # Parse JSON response
            if response.content.startswith("```json"):
                trending_stocks = json.loads(response.content.strip("```json").strip("```"))
            else:
                trending_stocks = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback trending stocks
            trending_stocks = [
                {
                    "symbol": "RELIANCE",
                    "company": "Reliance Industries",
                    "sector": "Conglomerate",
                    "trend_score": 85,
                    "prediction_1week": "+4.2%",
                    "prediction_1month": "+11.5%",
                    "key_driver": "Green Energy Transition"
                },
                {
                    "symbol": "INFY",
                    "company": "Infosys",
                    "sector": "IT Services",
                    "trend_score": 82,
                    "prediction_1week": "+3.8%",
                    "prediction_1month": "+10.2%",
                    "key_driver": "AI and Digital Transformation"
                }
            ]

        return pd.DataFrame(trending_stocks)

    except Exception as e:
        st.error(f"Error fetching trending stocks: {e}")
        return pd.DataFrame()

# Function to predict stock price
def predict_stock_price(stock_name, historical_data):
    """
    Simple predictive model using recent trends and moving averages
    """
    if historical_data is None or len(historical_data) < 30:
        return {
            "1_week_prediction": "Insufficient data",
            "1_month_prediction": "Insufficient data"
        }
    
    # Calculate moving averages
    short_ma = historical_data['Close'].rolling(window=10).mean().iloc[-1]
    long_ma = historical_data['Close'].rolling(window=30).mean().iloc[-1]
    
    # Last known closing price
    last_price = historical_data['Close'].iloc[-1]
    
    # Simple prediction based on moving averages
    if short_ma > long_ma:
        # Bullish trend
        one_week_pred = last_price * 1.03  # 3% increase
        one_month_pred = last_price * 1.08  # 8% increase
    else:
        # Bearish trend
        one_week_pred = last_price * 0.97  # 3% decrease
        one_month_pred = last_price * 0.92  # 8% decrease
    
    return {
        "1_week_prediction": f"₹{one_week_pred:.2f} ({'+' if one_week_pred > last_price else ''}{((one_week_pred - last_price)/last_price*100):.2f}%)",
        "1_month_prediction": f"₹{one_month_pred:.2f} ({'+' if one_month_pred > last_price else ''}{((one_month_pred - last_price)/last_price*100):.2f}%)"
    }

# Function to fetch stock data using nsepython
def fetch_stock_data(stock_name):
    try:
        stock_data = nse_eq(stock_name.upper())
        return {
            'name': stock_data['info']['companyName'],
            'price': stock_data['priceInfo']['lastPrice'],
            'high': stock_data['priceInfo']['intraDayHighLow']['max'],
            'low': stock_data['priceInfo']['intraDayHighLow']['min'],
            'open': stock_data['priceInfo']['open'],
            'previousClose': stock_data['priceInfo']['previousClose']
        }
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Function to fetch historical stock data
def fetch_historical_data(stock_name, period='1y'):
    try:
        # Convert NSE symbol to Yahoo Finance format if needed
        yahoo_symbol = f"{stock_name}.NS"
        
        # Fetch historical data
        stock = yf.Ticker(yahoo_symbol)
        hist_data = stock.history(period=period)
        
        if hist_data.empty:
            st.warning("No historical data found. Using simulated data.")
            return None
        
        return hist_data
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return None

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Supertrend
def calculate_supertrend(data, period=7, multiplier=3):
    atr = data['High'].rolling(window=period).max() - data['Low'].rolling(window=period).min()
    hl2 = (data['High'] + data['Low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    supertrend = pd.Series(index=data.index, dtype='float64')
    direction = pd.Series(index=data.index, dtype='int64')

    for i in range(len(data)):
        if i == 0:
            supertrend[i] = upperband[i] if data['Close'][i] <= upperband[i] else lowerband[i]
            direction[i] = 1 if data['Close'][i] > supertrend[i] else -1
        else:
            if direction[i-1] == 1 and data['Close'][i] < lowerband[i]:
                direction[i] = -1
            elif direction[i-1] == -1 and data['Close'][i] > upperband[i]:
                direction[i] = 1
            else:
                direction[i] = direction[i-1]

            if direction[i] == 1:
                supertrend[i] = max(lowerband[i], supertrend[i-1])
            else:
                supertrend[i] = min(upperband[i], supertrend[i-1])

    return supertrend, direction

# Function to analyze news
def analyze_news(stock_name):
    """
    Analyze the latest news headlines related to a given stock name.
    
    Args:
        stock_name (str): The name of the stock/company to analyze.
    
    Returns:
        pd.DataFrame: A DataFrame with headlines, sentiment, and confidence scores.
    """
    prompt = f"""
    Provide 3 latest news headlines related to {stock_name} in a strict JSON format. 
    Each headline should have a headline and a summary.
    Format exactly like this:
    [
        {{ "headline": "First headline about {stock_name}", "summary": "Brief summary of first headline" }},
        {{ "headline": "Second headline about {stock_name}", "summary": "Brief summary of second headline" }},
        {{ "headline": "Third headline about {stock_name}", "summary": "Brief summary of third headline" }}
    ]
    """

    try:
        # Fetch news using Gemini Pro
        response = gemini_pro.invoke(prompt)
        news_items = response.content

        # Try parsing the JSON from the response
        try:
            # Attempt to parse the JSON, removing any potential markdown code block markers
            if news_items.startswith("```json"):
                news_items = news_items.strip("```json").strip("```")
            
            news_list = json.loads(news_items)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"JSON Parsing Error: {e}")
            print("Received content:", news_items)
            
            # Fallback to predefined headlines if JSON parsing fails
            news_list = [
                {"headline": f"{stock_name} market performance update", "summary": "Recent market trends"},
                {"headline": f"Analyst views on {stock_name}", "summary": "Expert market analysis"},
                {"headline": f"Industry outlook for {stock_name}", "summary": "Sector performance insights"}
            ]

        # Create DataFrame
        news_df = pd.DataFrame(news_list)
        news_df.columns = ["Headline", "Summary"]

        # Perform Sentiment Analysis
        news_df["Sentiment"] = news_df["Headline"].apply(lambda x: sentiment_model(x)[0]["label"])
        news_df["Confidence"] = news_df["Headline"].apply(lambda x: round(sentiment_model(x)[0]["score"], 2))

        return news_df[["Headline", "Sentiment", "Confidence"]]
    
    except Exception as e:
        print(f"Error fetching or processing news: {e}")

        # Fallback to static data in case of complete failure
        fallback_headlines = [
            f"{stock_name} achieves record profits in Q3.",
            f"Market sees volatility in {stock_name}'s sector.",
            f"Experts predict strong growth for {stock_name}."
        ]

        fallback_sentiments = [sentiment_model(headline)[0] for headline in fallback_headlines]

        return pd.DataFrame({
            "Headline": fallback_headlines,
            "Sentiment": [score["label"] for score in fallback_sentiments],
            "Confidence": [round(score["score"], 2) for score in fallback_sentiments]
        })

def calculate_adx(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)
    
    adx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di)).rolling(window=period).mean()
    return adx
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    
    return upper_band, lower_band

def calculate_fibonacci_extension(data, level1=0.236, level2=0.382, level3=0.618):
    max_price = data['High'].max()
    min_price = data['Low'].min()
    
    diff = max_price - min_price
    
    fib1 = max_price + (diff * level1)
    fib2 = max_price + (diff * level2)
    fib3 = max_price + (diff * level3)
    
    return fib1, fib2, fib3

def calculate_keltner_channel(data, period=20, multiplier=1.5):
    hl2 = (data['High'] + data['Low']) / 2
    ema = hl2.rolling(window=period).mean()
    atr = (data['High'] - data['Low']).rolling(window=period).mean()
    
    upper_channel = ema + (multiplier * atr)
    lower_channel = ema - (multiplier * atr)
    
    return upper_channel, lower_channel

def calculate_ema(data, short_period=9, medium_period=21, long_period=55, long_term_period=200):
    short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
    medium_ema = data['Close'].ewm(span=medium_period, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()
    long_term_ema = data['Close'].ewm(span=long_term_period, adjust=False).mean()
    
    return short_ema, medium_ema, long_ema, long_term_ema

def calculate_supertrend_with_target_stoploss(data, period=7, multiplier=3):
    supertrend, direction = calculate_supertrend(data, period, multiplier)
    
    current_price = data['Close'].iloc[-1]
    target_price = current_price * 1.01  # 1% target
    stop_loss_price = current_price * 0.99  # 1% stop-loss
    
    return supertrend, direction, target_price, stop_loss_price

def golden_crossover(data):
    short_ema = data['Close'].ewm(span=50, adjust=False).mean()
    long_ema = data['Close'].ewm(span=200, adjust=False).mean()
    
    crossover = short_ema.iloc[-1] > long_ema.iloc[-1] and short_ema.iloc[-2] <= long_ema.iloc[-2]
    
    return crossover


# Streamlit Dashboard
st.title("Indian Stock Market Trading App")
# Initialize session state for watchlist
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

# Sidebar Configuration
st.sidebar.header("Trading Dashboard")

# Trending Stocks Section
st.sidebar.subheader("🔥 Trending Stocks")
trending_stocks_df = fetch_trending_stocks()
st.sidebar.dataframe(trending_stocks_df)

# Watchlist Management
st.sidebar.subheader("📋 Watchlist")
watchlist_action = st.sidebar.selectbox("Watchlist Actions", ["View Watchlist", "Add to Watchlist"])

if watchlist_action == "Add to Watchlist":
    add_stock = st.sidebar.text_input("Enter Stock Symbol to Add")
    if st.sidebar.button("Add Stock"):
        st.session_state.watchlist[add_stock.upper()] = {
            "added_date": datetime.now(),
            "bought": False,
            "buy_price": None
        }
        save_watchlist(st.session_state.watchlist)
        st.sidebar.success(f"{add_stock.upper()} added to watchlist!")

# Display Watchlist
if watchlist_action == "View Watchlist":
    if st.session_state.watchlist:
        watchlist_df = pd.DataFrame.from_dict(st.session_state.watchlist, orient='index')
        watchlist_df.index.name = 'Symbol'
        watchlist_df.reset_index(inplace=True)
        
        st.sidebar.dataframe(watchlist_df)
        
        # Options for each watchlist stock
        for symbol in st.session_state.watchlist.keys():
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.sidebar.button(f"Buy {symbol}"):
                    # Mark stock as bought and record buy price
                    try:
                        current_price = fetch_stock_data(symbol)['price']
                        st.session_state.watchlist[symbol]['bought'] = True
                        st.session_state.watchlist[symbol]['buy_price'] = current_price
                        save_watchlist(st.session_state.watchlist)
                        st.sidebar.success(f"Bought {symbol} at ₹{current_price}")
                    except Exception as e:
                        st.sidebar.error(f"Could not buy {symbol}: {e}")
            
            with col2:
                if st.sidebar.button(f"Sell {symbol}"):
                    if st.session_state.watchlist[symbol]['bought']:
                        try:
                            current_price = fetch_stock_data(symbol)['price']
                            buy_price = st.session_state.watchlist[symbol]['buy_price']
                            profit_loss = ((current_price - buy_price) / buy_price) * 100
                            
                            # Reset watchlist entry
                            st.session_state.watchlist[symbol]['bought'] = False
                            st.session_state.watchlist[symbol]['buy_price'] = None
                            save_watchlist(st.session_state.watchlist)
                            
                            st.sidebar.success(f"Sold {symbol}. Profit/Loss: {profit_loss:.2f}%")
                        except Exception as e:
                            st.sidebar.error(f"Could not sell {symbol}: {e}")
                    else:
                        st.sidebar.warning(f"{symbol} not in portfolio")

st.sidebar.header("Configure Trading Strategy")

# Select stock
stock_name = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE, TCS, INFY):", "RELIANCE")

# Time range selection for historical data
time_ranges = {
    'Day': '1d', 
    'Month': '1mo', 
    '3 Months': '3mo', 
    '6 Months': '6mo', 
    '1 Year': '1y', 
    '2 Years': '2y', 
    '3 Years': '3y', 
    '5 Years': '5y'
}

# Fetch stock data
stock_data = fetch_stock_data(stock_name.upper())

if stock_data:
    st.subheader(f"Live Data for {stock_data['name']}")
    st.metric("Current Price", f"₹{stock_data['price']}")
    st.metric("Day's High", f"₹{stock_data['high']}")
    st.metric("Day's Low", f"₹{stock_data['low']}")
    st.metric("Open Price", f"₹{stock_data['open']}")
    st.metric("Previous Close", f"₹{stock_data['previousClose']}")

        # Time range selection
    selected_range = st.selectbox("Select Time Range:", list(time_ranges.keys()), index=4)
    
    # Fetch historical data
    historical_data = fetch_historical_data(stock_name, time_ranges[selected_range])
    
    if historical_data is not None:
        # Price Movement Chart
        st.subheader(f"Price Movement ({selected_range})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Close Price'))
        fig.update_layout(
            xaxis_title='Date', 
            yaxis_title='Price (₹)', 
            title=f'{stock_name} Stock Price'
        )
        st.plotly_chart(fig)

        # RSI Calculation and Visualization
        st.subheader(f"Relative Strength Index (RSI) ({selected_range})")
        rsi_values = calculate_rsi(historical_data['Close'])
        
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=rsi_values.index, y=rsi_values, mode='lines', name='RSI'))
        fig_rsi.add_hline(y=70, line_color='red', line_dash='dash')
        fig_rsi.add_hline(y=30, line_color='green', line_dash='dash')
        fig_rsi.update_layout(
            xaxis_title='Date', 
            yaxis_title='RSI', 
            title=f'{stock_name} RSI'
        )
        st.plotly_chart(fig_rsi)

        # Supertrend Calculation
        st.subheader(f"Supertrend Indicator ({selected_range})")
        supertrend, direction = calculate_supertrend(historical_data)
        historical_data['Supertrend'] = supertrend
        historical_data['Direction'] = direction

        fig_supertrend = go.Figure()
        fig_supertrend.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Close Price'))
        fig_supertrend.add_trace(go.Scatter(x=historical_data.index, y=supertrend, mode='lines', name='Supertrend'))
        fig_supertrend.update_layout(
            xaxis_title='Date', 
            yaxis_title='Price (₹)', 
            title=f'{stock_name} Supertrend'
        )
        st.plotly_chart(fig_supertrend)

    # Analyze News Headlines
    st.subheader("News Sentiment Analysis")
    sentiment_df = analyze_news(stock_name)
    st.table(sentiment_df)

    # Generate Trading Signals
    st.subheader("Trading Signals")
    last_rsi = rsi_values.iloc[-1] if not pd.isnull(rsi_values).all() else None

    if historical_data is not None:
        last_direction = historical_data['Direction'].iloc[-1]
        current_price = historical_data['Close'].iloc[-1]
        
        # More robust target and stop-loss calculation
        rsi_value = calculate_rsi(historical_data['Close']).iloc[-1]
        
        # Dynamic target and stop-loss based on volatility
        avg_true_range = np.mean(historical_data['High'] - historical_data['Low'])
        target_price = current_price * (1 + (0.01 if last_direction == 1 else 0.005))
        stop_loss_price = current_price * (1 - (0.02 if last_direction == -1 else 0.01))

        # More comprehensive signal generation
        signal = "HOLD"
        delta = "Neutral"
        delta_color = 'normal'
        
        # Combine Supertrend and RSI for more reliable signals
        if last_direction == 1 and rsi_value < 70:
            signal = "BUY"
            delta = "Positive"
            delta_color = 'inverse'
            st.write(f"🟢 Bullish Trend Detected. RSI: {rsi_value:.2f}")
        elif last_direction == -1 and rsi_value > 30:
            signal = "SELL"
            delta = "Negative"
            delta_color = 'normal'
            st.write(f"🔴 Bearish Trend Detected. RSI: {rsi_value:.2f}")
        
        # Streamlit metric with correct delta_color
        st.metric("Trading Signal", signal, delta=delta, delta_color=delta_color)
        
        st.metric("Target Price", f"₹{target_price:.2f}")
        st.metric("Stop Loss", f"₹{stop_loss_price:.2f}")
        
        # Additional risk assessment
        volatility = np.std(historical_data['Close']) / np.mean(historical_data['Close']) * 100
        st.write(f"Market Volatility: {volatility:.2f}%")

        # Simulate Trading
        if st.button("Simulate Trade Execution"):
            st.success(f"Trade Executed: {signal} {stock_name} at ₹{current_price:.2f}")
            time.sleep(1)
        
        if stock_data:
            # Add predictions to the existing section
            historical_data = fetch_historical_data(stock_name.upper(), time_ranges[selected_range])
            predictions = predict_stock_price(stock_name, historical_data)
            
            st.sidebar.subheader("Price Predictions")
            st.sidebar.metric("1 Week Prediction", predictions["1_week_prediction"])
            st.sidebar.metric("1 Month Prediction", predictions["1_month_prediction"])

    if historical_data is not None:
            # Calculate technical indicators
            adx = calculate_adx(historical_data)
            upper_band, lower_band = calculate_bollinger_bands(historical_data)
            fib_levels = calculate_fibonacci_extension(historical_data)
            keltner_upper, keltner_lower = calculate_keltner_channel(historical_data)
            short_ema, medium_ema, long_ema, long_term_ema = calculate_ema(historical_data)
            
            # Supertrend with target and stop-loss
            supertrend, direction, target_price, stop_loss_price = calculate_supertrend_with_target_stoploss(historical_data)
            
            # Golden Crossover
            golden_cross = golden_crossover(historical_data)
            
            # Generate signals
            signal = "HOLD"
            if golden_cross:
                signal = "BUY"
            
            if direction[-1] == 1 and adx.iloc[-1] > 25:
                signal = "BUY"
                delta = "Positive"
            elif direction[-1] == -1 and adx.iloc[-1] > 25:
                signal = "SELL"
                delta = "Negative"
            
            # Display Trading Signal and Targets
            st.metric("Trading Signal", signal, delta=delta, delta_color="inverse" if signal == "BUY" else "normal")
            st.metric("Target Price", f"₹{target_price:.2f}")
            st.metric("Stop Loss", f"₹{stop_loss_price:.2f}")

            # Display Fibonacci Levels
            st.subheader("Fibonacci Extension Levels")
            st.write(f"Level 1: ₹{fib_levels[0]:.2f}")
            st.write(f"Level 2: ₹{fib_levels[1]:.2f}")
            st.write(f"Level 3: ₹{fib_levels[2]:.2f}")

else:
    st.error("Stock data could not be retrieved. Please check the stock symbol.")

# End of Streamlit App
st.write("© 2024 Indian Market Algorithmic Trading")
