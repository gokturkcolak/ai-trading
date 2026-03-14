import os
import time
import logging
import feedparser
import pandas as pd
import numpy as np
import pandas_ta as ta
import ccxt
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
from xgboost import XGBClassifier

# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = '4h'
LEVERAGE = int(os.getenv("LEVERAGE", 5))
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() == "true"

# --- INITIALIZE APIs & MODELS ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
ai_model = genai.GenerativeModel('gemini-2.0-flash')

# Load the pre-trained XGBoost model
MODEL_PATH = "xgboost_model.json"
if os.path.exists(MODEL_PATH):
    ml_model = XGBClassifier()
    ml_model.load_model(MODEL_PATH)
    logging.info(f"✅ Pre-trained XGBoost model loaded from {MODEL_PATH}")
else:
    logging.error(f"❌ Model file {MODEL_PATH} not found! Run the notebook first.")
    exit(1)

def get_binance_client():
    config = {
        'options': {'defaultType': 'future'},
        'enableRateLimit': True,
    }
    
    if USE_TESTNET:
        config['apiKey'] = os.getenv("TESTNET_API_KEY")
        config['secret'] = os.getenv("TESTNET_API_SECRET")
        exchange = ccxt.binance(config)
        exchange.set_sandbox_mode(True)
        logging.info("Connected to BINANCE TESTNET")
    else:
        config['apiKey'] = os.getenv("BINANCE_API_KEY")
        config['secret'] = os.getenv("BINANCE_API_SECRET")
        exchange = ccxt.binance(config)
        logging.info("Connected to BINANCE MAINNET")
    
    return exchange

exchange = get_binance_client()

# --- CORE FUNCTIONS ---

def fetch_data(limit=500):
    """Fetches live OHLCV and calculates technical indicators."""
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # 1. Feature Engineering
        df['RSI'] = df.ta.rsi(length=14)
        df['ATR'] = df.ta.atr(length=14)
        df['ROC'] = df.ta.roc(length=12)
        
        # Bollinger Bands
        bbands = df.ta.bbands(length=20, std=2)
        df['BBM'] = bbands.iloc[:, 1]
        df['BBB'] = bbands.iloc[:, 3]
        df['BBP'] = bbands.iloc[:, 4]
        
        # Volume & Momentum
        df['OBV'] = df.ta.obv()
        df['vol_change'] = df['volume'].pct_change()
        
        # Donchian Channels
        dc = df.ta.donchian(lower_length=20, upper_length=20)
        df['DCL'] = dc.iloc[:, 0]
        df['DCU'] = dc.iloc[:, 2]
        
        return df.dropna()
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

def get_prediction(df):
    """Feeds the latest indicators into the pre-trained model."""
    try:
        features = ['RSI', 'ATR', 'ROC', 'BBM', 'BBB', 'BBP', 'OBV', 'vol_change']
        latest_features = df[features].tail(1)
        
        # Ask the loaded model for a prediction
        prediction = ml_model.predict(latest_features)[0]
        return prediction
    except Exception as e:
        logging.error(f"Error during ML prediction: {e}")
        return None

def get_latest_news():
    """Fetches recent headlines for AI context."""
    try:
        # Fetching from a general crypto RSS feed
        feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeed/rss/all/")
        headlines = [entry.title for entry in feed.entries[:8]]
        return " | ".join(headlines)
    except Exception as e:
        logging.error(f"Error fetching news: {e}")
        return "No recent news available."

def ai_strategic_filter(ml_pred, news_context, signal_type):
    """Gemini-powered filter to block trades based on news context."""
    direction = "LONG" if signal_type == 1 else "SHORT"
    prompt = f"""
    You are a Strategic Risk Manager for an automated 24/7 crypto trading bot.
    The technical system (ML + Indicators) wants to open a {direction} position on {SYMBOL}.
    
    LATEST NEWS HEADLINES:
    {news_context}
    
    TASK: Determine if there is a 'Black Swan' event (exchange hack, regulatory ban, market crash news) 
    that directly contradicts this {direction} move.
    
    RESPONSE FORMAT: Answer only with one word: 'PROCEED' or 'BLOCK'.
    """
    try:
        response = ai_model.generate_content(prompt)
        decision = response.text.strip().upper()
        return "PROCEED" in decision
    except Exception as e:
        logging.error(f"AI Filter Error: {e}")
        return False # Safety default

def execute_trade(direction):
    """Executes order on Binance Futures."""
    try:
        # 1. Set Leverage
        exchange.set_leverage(LEVERAGE, SYMBOL)
        
        # 2. Risk Management: Use small % of total balance
        balance = exchange.fetch_balance()['total']['USDT']
        ticker = exchange.fetch_ticker(SYMBOL)
        price = ticker['last']
        
        # Risking 10% of balance with leverage
        qty = (balance * 0.10 * LEVERAGE) / price
        
        side = 'buy' if direction == 1 else 'sell'
        logging.info(f"PLACING {side.upper()} ORDER: {qty:.4f} {SYMBOL} at {price}")
        
        order = exchange.create_market_order(SYMBOL, side, qty)
        return order
    except Exception as e:
        logging.error(f"Order Execution Failed: {e}")
        return None

# --- MAIN AGENT LOOP ---

def run_agent():
    logging.info(f"AI TRADING AGENT STARTED ON {SYMBOL} (4H)")
    
    while True:
        try:
            # 1. Sense (Fetch Data)
            df = fetch_data()
            if df is None:
                time.sleep(60)
                continue
                
            latest_row = df.iloc[-1]
            
            # 2. Think (ML Signal)
            ml_pred = get_prediction(df)
            if ml_pred is None:
                time.sleep(60)
                continue
            
            # 3. Think (Strategy Signal)
            signal = 0
            # Breakout Up + ML Agreement
            if latest_row['close'] > df['DCU'].iloc[-2] and ml_pred == 1:
                signal = 1
            # Breakout Down + ML Agreement (ML 0 = Down)
            elif latest_row['close'] < df['DCL'].iloc[-2] and ml_pred == 0:
                signal = -1
            
            # 4. Act (AI Filter & Execution)
            if signal != 0:
                news = get_latest_news()
                if ai_strategic_filter(ml_pred, news, signal):
                    logging.info(f"AI Filter PASSED. Executing signal: {signal}")
                    execute_trade(signal)
                else:
                    logging.info("AI Filter BLOCKED the signal due to News Risk.")
            else:
                logging.info("No valid ML + Breakout signal detected.")

            # Sleep until next 4h candle (checks every 5 mins to stay updated)
            logging.info("Waiting for next evaluation cycle...")
            time.sleep(300) 

        except Exception as e:
            logging.error(f"Agent Loop Crash: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_agent()
