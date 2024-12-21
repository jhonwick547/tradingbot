import os
import ccxt
import pandas as pd
import ta
import time
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler for logging
file_handler = logging.FileHandler('trading_bot.log')
file_handler.setLevel(logging.INFO)

# Console handler for logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter for logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Adding handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class TradingBot:
    def __init__(self, api_key, api_secret, model_path=None, testnet=True):
        self.api_key = api_key
        self.secret = api_secret

        # Initialize Binance API using ccxt
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Futures market
                'recvWindow': 6000,  # Set recvWindow to avoid time sync issues
            }
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)  # Enable Testnet mode
            self.exchange.urls['api']['public'] = 'https://testnet.binancefuture.com/fapi/v1'
            self.exchange.urls['api']['private'] = 'https://testnet.binancefuture.com/fapi/v1'
            self.exchange.urls['api']['fapiPublic'] = 'https://testnet.binancefuture.com/fapi/v1'
            self.exchange.urls['api']['fapiPrivate'] = 'https://testnet.binancefuture.com/fapi/v1'

        # Synchronize time to avoid timing issues with API requests
        self.exchange.load_time_difference()

        # Trading settings
        self.symbols = ['1000PEPEUSDT', 'XRPUSDT', 'ETHUSDT']
        self.timeframe = '5m'
        self.balance_pct = 0.1  # Percentage of balance used per trade (10%)
        self.stop_loss_pct = 0.01  # 1% stop loss
        self.take_profit_pct = 0.03  # 3% take profit
        self.starting_balance = 100  # Start with $100
        self.wins = 0
        self.losses = 0
        self.active_trades = {symbol: None for symbol in self.symbols}  # Track active trades

        # ML Model
        self.model = None
        if model_path:
            self.model = joblib.load(model_path)

        # Cooldown mechanism
        self.last_trade_time = {symbol: 0 for symbol in self.symbols}  # Track last trade time per symbol
        self.cooldown_period = 300  # Cooldown period of 5 minutes (300 seconds)

    def fetch_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for trading signals."""
        df['MA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['MA_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['MACD'] = ta.trend.macd(df['close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['close'])
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['Bollinger_High'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
        df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
        df['Stochastic_Oscillator'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        df.fillna(0, inplace=True)  # Replace NaN values with 0
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> str:
        """Generate buy/sell signals based on multiple indicators."""
        if 'RSI' not in df.columns:
            logger.error(f"RSI indicator not calculated for {symbol}.")
            return 'neutral'

        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_signal'].iloc[-1]
        ema_20 = df['EMA_20'].iloc[-1]
        ema_50 = df['EMA_50'].iloc[-1]
        cci = df['CCI'].iloc[-1]

        if (rsi < 60 and macd > macd_signal and ema_20 > ema_50) or (df['close'].iloc[-1] < df['Bollinger_Low'].iloc[-1] and df['Stochastic_Oscillator'].iloc[-1] < 50 and cci < -50):
            logger.info(f"Buy signal generated for {symbol}")
            return 'buy'

        elif (rsi > 40 and macd < macd_signal and ema_20 < ema_50) or (df['close'].iloc[-1] > df['Bollinger_High'].iloc[-1] and df['Stochastic_Oscillator'].iloc[-1] > 50 and cci > 50):
            logger.info(f"Sell signal generated for {symbol}")
            return 'sell'

        logger.info(f"Neutral signal for {symbol} - RSI: {rsi}, MACD: {macd}, MACD Signal: {macd_signal}")
        return 'neutral'

    def calculate_position_size(self, symbol: str, entry_price: float) -> float:
        """Calculate position size based on risk management for a $100 balance."""
        balance = min(self.starting_balance, self.exchange.fetch_balance()['free']['USDT'])
        risk_amount = balance * self.balance_pct  # Risk 10% of balance
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        position_size = risk_amount / (entry_price - stop_loss)
        return min(position_size, balance / entry_price)

    def execute_trade(self, symbol: str, signal: str):
        """Execute a trade based on the signal, with cooldown checks."""
        try:
            # Check if the cooldown period has passed
            current_time = time.time()
            if current_time - self.last_trade_time[symbol] < self.cooldown_period:
                logger.info(f"Cooldown active for {symbol}. Skipping trade.")
                return

            df = self.fetch_data(symbol)
            if df is None or df.empty:
                logger.error(f"No data fetched for {symbol}. Skipping trade execution.")
                return

            df = self.calculate_indicators(df)
            entry_price = df['close'].iloc[-1]

            if signal == 'buy':
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                take_profit = entry_price * (1 + self.take_profit_pct)
            elif signal == 'sell':
                stop_loss = entry_price * (1 + self.stop_loss_pct)
                take_profit = entry_price * (1 - self.take_profit_pct)
            else:
                logger.info(f"Invalid signal '{signal}' for {symbol}. No trade executed.")
                return

            position_size = self.calculate_position_size(symbol, entry_price)
            if position_size <= 0:
                logger.error(f"Calculated position size is non-positive for {symbol}. Skipping trade.")
                return

            logger.info(f"Executing {signal} order for {symbol} with position size {position_size}")

            # Execute Buy or Sell order
            if signal == 'buy':
                order = self.exchange.create_market_buy_order(symbol, position_size)
                logger.info(f"Buy order executed: {order}")
            elif signal == 'sell':
                order = self.exchange.create_market_sell_order(symbol, position_size)
                logger.info(f"Sell order executed: {order}")

            # Set stop-loss and take-profit orders
            if signal == 'buy':
                self.exchange.create_order(symbol, 'STOP_MARKET', 'sell', position_size, params={'stopPrice': stop_loss})
                self.exchange.create_order(symbol, 'LIMIT', 'sell', position_size, price=take_profit)
            elif signal == 'sell':
                self.exchange.create_order(symbol, 'STOP_MARKET', 'buy', position_size, params={'stopPrice': stop_loss})
                self.exchange.create_order(symbol, 'LIMIT', 'buy', position_size, price=take_profit)

            logger.info(f"Stop loss and take profit orders placed for {symbol}: SL={stop_loss}, TP={take_profit}")

            # Update last trade time after successful trade
            self.last_trade_time[symbol] = current_time

        except Exception as e:
            logger.error(f"Failed to execute trade for {symbol}: {str(e)}")

    def start_trading(self):
        """Start trading loop that fetches data and executes trades."""
        while True:
            for symbol in self.symbols:
                try:
                    df = self.fetch_data(symbol)
                    if df is None:
                        continue

                    df = self.calculate_indicators(df)
                    signal = self.generate_signals(df, symbol)
                    if signal in ['buy', 'sell']:
                        self.execute_trade(symbol, signal)

                except Exception as e:
                    logger.error(f"Error in trading loop for {symbol}: {str(e)}")

            time.sleep(300)  # Wait 5 minutes before fetching new data




if __name__ == "__main__":
    # Hardcoded API key and secret (not recommended for production)
    api_key = 'YOUR_API_KEY' # Replace with your actual API key
    api_secret = 'YOUR_API_SECRET'  # Replace with your actual API secret
    
    bot = TradingBot(api_key, api_secret, testnet=True)
    bot.start_trading()
