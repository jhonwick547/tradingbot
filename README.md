# TradingBot

A Python-based automated trading bot designed for Binance Futures. The bot uses machine learning and technical indicators to generate buy/sell signals and manage trades efficiently.

---

## Features

- **Automated Trading**: Handles trade execution, stop-loss, and take-profit orders.
- **Technical Analysis**: Utilizes indicators like RSI, MACD, Bollinger Bands, and more.
- **Machine Learning Integration**: Optional support for trained models to enhance trading signals.
- **Cooldown Mechanism**: Prevents over-trading by introducing a 5-minute cooldown period.
- **Error Logging**: Logs all activities and errors for easy debugging.
- **Testnet Support**: Allows testing strategies without real monetary risk.

---

## Prerequisites

- Python 3.10 or later
- Binance API Key and Secret
- Libraries:
  - `ccxt`
  - `pandas`
  - `ta`
  - `joblib`
  - `scikit-learn`
  - `logging`

Install dependencies:
```bash
pip install ccxt pandas ta joblib scikit-learn
```

---

## Usage

1. **Setup API Keys**:
   Replace `YOUR_API_KEY` and `YOUR_API_SECRET` in the script with your Binance API credentials.

2. **Run the Bot**:
   ```bash
   python bot.py
   ```

3. **Optional: Train and Load a Model**:
   Place the trained model in the project directory and specify its path during initialization.

---

## Key Configuration

- **Symbols**: Set the trading pairs in the `symbols` list (e.g., `['XRPUSDT', 'BTCUSDT']`).
- **Risk Management**:
  - Balance percentage: `balance_pct = 0.1` (10% of balance per trade)
  - Stop loss: `stop_loss_pct = 0.01` (1% stop loss)
  - Take profit: `take_profit_pct = 0.03` (3% take profit)
- **Timeframe**: Currently set to `5m` (5-minute candles).

---

## File Structure

```
TradingBot/
│
├── bot.py         # Main trading bot script
├── README.md      # Project documentation
└── .gitignore     # Excludes unnecessary files (e.g., logs)
```

---

## Logs

The bot creates a `trading_bot.log` file to track activities and errors. Check this file to monitor trades and debug issues.

---

## Disclaimer

This bot is for educational purposes only. Trading cryptocurrencies involves significant risk, and this bot should not be used with real funds unless thoroughly tested.
