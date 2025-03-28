import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# Initialize MT5 connection
mt5.initialize(server="Exness-MT5Trial8", login=79700174, password="Botmudra.com@01")

# Check connection status
if not mt5.initialize():
    print("Failed to connect:", mt5.last_error())
    quit()

# Fetch XAUUSD 5-minute interval data (1000 candles)
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)

# Convert to DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Save to CSV
df.to_csv("xauusd_M5.csv", index=False)

# Print sample data
print(df.head())

# Close MT5 connection
mt5.shutdown() 