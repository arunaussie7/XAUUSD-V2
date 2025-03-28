import MetaTrader5 as mt5

# Initialize connection to MT5
mt5.initialize(server="Exness-MT5Trial8", login=79700174, password="Botmudra.com@01")

# Check if connection is successful
if not mt5.initialize():
    print("Failed to connect:", mt5.last_error())
else:
    print("✅ Connected to MT5 successfully!")

# Shutdown connection after testing
mt5.shutdown() 