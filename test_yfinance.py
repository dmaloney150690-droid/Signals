import yfinance as yf

ticker = 'AAPL'
print(f"Attempting to download data for {ticker}...")

try:
    data = yf.download(ticker, period='30d')
    if data.empty:
        print("\nDownload failed: No data was returned.")
        print("This confirms a network issue (firewall, proxy) or a problem with the yfinance library.")
    else:
        print("\nSuccess! Data downloaded correctly:")
        print(data.tail())
except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("This confirms a network connectivity problem (firewall, proxy) or an installation issue.")
