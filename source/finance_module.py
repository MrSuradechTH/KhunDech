import yfinance as yf

async def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if hist.empty:
        return f"Could not find data for {ticker}."
    info = stock.info
    current_price = info.get('currentPrice', 'N/A')
    return f"Stock: {ticker.upper()} | Price: {current_price} | Sector: {info.get('sector', 'N/A')}"