from crewai.tools import BaseTool
import yfinance as yf

class FetchStockDataTool(BaseTool):
    name: str = "fetch_stock_data"
    description: str = "Useful for fetching historical price data and basic info for a given stock symbol. Input should be a ticker symbol (e.g., 'NVDA')."

    def _run(self, symbol: str) -> str:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")
        info = ticker.info
        return str({
            "price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "history": hist.tail(5).to_dict(),
            "summary": info.get("longBusinessSummary")
        })

class GetMarketSentimentTool(BaseTool):
    name: str = "get_market_sentiment"
    description: str = "Useful for analyzing the current trend of a stock based on moving averages. Input should be a ticker symbol (e.g., 'NVDA')."

    def _run(self, symbol: str) -> str:
        data = yf.download(symbol, period="6mo", interval="1d")
        if data.empty:
            return "No data available"

        close_prices = data['Close']
        ma50 = close_prices.rolling(window=50).mean().iloc[-1]
        ma200 = close_prices.rolling(window=200).mean().iloc[-1]

        return "Bullish" if ma50 > ma200 else "Bearish"

# Instantiate the tools
fetch_stock_data = FetchStockDataTool()
get_market_sentiment = GetMarketSentimentTool()
