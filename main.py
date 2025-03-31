from fastapi import FastAPI
import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()

# Function to calculate volatility (standard deviation) of price
def calculate_volatility(prices):
    return np.std(prices)

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Function to calculate moving average
def calculate_moving_average(prices, window=20):
    return np.mean(prices[-window:])

# Function to categorize risk based on volatility, RSI, price, and moving average
def categorize_risk(volatility, rsi, price, moving_avg):
    if volatility > 0.05 and rsi > 70 and price > moving_avg:
        return "High Risk"
    if 0.02 < volatility <= 0.05 and 40 < rsi < 60:
        return "Medium Risk"
    if volatility <= 0.02 and rsi < 30:
        return "Low Risk"
    return "Medium Risk"

# Function to get price data from CoinGecko API
def get_coingecko_data(token_id: str):
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': '30'}
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    return pd.DataFrame(data['prices'], columns=['timestamp', 'price'])

# Function to get NFT data from CoinGecko API
def get_nft_data_from_coingecko(token_id: str):
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/nfts"
    response = requests.get(url)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    # Return data in a DataFrame
    return pd.DataFrame(data)

# Function to get data from The Graph API
def get_the_graph_data():
    url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
    query = """
    {
      tokenDayData(first: 5, orderBy: date, orderDirection: desc) {
        date
        priceUSD
        volumeUSD
      }
    }
    """
    response = requests.post(url, json={'query': query})
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    token_data = data['data']['tokenDayData']
    return pd.DataFrame(token_data)

# Function to get data from Dune Analytics API
def get_dune_data():
    dune_url = "https://api.dune.com/api/v1/query/<YOUR_QUERY_ID>/results"
    headers = {"Authorization": "Bearer <YOUR_API_KEY>"}
    response = requests.get(dune_url, headers=headers)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    return pd.DataFrame(data['data'])

# Endpoint to get the token status (including risk)
@app.get("/token_status")
async def token_status(token_id: str):
    df = get_coingecko_data(token_id)
    if df is None:
        return {"error": "Failed to fetch data from CoinGecko"}

    volatility = calculate_volatility(df['price'])
    rsi = calculate_rsi(df['price'])
    moving_avg = calculate_moving_average(df['price'])
    current_price = df['price'].iloc[-1]
    
    risk_category = categorize_risk(volatility, rsi, current_price, moving_avg)
    
    return {
        "token_id": token_id,
        "current_price": current_price,
        "volatility": volatility,
        "rsi": rsi,
        "moving_average": moving_avg,
        "risk_category": risk_category
    }

# Endpoint to get NFT status (including trending analysis)
@app.get("/nft_status")
async def nft_status(token_id: str):
    df = get_nft_data_from_coingecko(token_id)
    if df is None:
        return {"error": "Failed to fetch data from CoinGecko NFTs"}
    
    # Analyzing NFT trends, volume, and other metrics
    if len(df) == 0:
        return {"error": "No NFT data available for this token."}
    
    trending_nfts = df[df['volume'] > df['volume'].quantile(0.75)]  # Example: Trending if volume is in top 25%
    
    return {
        "token_id": token_id,
        "total_nfts": len(df),
        "trending_nfts": trending_nfts.to_dict(),
    }

# Endpoint to generate automatic insights (using ARIMA for price prediction)
@app.get("/generate_insights")
async def generate_insights():
    df = get_coingecko_data("bitcoin")
    if df is None:
        return {"error": "Failed to fetch data from CoinGecko"}
    
    model = ARIMA(df['price'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)
    
    last_price = df['price'].iloc[-1]
    predicted_price = forecast[0]
    
    if predicted_price > last_price:
        insight = "Market trend is positive, consider buying."
    else:
        insight = "Market trend is negative, consider selling."
    
    return {"insight": insight, "forecast": forecast.tolist()}

# Endpoint to get data from The Graph (e.g., Uniswap)
@app.get("/the_graph_risk_analysis")
async def the_graph_risk_analysis():
    df = get_the_graph_data()
    if df is None:
        return {"error": "Failed to fetch data from The Graph"}
    
    volatility = calculate_volatility(df['priceUSD'])
    rsi = calculate_rsi(df['priceUSD'])
    moving_avg = calculate_moving_average(df['priceUSD'])
    current_price = df['priceUSD'].iloc[0]
    
    risk_category = categorize_risk(volatility, rsi, current_price, moving_avg)
    
    return {
        "current_price": current_price,
        "volatility": volatility,
        "rsi": rsi,
        "moving_average": moving_avg,
        "risk_category": risk_category
    }

# Endpoint to get data from Dune Analytics
@app.get("/dune_data")
async def dune_data():
    df = get_dune_data()
    if df is None:
        return {"error": "Failed to fetch data from Dune"}
    
    return {"data": df.to_dict()}
