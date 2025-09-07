# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np

# Initialize the FastAPI app
app = FastAPI(
    title="AI Overseer - Regime Filter",
    description="Analyzes market data to determine the current trading regime."
)

# --- Pydantic Models for Request Data ---
class MarketData(BaseModel):
    closes: List[float] = Field(..., min_items=50)
    ema_fast: List[float] = Field(..., min_items=50)
    ema_slow: List[float] = Field(..., min_items=50)
    adx: List[float] = Field(..., min_items=50)

# --- Health Check Endpoint (NEW) ---
# This gives Render a URL to check if the service is online.
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# --- Main Regime Analysis Endpoint ---
@app.post("/regime")
async def get_market_regime(data: MarketData):
    """
    Analyzes the provided market data and returns the current regime.
    Regimes: TRENDING_UP, TRENDING_DOWN, RANGING.
    """
    try:
        # Get the most recent values from the lists provided by the cBot
        latest_adx = data.adx[-1]
        latest_ema_fast = data.ema_fast[-1]
        latest_ema_slow = data.ema_slow[-1]

        regime = "RANGING" # Default to RANGING

        # --- Regime Classification Logic ---
        if latest_adx > 25:
            if latest_ema_fast > latest_ema_slow:
                regime = "TRENDING_UP"
            else:
                regime = "TRENDING_DOWN"
        
        return {"regime": regime, "adx_value": latest_adx}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during regime analysis: {str(e)}"
        )
