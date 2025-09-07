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
# This defines the structure of the data we expect from the cBot
class MarketData(BaseModel):
    closes: List[float] = Field(..., min_items=50)
    ema_fast: List[float] = Field(..., min_items=50)
    ema_slow: List[float] = Field(..., min_items=50)
    adx: List[float] = Field(..., min_items=50)

# --- API Endpoint ---
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
        # 1. Check for Trend Strength (using ADX)
        # We'll use a common threshold of 25 for the ADX.
        if latest_adx > 25:
            # If trend is strong, determine its direction
            # 2. Check for Trend Direction (using EMA crossover)
            if latest_ema_fast > latest_ema_slow:
                regime = "TRENDING_UP"
            else:
                regime = "TRENDING_DOWN"
        
        # If ADX is not above 25, the regime remains "RANGING".

        return {"regime": regime, "adx_value": latest_adx}

    except Exception as e:
        # Handle potential errors, like empty lists or calculation issues
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during regime analysis: {str(e)}"
        )
