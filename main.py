"""
AI Overseer Market Analysis API
Compatible with cTrader AI-Enhanced Scalping Bot
Deploy on Render alongside your existing AI Overseer
"""

import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from typing import Dict, Any, Optional
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cTrader access

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
PORT = int(os.getenv('PORT', 5000))
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Market state definitions matching cTrader bot
MARKET_STATES = [
    "Trending_Strong",
    "Trending_Weak", 
    "Ranging",
    "Volatile",
    "Choppy",
    "Unknown"
]

TRADING_BIASES = [
    "Bullish_Strong",
    "Bullish_Weak",
    "Neutral",
    "Bearish_Weak",
    "Bearish_Strong"
]

class MarketAnalyzer:
    """Core market analysis logic"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.cache_duration = timedelta(minutes=15)  # Cache results for 15 minutes
        
    def parse_market_data(self, data: str) -> Dict[str, Any]:
        """Parse the market data string from cTrader into structured format"""
        parsed = {}
        lines = data.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Try to parse numerical values
                try:
                    if 'pips' in value:
                        value = float(value.replace('pips', '').strip())
                    elif '%' in value:
                        value = float(value.replace('%', '').strip())
                    else:
                        value = float(value)
                except:
                    pass  # Keep as string if not numeric
                    
                parsed[key] = value
                
        return parsed
    
    def calculate_market_metrics(self, parsed_data: Dict) -> Dict[str, float]:
        """Calculate additional metrics for analysis"""
        metrics = {}
        
        # RSI-based momentum
        rsi = parsed_data.get('rsi', 50)
        if rsi < 30:
            metrics['momentum_score'] = -2
        elif rsi < 40:
            metrics['momentum_score'] = -1
        elif rsi > 70:
            metrics['momentum_score'] = 2
        elif rsi > 60:
            metrics['momentum_score'] = 1
        else:
            metrics['momentum_score'] = 0
            
        # Volatility assessment from ATR
        atr = parsed_data.get('atr', 10)
        if atr < 5:
            metrics['volatility_level'] = 0.2
        elif atr < 10:
            metrics['volatility_level'] = 0.4
        elif atr < 15:
            metrics['volatility_level'] = 0.6
        elif atr < 20:
            metrics['volatility_level'] = 0.8
        else:
            metrics['volatility_level'] = 1.0
            
        # Trend strength from 24h change
        change_24h = parsed_data.get('24h_change', 0)
        metrics['trend_strength'] = min(abs(change_24h) / 2, 1.0)
        
        # Recent performance bias
        wins = parsed_data.get('recent_wins', 0)
        losses = parsed_data.get('recent_losses', 0)
        total_trades = wins + losses
        
        if total_trades > 0:
            metrics['win_rate'] = wins / total_trades
        else:
            metrics['win_rate'] = 0.5
            
        return metrics
    
    def determine_market_state(self, metrics: Dict[str, float], parsed_data: Dict) -> str:
        """Determine the current market state based on metrics"""
        volatility = metrics['volatility_level']
        trend_strength = metrics['trend_strength']
        
        if trend_strength > 0.7:
            return "Trending_Strong"
        elif trend_strength > 0.4:
            return "Trending_Weak"
        elif volatility > 0.7:
            return "Volatile"
        elif volatility < 0.3 and trend_strength < 0.2:
            return "Ranging"
        elif volatility > 0.5 and trend_strength < 0.3:
            return "Choppy"
        else:
            return "Ranging"
    
    def determine_trading_bias(self, metrics: Dict[str, float], parsed_data: Dict) -> str:
        """Determine trading bias based on indicators"""
        momentum = metrics['momentum_score']
        change_24h = parsed_data.get('24h_change', 0)
        rsi = parsed_data.get('rsi', 50)
        
        # Combine signals
        bias_score = 0
        
        # Momentum contribution
        bias_score += momentum * 0.4
        
        # 24h change contribution
        if change_24h > 0.5:
            bias_score += 1
        elif change_24h > 0:
            bias_score += 0.5
        elif change_24h < -0.5:
            bias_score -= 1
        elif change_24h < 0:
            bias_score -= 0.5
            
        # RSI contribution
        if rsi > 60:
            bias_score += 0.3
        elif rsi < 40:
            bias_score -= 0.3
            
        # Map score to bias
        if bias_score >= 1.5:
            return "Bullish_Strong"
        elif bias_score >= 0.5:
            return "Bullish_Weak"
        elif bias_score <= -1.5:
            return "Bearish_Strong"
        elif bias_score <= -0.5:
            return "Bearish_Weak"
        else:
            return "Neutral"
    
    def calculate_confidence_level(self, metrics: Dict[str, float], state: str) -> float:
        """Calculate confidence level based on market conditions"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on win rate
        win_rate = metrics.get('win_rate', 0.5)
        confidence += (win_rate - 0.5) * 0.3
        
        # Adjust based on market state
        if state in ["Trending_Strong", "Ranging"]:
            confidence += 0.2
        elif state == "Choppy":
            confidence -= 0.2
            
        # Adjust based on volatility
        volatility = metrics['volatility_level']
        if volatility < 0.3 or volatility > 0.8:
            confidence -= 0.1
            
        return max(0.1, min(1.0, confidence))
    
    def calculate_risk_multiplier(self, confidence: float, state: str, bias: str) -> float:
        """Calculate suggested risk multiplier"""
        base_multiplier = 1.0
        
        # Confidence adjustment
        base_multiplier *= (0.5 + confidence)
        
        # Market state adjustment
        if state == "Trending_Strong" and bias != "Neutral":
            base_multiplier *= 1.2
        elif state == "Choppy":
            base_multiplier *= 0.7
        elif state == "Volatile":
            base_multiplier *= 0.8
            
        return max(0.5, min(2.0, base_multiplier))
    
    def identify_key_levels(self, parsed_data: Dict) -> str:
        """Identify key support/resistance levels"""
        pivot = parsed_data.get('daily_pivot', 0)
        current_price = parsed_data.get('current_price', 0)
        
        if pivot and current_price:
            # Simple pivot point calculation
            r1 = 2 * pivot - parsed_data.get('current_price', pivot)
            s1 = 2 * pivot - parsed_data.get('current_price', pivot)
            
            return f"Pivot: {pivot:.5f}, R1: {r1:.5f}, S1: {s1:.5f}"
        else:
            return "Key levels require more data"
    
    def get_strategy_recommendation(self, state: str, confidence: float) -> str:
        """Recommend trading strategy based on conditions"""
        if state == "Trending_Strong" and confidence > 0.7:
            return "Aggressive"
        elif state in ["Choppy", "Volatile"] or confidence < 0.4:
            return "Conservative"
        else:
            return "Normal"
    
    async def analyze_with_ai(self, market_data: str) -> Optional[Dict[str, Any]]:
        """Get AI analysis using OpenAI GPT"""
        try:
            prompt = f"""Analyze this forex market data and provide a JSON response:

{market_data}

Respond ONLY with JSON in this exact format:
{{
    "market_state": "Trending_Strong|Trending_Weak|Ranging|Volatile|Choppy",
    "trading_bias": "Bullish_Strong|Bullish_Weak|Neutral|Bearish_Weak|Bearish_Strong",
    "confidence_level": 0.0-1.0,
    "volatility_expectation": 0.0-1.0,
    "key_levels": "description of support/resistance",
    "recommended_strategy": "Conservative|Normal|Aggressive",
    "risk_multiplier": 0.5-2.0,
    "analysis_reasoning": "brief explanation"
}}"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use gpt-4 if available
                messages=[
                    {"role": "system", "content": "You are a forex market analyst providing JSON responses only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            ai_response = response.choices[0].message.content
            return json.loads(ai_response)
            
        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            return None
    
    def perform_analysis(self, market_data: str, use_ai: bool = True) -> Dict[str, Any]:
        """Main analysis function"""
        # Check cache
        cache_key = hash(market_data)
        if cache_key in self.analysis_cache:
            cached_result, cache_time = self.analysis_cache[cache_key]
            if datetime.now() - cache_time < self.cache_duration:
                logger.info("Returning cached analysis")
                return cached_result
        
        # Parse market data
        parsed_data = self.parse_market_data(market_data)
        metrics = self.calculate_market_metrics(parsed_data)
        
        # Try AI analysis first if enabled
        if use_ai and OPENAI_API_KEY != 'your-api-key-here':
            ai_result = self.analyze_with_ai(market_data)
            if ai_result:
                # Validate and use AI result
                result = {
                    "market_state": ai_result.get("market_state", "Unknown"),
                    "trading_bias": ai_result.get("trading_bias", "Neutral"),
                    "confidence_level": float(ai_result.get("confidence_level", 0.5)),
                    "volatility_expectation": float(ai_result.get("volatility_expectation", 0.5)),
                    "key_levels": ai_result.get("key_levels", ""),
                    "recommended_strategy": ai_result.get("recommended_strategy", "Normal"),
                    "risk_multiplier": float(ai_result.get("risk_multiplier", 1.0)),
                    "analysis_method": "AI-Enhanced",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache the result
                self.analysis_cache[cache_key] = (result, datetime.now())
                return result
        
        # Fallback to rule-based analysis
        state = self.determine_market_state(metrics, parsed_data)
        bias = self.determine_trading_bias(metrics, parsed_data)
        confidence = self.calculate_confidence_level(metrics, state)
        risk_mult = self.calculate_risk_multiplier(confidence, state, bias)
        
        result = {
            "market_state": state,
            "trading_bias": bias,
            "confidence_level": confidence,
            "volatility_expectation": metrics['volatility_level'],
            "key_levels": self.identify_key_levels(parsed_data),
            "recommended_strategy": self.get_strategy_recommendation(state, confidence),
            "risk_multiplier": risk_mult,
            "analysis_method": "Rule-Based",
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the result
        self.analysis_cache[cache_key] = (result, datetime.now())
        return result

# Initialize analyzer
analyzer = MarketAnalyzer()

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "service": "AI Overseer Market Analyzer",
        "version": "1.0.0",
        "endpoints": [
            "/analyze",
            "/health",
            "/clear_cache"
        ]
    })

@app.route('/health')
def health():
    """Health check for monitoring"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": OPENAI_API_KEY != 'your-api-key-here'
    })

@app.route('/analyze', methods=['POST'])
def analyze_market():
    """Main analysis endpoint for cTrader bot"""
    try:
        # Get market data from request
        data = request.get_json()
        
        if not data or 'market_data' not in data:
            return jsonify({
                "error": "Missing market_data in request"
            }), 400
        
        market_data = data['market_data']
        use_ai = data.get('use_ai', True)
        
        # Perform analysis
        analysis = analyzer.perform_analysis(market_data, use_ai)
        
        logger.info(f"Analysis completed: {analysis['market_state']} - {analysis['trading_bias']}")
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            "error": str(e),
            "market_state": "Unknown",
            "trading_bias": "Neutral",
            "confidence_level": 0.5,
            "volatility_expectation": 0.5,
            "key_levels": "Error in analysis",
            "recommended_strategy": "Conservative",
            "risk_multiplier": 1.0
        }), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear analysis cache"""
    analyzer.analysis_cache.clear()
    return jsonify({
        "status": "Cache cleared",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG_MODE)
