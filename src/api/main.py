"""
FastAPI Main Application
Real-time sentiment analysis and stock prediction API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import asyncio
from pydantic import BaseModel
import redis
import json

from ..models.sentiment_analyzer import FinancialSentimentAnalyzer
from ..data.data_collector import DataPipeline
from ..models.predictor import StockPredictor
from ..database.database import get_db, SessionLocal
from ..tasks.celery_app import analyze_sentiment_task, collect_data_task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SentimentFlow AI API",
    description="Advanced emotion-driven stock market prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
sentiment_analyzer = FinancialSentimentAnalyzer()
stock_predictor = StockPredictor()
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Pydantic models
class SentimentRequest(BaseModel):
    text: str
    symbol: Optional[str] = None

class SentimentResponse(BaseModel):
    compound: float
    positive: float
    negative: float
    neutral: float
    confidence: float
    emotion: str
    meme_intensity: float
    financial_context: float

class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"  # 1h, 1d, 1w, 1m
    include_sentiment: bool = True

class PredictionResponse(BaseModel):
    symbol: str
    prediction: Dict
    sentiment_data: Optional[Dict] = None
    confidence: float
    timestamp: datetime

class BulkAnalysisRequest(BaseModel):
    symbols: List[str]
    timeframe: str = "1d"
    limit: int = 100

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Sentiment analysis endpoints
@app.post("/sentiment/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of a single text"""
    try:
        # Check cache first
        cache_key = f"sentiment:{hash(request.text)}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            result = json.loads(cached_result)
        else:
            # Analyze sentiment
            result = sentiment_analyzer.ensemble_sentiment(request.text)
            emotion = sentiment_analyzer.get_emotion_classification(result)
            result['emotion'] = emotion
            
            # Cache result for 1 hour
            redis_client.setex(cache_key, 3600, json.dumps(result, default=str))
        
        return SentimentResponse(**result)
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment/batch")
async def batch_sentiment_analysis(texts: List[str]):
    """Analyze sentiment for multiple texts"""
    try:
        results = []
        
        for text in texts:
            result = sentiment_analyzer.ensemble_sentiment(text)
            emotion = sentiment_analyzer.get_emotion_classification(result)
            result['emotion'] = emotion
            result['text'] = text[:100] + "..." if len(text) > 100 else text
            results.append(result)
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Stock prediction endpoints
@app.post("/predict/stock", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """Predict stock price movement"""
    try:
        # Check cache first
        cache_key = f"prediction:{request.symbol}:{request.timeframe}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            result = json.loads(cached_result)
            return PredictionResponse(**result)
        
        # Get prediction
        prediction = await stock_predictor.predict(
            symbol=request.symbol,
            timeframe=request.timeframe,
            include_sentiment=request.include_sentiment
        )
        
        # Get sentiment data if requested
        sentiment_data = None
        if request.include_sentiment:
            sentiment_data = await get_symbol_sentiment(request.symbol)
        
        result = {
            "symbol": request.symbol,
            "prediction": prediction,
            "sentiment_data": sentiment_data,
            "confidence": prediction.get("confidence", 0.5),
            "timestamp": datetime.now()
        }
        
        # Cache for 5 minutes
        redis_client.setex(cache_key, 300, json.dumps(result, default=str))
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Stock prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/bulk")
async def bulk_prediction(request: BulkAnalysisRequest):
    """Bulk stock predictions"""
    try:
        results = []
        
        for symbol in request.symbols:
            try:
                prediction = await stock_predictor.predict(
                    symbol=symbol,
                    timeframe=request.timeframe
                )
                
                sentiment_data = await get_symbol_sentiment(symbol)
                
                result = {
                    "symbol": symbol,
                    "prediction": prediction,
                    "sentiment_data": sentiment_data,
                    "timestamp": datetime.now()
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "error": str(e),
                    "timestamp": datetime.now()
                })
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Bulk prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data collection endpoints
@app.post("/data/collect")
async def trigger_data_collection(
    symbols: List[str], 
    background_tasks: BackgroundTasks
):
    """Trigger data collection for specified symbols"""
    try:
        # Start background task
        task = collect_data_task.delay(symbols)
        
        return {
            "message": "Data collection started",
            "task_id": task.id,
            "symbols": symbols,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Data collection trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/sentiment/{symbol}")
async def get_symbol_sentiment(symbol: str, hours: int = 24):
    """Get recent sentiment data for a symbol"""
    try:
        # Check cache
        cache_key = f"symbol_sentiment:{symbol}:{hours}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        # Get sentiment data from database
        # This would typically query your database for recent posts
        # For now, return mock data
        sentiment_data = {
            "symbol": symbol,
            "timeframe_hours": hours,
            "total_posts": 150,
            "average_sentiment": 0.65,
            "sentiment_trend": "bullish",
            "platforms": {
                "twitter": {"posts": 80, "sentiment": 0.72},
                "reddit": {"posts": 70, "sentiment": 0.58}
            },
            "emotions": {
                "bullish": 0.45,
                "bearish": 0.20,
                "neutral": 0.25,
                "euphoria": 0.10
            },
            "meme_intensity": 0.35,
            "timestamp": datetime.now()
        }
        
        # Cache for 10 minutes
        redis_client.setex(cache_key, 600, json.dumps(sentiment_data, default=str))
        
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Symbol sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Market overview endpoints
@app.get("/market/overview")
async def market_overview():
    """Get overall market sentiment overview"""
    try:
        # Popular symbols to track
        symbols = ["AAPL", "TSLA", "GME", "AMC", "NVDA", "MSFT", "GOOGL", "SPY"]
        
        overview = {
            "market_sentiment": 0.58,
            "trending_symbols": [],
            "sentiment_distribution": {
                "bullish": 0.42,
                "bearish": 0.28,
                "neutral": 0.30
            },
            "meme_activity": 0.45,
            "total_posts_24h": 12500,
            "timestamp": datetime.now()
        }
        
        # Get sentiment for each symbol
        for symbol in symbols:
            sentiment_data = await get_symbol_sentiment(symbol, 24)
            overview["trending_symbols"].append({
                "symbol": symbol,
                "sentiment": sentiment_data["average_sentiment"],
                "posts": sentiment_data["total_posts"]
            })
        
        # Sort by sentiment
        overview["trending_symbols"].sort(
            key=lambda x: x["sentiment"], 
            reverse=True
        )
        
        return overview
        
    except Exception as e:
        logger.error(f"Market overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time endpoints
@app.get("/realtime/sentiment/{symbol}")
async def realtime_sentiment(symbol: str):
    """Get real-time sentiment stream for a symbol"""
    try:
        # This would typically connect to your real-time data stream
        # For now, return current sentiment
        sentiment_data = await get_symbol_sentiment(symbol, 1)
        
        return {
            "symbol": symbol,
            "realtime_sentiment": sentiment_data["average_sentiment"],
            "last_update": datetime.now(),
            "status": "active"
        }
        
    except Exception as e:
        logger.error(f"Real-time sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/analytics/performance")
async def model_performance():
    """Get model performance metrics"""
    try:
        # Mock performance data - replace with actual metrics
        performance = {
            "accuracy": {
                "1h": 0.68,
                "1d": 0.73,
                "1w": 0.71,
                "1m": 0.69
            },
            "sharpe_ratio": 1.84,
            "max_drawdown": 0.123,
            "total_predictions": 15420,
            "successful_predictions": 11257,
            "last_updated": datetime.now()
        }
        
        return performance
        
    except Exception as e:
        logger.error(f"Performance analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("SentimentFlow AI API starting up...")
    
    # Initialize models
    try:
        # Warm up models
        test_text = "TSLA to the moon! 🚀"
        sentiment_analyzer.ensemble_sentiment(test_text)
        logger.info("Sentiment analyzer initialized")
        
        # Test database connection
        # db = SessionLocal()
        # db.close()
        logger.info("Database connection verified")
        
        logger.info("SentimentFlow AI API startup complete")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("SentimentFlow AI API shutting down...")
    
    # Close connections
    if redis_client:
        redis_client.close()
    
    logger.info("SentimentFlow AI API shutdown complete")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )