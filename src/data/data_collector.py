"""
Multi-Source Data Collection Pipeline
Collects real-time data from Twitter, Reddit, Discord, and Financial APIs
"""

import asyncio
import aiohttp
import tweepy
import praw
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import time
from typing import Dict, List, Optional, Tuple
import os
from dataclasses import dataclass
import redis
from kafka import KafkaProducer
import requests

logger = logging.getLogger(__name__)


@dataclass
class SocialMediaPost:
    """Data structure for social media posts"""
    platform: str
    post_id: str
    text: str
    author: str
    timestamp: datetime
    likes: int
    shares: int
    comments: int
    symbols_mentioned: List[str]
    url: Optional[str] = None


@dataclass
class FinancialData:
    """Data structure for financial data"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    change_percent: Optional[float] = None


class TwitterCollector:
    """Collect tweets related to stock symbols"""
    
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_token_secret: str):
        self.auth = tweepy.OAuthHandler(api_key, api_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
        
        # Twitter API v2 client
        self.client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            wait_on_rate_limit=True
        )
    
    def collect_tweets(self, symbols: List[str], count: int = 100) -> List[SocialMediaPost]:
        """Collect recent tweets mentioning stock symbols"""
        posts = []
        
        for symbol in symbols:
            try:
                # Search for tweets mentioning the symbol
                query = f"${symbol} OR #{symbol} -is:retweet lang:en"
                tweets = tweepy.Cursor(
                    self.api.search_tweets,
                    q=query,
                    result_type="recent",
                    tweet_mode="extended"
                ).items(count)
                
                for tweet in tweets:
                    post = SocialMediaPost(
                        platform="twitter",
                        post_id=str(tweet.id),
                        text=tweet.full_text,
                        author=tweet.user.screen_name,
                        timestamp=tweet.created_at,
                        likes=tweet.favorite_count,
                        shares=tweet.retweet_count,
                        comments=tweet.reply_count if hasattr(tweet, 'reply_count') else 0,
                        symbols_mentioned=[symbol],
                        url=f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
                    )
                    posts.append(post)
                    
                logger.info(f"Collected {len(posts)} tweets for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting tweets for {symbol}: {e}")
        
        return posts
    
    def stream_tweets(self, symbols: List[str], callback):
        """Stream real-time tweets"""
        class TweetStreamListener(tweepy.StreamListener):
            def on_status(self, status):
                post = SocialMediaPost(
                    platform="twitter",
                    post_id=str(status.id),
                    text=status.full_text if hasattr(status, 'full_text') else status.text,
                    author=status.user.screen_name,
                    timestamp=status.created_at,
                    likes=status.favorite_count,
                    shares=status.retweet_count,
                    comments=0,
                    symbols_mentioned=self._extract_symbols(status.text),
                    url=f"https://twitter.com/{status.user.screen_name}/status/{status.id}"
                )
                callback(post)
                return True
            
            def on_error(self, status_code):
                logger.error(f"Twitter stream error: {status_code}")
                return True
        
        listener = TweetStreamListener()
        stream = tweepy.Stream(auth=self.api.auth, listener=listener)
        
        # Track symbols
        track_terms = [f"${symbol}" for symbol in symbols] + [f"#{symbol}" for symbol in symbols]
        stream.filter(track=track_terms, is_async=True)
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        import re
        symbols = re.findall(r'\$([A-Z]{1,5})', text.upper())
        return list(set(symbols))


class RedditCollector:
    """Collect posts from Reddit financial subreddits"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        self.target_subreddits = [
            'wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis',
            'ValueInvesting', 'StockMarket', 'pennystocks', 'options'
        ]
    
    def collect_posts(self, symbols: List[str], limit: int = 100) -> List[SocialMediaPost]:
        """Collect Reddit posts mentioning stock symbols"""
        posts = []
        
        for subreddit_name in self.target_subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot posts
                for submission in subreddit.hot(limit=limit):
                    mentioned_symbols = self._extract_symbols_from_text(
                        f"{submission.title} {submission.selftext}", symbols
                    )
                    
                    if mentioned_symbols:
                        post = SocialMediaPost(
                            platform="reddit",
                            post_id=submission.id,
                            text=f"{submission.title}\n{submission.selftext}",
                            author=str(submission.author) if submission.author else "deleted",
                            timestamp=datetime.fromtimestamp(submission.created_utc),
                            likes=submission.score,
                            shares=0,  # Reddit doesn't have shares
                            comments=submission.num_comments,
                            symbols_mentioned=mentioned_symbols,
                            url=f"https://reddit.com{submission.permalink}"
                        )
                        posts.append(post)
                
                logger.info(f"Collected {len(posts)} posts from r/{subreddit_name}")
                
            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit_name}: {e}")
        
        return posts
    
    def collect_comments(self, post_ids: List[str]) -> List[SocialMediaPost]:
        """Collect comments from specific Reddit posts"""
        comments = []
        
        for post_id in post_ids:
            try:
                submission = self.reddit.submission(id=post_id)
                submission.comments.replace_more(limit=0)
                
                for comment in submission.comments.list():
                    if hasattr(comment, 'body') and comment.body != '[deleted]':
                        comment_post = SocialMediaPost(
                            platform="reddit_comment",
                            post_id=comment.id,
                            text=comment.body,
                            author=str(comment.author) if comment.author else "deleted",
                            timestamp=datetime.fromtimestamp(comment.created_utc),
                            likes=comment.score,
                            shares=0,
                            comments=0,
                            symbols_mentioned=self._extract_symbols_from_text(comment.body, []),
                            url=f"https://reddit.com{comment.permalink}"
                        )
                        comments.append(comment_post)
                        
            except Exception as e:
                logger.error(f"Error collecting comments for {post_id}: {e}")
        
        return comments
    
    def _extract_symbols_from_text(self, text: str, target_symbols: List[str]) -> List[str]:
        """Extract stock symbols from text"""
        import re
        text_upper = text.upper()
        
        # Find all potential symbols
        found_symbols = re.findall(r'\$([A-Z]{1,5})', text_upper)
        
        # Also check for symbols without $ prefix
        for symbol in target_symbols:
            if symbol.upper() in text_upper:
                found_symbols.append(symbol.upper())
        
        return list(set(found_symbols))


class FinancialDataCollector:
    """Collect financial data from various APIs"""
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.cache = redis.Redis(host='localhost', port=6379, db=0) if redis else None
    
    def get_stock_data(self, symbols: List[str]) -> List[FinancialData]:
        """Get current stock data using yfinance"""
        financial_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    
                    data = FinancialData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        price=float(latest['Close']),
                        volume=int(latest['Volume']),
                        market_cap=info.get('marketCap'),
                        pe_ratio=info.get('trailingPE'),
                        change_percent=self._calculate_change_percent(hist)
                    )
                    financial_data.append(data)
                    
                logger.info(f"Collected financial data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting financial data for {symbol}: {e}")
        
        return financial_data
    
    def get_options_data(self, symbol: str) -> Dict:
        """Get options data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            
            if options_dates:
                # Get nearest expiration
                nearest_date = options_dates[0]
                calls = ticker.option_chain(nearest_date).calls
                puts = ticker.option_chain(nearest_date).puts
                
                return {
                    'symbol': symbol,
                    'expiration': nearest_date,
                    'calls': calls.to_dict('records'),
                    'puts': puts.to_dict('records'),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"Error collecting options data for {symbol}: {e}")
        
        return {}
    
    def _calculate_change_percent(self, hist_data: pd.DataFrame) -> float:
        """Calculate percentage change from previous close"""
        if len(hist_data) >= 2:
            current = hist_data.iloc[-1]['Close']
            previous = hist_data.iloc[-2]['Close']
            return ((current - previous) / previous) * 100
        return 0.0


class DataPipeline:
    """Main data collection pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.twitter_collector = TwitterCollector(
            config['twitter']['api_key'],
            config['twitter']['api_secret'],
            config['twitter']['access_token'],
            config['twitter']['access_token_secret']
        )
        self.reddit_collector = RedditCollector(
            config['reddit']['client_id'],
            config['reddit']['client_secret'],
            config['reddit']['user_agent']
        )
        self.financial_collector = FinancialDataCollector(
            config.get('alpha_vantage', {}).get('api_key')
        )
        
        # Kafka producer for real-time streaming
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=config.get('kafka', {}).get('servers', ['localhost:9092']),
            value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8')
        )
    
    def collect_all_data(self, symbols: List[str]) -> Dict[str, List]:
        """Collect data from all sources"""
        logger.info(f"Starting data collection for symbols: {symbols}")
        
        # Collect social media data
        twitter_posts = self.twitter_collector.collect_tweets(symbols)
        reddit_posts = self.reddit_collector.collect_posts(symbols)
        
        # Collect financial data
        financial_data = self.financial_collector.get_stock_data(symbols)
        
        # Combine all data
        all_data = {
            'twitter_posts': [self._post_to_dict(post) for post in twitter_posts],
            'reddit_posts': [self._post_to_dict(post) for post in reddit_posts],
            'financial_data': [self._financial_to_dict(data) for data in financial_data],
            'collection_timestamp': datetime.now().isoformat()
        }
        
        # Send to Kafka for real-time processing
        self._send_to_kafka(all_data)
        
        logger.info(f"Data collection completed. Total posts: {len(twitter_posts + reddit_posts)}")
        
        return all_data
    
    def _post_to_dict(self, post: SocialMediaPost) -> Dict:
        """Convert SocialMediaPost to dictionary"""
        return {
            'platform': post.platform,
            'post_id': post.post_id,
            'text': post.text,
            'author': post.author,
            'timestamp': post.timestamp.isoformat(),
            'likes': post.likes,
            'shares': post.shares,
            'comments': post.comments,
            'symbols_mentioned': post.symbols_mentioned,
            'url': post.url
        }
    
    def _financial_to_dict(self, data: FinancialData) -> Dict:
        """Convert FinancialData to dictionary"""
        return {
            'symbol': data.symbol,
            'timestamp': data.timestamp.isoformat(),
            'price': data.price,
            'volume': data.volume,
            'market_cap': data.market_cap,
            'pe_ratio': data.pe_ratio,
            'change_percent': data.change_percent
        }
    
    def _send_to_kafka(self, data: Dict):
        """Send data to Kafka topic"""
        try:
            self.kafka_producer.send('market_sentiment_data', value=data)
            self.kafka_producer.flush()
            logger.info("Data sent to Kafka successfully")
        except Exception as e:
            logger.error(f"Error sending data to Kafka: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'twitter': {
            'api_key': os.getenv('TWITTER_API_KEY'),
            'api_secret': os.getenv('TWITTER_API_SECRET'),
            'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
            'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        },
        'reddit': {
            'client_id': os.getenv('REDDIT_CLIENT_ID'),
            'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'user_agent': 'SentimentFlow:v1.0 (by /u/your_username)'
        },
        'alpha_vantage': {
            'api_key': os.getenv('ALPHA_VANTAGE_API_KEY')
        },
        'kafka': {
            'servers': ['localhost:9092']
        }
    }
    
    # Initialize pipeline
    pipeline = DataPipeline(config)
    
    # Collect data for popular stocks
    symbols = ['AAPL', 'TSLA', 'GME', 'AMC', 'NVDA', 'MSFT', 'GOOGL']
    data = pipeline.collect_all_data(symbols)
    
    print(f"Collected data for {len(symbols)} symbols")
    print(f"Twitter posts: {len(data['twitter_posts'])}")
    print(f"Reddit posts: {len(data['reddit_posts'])}")
    print(f"Financial data points: {len(data['financial_data'])}")