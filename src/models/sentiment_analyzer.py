"""
Advanced Sentiment Analysis Engine for Financial Markets
Combines FinBERT, custom transformers, and emotion detection
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FinancialSentimentAnalyzer:
    """
    Multi-model sentiment analyzer specialized for financial text
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_models()
        self.vader = SentimentIntensityAnalyzer()
        
        # Meme stock vocabulary and patterns
        self.meme_patterns = {
            'bullish': ['🚀', '💎', '🙌', 'moon', 'lambo', 'tendies', 'stonks'],
            'bearish': ['📉', '💸', 'rekt', 'bag holder', 'paper hands'],
            'neutral': ['hodl', 'diamond hands', 'ape', 'retard']
        }
        
        # Financial keywords for context weighting
        self.financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
            'merger', 'acquisition', 'ipo', 'dividend', 'split', 'buyback'
        ]
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            # FinBERT for financial sentiment
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                'ProsusAI/finbert'
            )
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                'ProsusAI/finbert'
            ).to(self.device)
            
            logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase for processing (but preserve original case for some models)
        return text.strip()
    
    def analyze_finbert(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using FinBERT"""
        try:
            inputs = self.finbert_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # FinBERT labels: negative, neutral, positive
            scores = predictions.cpu().numpy()[0]
            
            return {
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2]),
                'compound': float(scores[2] - scores[0])  # Net sentiment
            }
            
        except Exception as e:
            logger.error(f"FinBERT analysis error: {e}")
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33, 'compound': 0.0}
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        scores = self.vader.polarity_scores(text)
        return scores
    
    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def detect_meme_sentiment(self, text: str) -> Dict[str, float]:
        """Detect meme stock sentiment patterns"""
        text_lower = text.lower()
        
        bullish_score = sum(1 for pattern in self.meme_patterns['bullish'] 
                           if pattern in text_lower)
        bearish_score = sum(1 for pattern in self.meme_patterns['bearish'] 
                           if pattern in text_lower)
        neutral_score = sum(1 for pattern in self.meme_patterns['neutral'] 
                           if pattern in text_lower)
        
        total = bullish_score + bearish_score + neutral_score
        
        if total == 0:
            return {'bullish': 0.0, 'bearish': 0.0, 'neutral': 0.0, 'meme_intensity': 0.0}
        
        return {
            'bullish': bullish_score / total,
            'bearish': bearish_score / total,
            'neutral': neutral_score / total,
            'meme_intensity': min(total / 5.0, 1.0)  # Normalize intensity
        }
    
    def calculate_financial_context_weight(self, text: str) -> float:
        """Calculate weight based on financial context"""
        text_lower = text.lower()
        financial_mentions = sum(1 for keyword in self.financial_keywords 
                               if keyword in text_lower)
        
        # Weight between 0.5 (low financial context) and 2.0 (high financial context)
        return min(0.5 + (financial_mentions * 0.3), 2.0)
    
    def ensemble_sentiment(self, text: str) -> Dict[str, float]:
        """
        Combine multiple sentiment analysis methods
        Returns comprehensive sentiment scores
        """
        processed_text = self.preprocess_text(text)
        
        # Get individual model scores
        finbert_scores = self.analyze_finbert(processed_text)
        vader_scores = self.analyze_vader(processed_text)
        textblob_scores = self.analyze_textblob(processed_text)
        meme_scores = self.detect_meme_sentiment(processed_text)
        
        # Calculate financial context weight
        context_weight = self.calculate_financial_context_weight(processed_text)
        
        # Weighted ensemble (FinBERT gets highest weight for financial text)
        finbert_weight = 0.5 * context_weight
        vader_weight = 0.3
        textblob_weight = 0.2
        
        # Normalize weights
        total_weight = finbert_weight + vader_weight + textblob_weight
        finbert_weight /= total_weight
        vader_weight /= total_weight
        textblob_weight /= total_weight
        
        # Calculate ensemble scores
        ensemble_compound = (
            finbert_scores['compound'] * finbert_weight +
            vader_scores['compound'] * vader_weight +
            textblob_scores['polarity'] * textblob_weight
        )
        
        # Adjust for meme intensity
        if meme_scores['meme_intensity'] > 0.3:
            meme_adjustment = (meme_scores['bullish'] - meme_scores['bearish']) * meme_scores['meme_intensity']
            ensemble_compound = ensemble_compound * 0.7 + meme_adjustment * 0.3
        
        return {
            'compound': float(ensemble_compound),
            'positive': float(max(0, ensemble_compound)),
            'negative': float(max(0, -ensemble_compound)),
            'neutral': float(1 - abs(ensemble_compound)),
            'confidence': float(abs(ensemble_compound)),
            'meme_intensity': meme_scores['meme_intensity'],
            'financial_context': context_weight,
            'individual_scores': {
                'finbert': finbert_scores,
                'vader': vader_scores,
                'textblob': textblob_scores,
                'meme': meme_scores
            }
        }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze multiple texts efficiently"""
        return [self.ensemble_sentiment(text) for text in texts]
    
    def get_emotion_classification(self, sentiment_scores: Dict[str, float]) -> str:
        """Classify emotion based on sentiment scores"""
        compound = sentiment_scores['compound']
        confidence = sentiment_scores['confidence']
        meme_intensity = sentiment_scores.get('meme_intensity', 0)
        
        if meme_intensity > 0.5:
            if compound > 0.3:
                return 'euphoria'
            elif compound < -0.3:
                return 'panic'
            else:
                return 'meme_neutral'
        
        if confidence < 0.2:
            return 'neutral'
        elif compound > 0.5:
            return 'bullish'
        elif compound > 0.1:
            return 'optimistic'
        elif compound < -0.5:
            return 'bearish'
        elif compound < -0.1:
            return 'pessimistic'
        else:
            return 'neutral'


# Example usage and testing
if __name__ == "__main__":
    analyzer = FinancialSentimentAnalyzer()
    
    # Test samples
    test_texts = [
        "TSLA to the moon! 🚀🚀🚀 Diamond hands baby!",
        "Apple earnings beat expectations, strong guidance for Q4",
        "Market crash incoming, everything is overvalued",
        "Just bought more GME, apes together strong 💎🙌",
        "Fed meeting tomorrow, expecting rate cuts"
    ]
    
    for text in test_texts:
        result = analyzer.ensemble_sentiment(text)
        emotion = analyzer.get_emotion_classification(result)
        
        print(f"\nText: {text}")
        print(f"Sentiment: {result['compound']:.3f}")
        print(f"Emotion: {emotion}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Meme Intensity: {result['meme_intensity']:.3f}")