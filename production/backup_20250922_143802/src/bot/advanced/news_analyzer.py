"""
News Sentiment Analysis System

This module provides comprehensive news sentiment analysis capabilities including
sentiment scoring, market event detection, and trading halt triggers during
high-impact news for the Bybit trading bot.

Key Features:
- Multi-source news aggregation
- Real-time sentiment analysis using NLP
- Market event classification and impact scoring
- News blackout periods and trading halt triggers
- Mode-specific sentiment rules
- Economic calendar integration
- Social media sentiment monitoring
- Custom news filters and alerts

Author: Trading Bot Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
from dataclasses import dataclass
import requests
import time
import re
from urllib.parse import urlencode
import feedparser
import json
import asyncio
import aiohttp
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("Could not download NLTK data")


class SentimentLevel(Enum):
    """Sentiment levels"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class NewsCategory(Enum):
    """News categories"""
    CRYPTO = "crypto"
    REGULATORY = "regulatory"
    MARKET = "market"
    ECONOMIC = "economic"
    TECHNICAL = "technical"
    SECURITY = "security"
    ADOPTION = "adoption"
    INSTITUTIONAL = "institutional"


class EventImpact(Enum):
    """Event impact levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float = 0.0
    sentiment_level: SentimentLevel = SentimentLevel.NEUTRAL
    category: NewsCategory = NewsCategory.CRYPTO
    impact_score: float = 0.0
    keywords: List[str] = None
    entities: List[str] = None


@dataclass
class MarketEvent:
    """Market event data structure"""
    event_type: str
    title: str
    description: str
    scheduled_time: datetime
    impact_level: EventImpact
    currency: str
    actual_value: Optional[float] = None
    forecast_value: Optional[float] = None
    previous_value: Optional[float] = None
    market_reaction: Optional[float] = None


@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    overall_sentiment: SentimentLevel
    sentiment_score: float
    confidence: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    article_count: int
    impact_score: float
    recommendation: str
    blackout_recommended: bool = False


class NewsAnalyzer:
    """
    Comprehensive News Sentiment Analysis System
    
    This class provides sophisticated news sentiment analysis with
    market event detection and trading recommendations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the news analyzer
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or self._get_default_config()
        self.news_sources = self._initialize_news_sources()
        self.sentiment_analyzer = self._initialize_sentiment_analyzer()
        self.economic_calendar = []
        self.news_cache = {}
        self.sentiment_history = []
        self.blackout_periods = []
        
        # Initialize transformers model for advanced sentiment analysis
        try:
            self.finbert_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
        except Exception as e:
            logger.warning(f"Could not load FinBERT model: {e}")
            self.finbert_analyzer = None
        
        logger.info("NewsAnalyzer initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for news analysis"""
        return {
            'news_sources': {
                'coindesk': {'enabled': True, 'weight': 0.8},
                'cointelegraph': {'enabled': True, 'weight': 0.7},
                'bloomberg': {'enabled': True, 'weight': 0.9},
                'reuters': {'enabled': True, 'weight': 0.9},
                'cryptonews': {'enabled': True, 'weight': 0.6},
                'reddit': {'enabled': False, 'weight': 0.4},
                'twitter': {'enabled': False, 'weight': 0.3}
            },
            'analysis_settings': {
                'sentiment_threshold_high': 0.6,
                'sentiment_threshold_low': -0.6,
                'impact_threshold_critical': 0.8,
                'impact_threshold_high': 0.6,
                'blackout_duration_minutes': 30,
                'news_relevance_hours': 24,
                'min_articles_for_analysis': 3
            },
            'keywords': {
                'positive': ['bullish', 'adoption', 'partnership', 'investment', 'growth', 'breakthrough'],
                'negative': ['bearish', 'crash', 'regulation', 'ban', 'hack', 'scandal', 'decline'],
                'crypto_specific': ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi', 'nft'],
                'high_impact': ['sec', 'fed', 'regulation', 'ban', 'hack', 'exchange']
            },
            'trading_modes': {
                'conservative': {
                    'sentiment_threshold': 0.4,
                    'blackout_on_high_impact': True,
                    'require_consensus': True
                },
                'aggressive': {
                    'sentiment_threshold': 0.6,
                    'blackout_on_high_impact': False,
                    'require_consensus': False
                }
            },
            'api_keys': {
                'newsapi': '',
                'alpha_vantage': '',
                'twitter_bearer_token': '',
                'reddit_client_id': '',
                'reddit_client_secret': ''
            }
        }
    
    def _initialize_news_sources(self) -> Dict[str, Dict]:
        """Initialize news source configurations"""
        return {
            'coindesk': {
                'rss_url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'weight': self.config['news_sources']['coindesk']['weight'],
                'enabled': self.config['news_sources']['coindesk']['enabled']
            },
            'cointelegraph': {
                'rss_url': 'https://cointelegraph.com/rss',
                'weight': self.config['news_sources']['cointelegraph']['weight'],
                'enabled': self.config['news_sources']['cointelegraph']['enabled']
            },
            'cryptonews': {
                'rss_url': 'https://cryptonews.com/news/feed/',
                'weight': self.config['news_sources']['cryptonews']['weight'],
                'enabled': self.config['news_sources']['cryptonews']['enabled']
            }
        }
    
    def _initialize_sentiment_analyzer(self) -> SentimentIntensityAnalyzer:
        """Initialize NLTK sentiment analyzer"""
        try:
            return SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Could not initialize sentiment analyzer: {e}")
            return None
    
    async def fetch_news_articles(self, hours_back: int = 24) -> List[NewsArticle]:
        """
        Fetch news articles from configured sources
        
        Args:
            hours_back: Number of hours to look back for news
            
        Returns:
            List of NewsArticle objects
        """
        try:
            articles = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for source_name, source_config in self.news_sources.items():
                if not source_config['enabled']:
                    continue
                
                try:
                    if source_name in ['coindesk', 'cointelegraph', 'cryptonews']:
                        source_articles = await self._fetch_rss_articles(source_name, source_config, cutoff_time)
                        articles.extend(source_articles)
                    
                    # Add other source types here (API-based, etc.)
                    
                except Exception as e:
                    logger.error(f"Error fetching from {source_name}: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from {len(self.news_sources)} sources")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news articles: {e}")
            return []
    
    async def _fetch_rss_articles(self, source_name: str, source_config: Dict, cutoff_time: datetime) -> List[NewsArticle]:
        """Fetch articles from RSS feed"""
        try:
            articles = []
            
            async with aiohttp.ClientSession() as session:
                async with session.get(source_config['rss_url'], timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries:
                            try:
                                # Parse published date
                                published_time = datetime.now()
                                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                    published_time = datetime(*entry.published_parsed[:6])
                                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                                    published_time = datetime(*entry.updated_parsed[:6])
                                
                                # Skip old articles
                                if published_time < cutoff_time:
                                    continue
                                
                                # Extract content
                                content = entry.get('summary', entry.get('description', ''))
                                if hasattr(entry, 'content') and entry.content:
                                    content = entry.content[0].value if isinstance(entry.content, list) else entry.content
                                
                                # Clean content
                                content = self._clean_text(content)
                                
                                article = NewsArticle(
                                    title=entry.get('title', ''),
                                    content=content,
                                    source=source_name,
                                    url=entry.get('link', ''),
                                    published_at=published_time,
                                    category=self._categorize_article(entry.get('title', '') + ' ' + content)
                                )
                                
                                articles.append(article)
                                
                            except Exception as e:
                                logger.warning(f"Error parsing article from {source_name}: {e}")
                                continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS articles from {source_name}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s\.,!?;:-]', '', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text
    
    def _categorize_article(self, text: str) -> NewsCategory:
        """Categorize article based on content"""
        try:
            text_lower = text.lower()
            
            # Check for keywords
            if any(word in text_lower for word in ['regulation', 'sec', 'cftc', 'ban', 'legal']):
                return NewsCategory.REGULATORY
            elif any(word in text_lower for word in ['hack', 'security', 'breach', 'stolen']):
                return NewsCategory.SECURITY
            elif any(word in text_lower for word in ['adoption', 'partnership', 'institution']):
                return NewsCategory.ADOPTION
            elif any(word in text_lower for word in ['market', 'price', 'trading', 'volume']):
                return NewsCategory.MARKET
            elif any(word in text_lower for word in ['technical', 'upgrade', 'fork', 'protocol']):
                return NewsCategory.TECHNICAL
            elif any(word in text_lower for word in ['economic', 'inflation', 'fed', 'gdp']):
                return NewsCategory.ECONOMIC
            else:
                return NewsCategory.CRYPTO
                
        except Exception as e:
            logger.error(f"Error categorizing article: {e}")
            return NewsCategory.CRYPTO
    
    def analyze_sentiment(self, articles: List[NewsArticle]) -> SentimentAnalysis:
        """
        Analyze sentiment of news articles
        
        Args:
            articles: List of news articles
            
        Returns:
            SentimentAnalysis object
        """
        try:
            if not articles:
                return SentimentAnalysis(
                    overall_sentiment=SentimentLevel.NEUTRAL,
                    sentiment_score=0.0,
                    confidence=0.0,
                    positive_ratio=0.0,
                    negative_ratio=0.0,
                    neutral_ratio=1.0,
                    article_count=0,
                    impact_score=0.0,
                    recommendation="No news data available"
                )
            
            # Analyze each article
            sentiment_scores = []
            impact_scores = []
            
            for article in articles:
                # Get sentiment score
                sentiment_score = self._calculate_article_sentiment(article)
                article.sentiment_score = sentiment_score
                article.sentiment_level = self._score_to_sentiment_level(sentiment_score)
                
                # Calculate impact score
                impact_score = self._calculate_impact_score(article)
                article.impact_score = impact_score
                
                # Weight by source reliability
                source_weight = self.news_sources.get(article.source, {}).get('weight', 0.5)
                weighted_sentiment = sentiment_score * source_weight
                weighted_impact = impact_score * source_weight
                
                sentiment_scores.append(weighted_sentiment)
                impact_scores.append(weighted_impact)
            
            # Calculate overall metrics
            overall_sentiment_score = np.mean(sentiment_scores) if sentiment_scores else 0.0
            overall_impact_score = np.mean(impact_scores) if impact_scores else 0.0
            
            # Calculate sentiment distribution
            positive_count = sum(1 for score in sentiment_scores if score > 0.1)
            negative_count = sum(1 for score in sentiment_scores if score < -0.1)
            neutral_count = len(sentiment_scores) - positive_count - negative_count
            
            total_articles = len(articles)
            positive_ratio = positive_count / total_articles if total_articles > 0 else 0.0
            negative_ratio = negative_count / total_articles if total_articles > 0 else 0.0
            neutral_ratio = neutral_count / total_articles if total_articles > 0 else 1.0
            
            # Determine overall sentiment level
            overall_sentiment_level = self._score_to_sentiment_level(overall_sentiment_score)
            
            # Calculate confidence based on article count and consensus
            confidence = self._calculate_confidence(sentiment_scores, total_articles)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                overall_sentiment_score, 
                overall_impact_score, 
                confidence,
                total_articles
            )
            
            # Determine if blackout is recommended
            blackout_recommended = self._should_recommend_blackout(articles, overall_impact_score)
            
            analysis = SentimentAnalysis(
                overall_sentiment=overall_sentiment_level,
                sentiment_score=float(overall_sentiment_score),
                confidence=float(confidence),
                positive_ratio=float(positive_ratio),
                negative_ratio=float(negative_ratio),
                neutral_ratio=float(neutral_ratio),
                article_count=total_articles,
                impact_score=float(overall_impact_score),
                recommendation=recommendation,
                blackout_recommended=blackout_recommended
            )
            
            # Store in history
            self.sentiment_history.append({
                'timestamp': datetime.now(),
                'analysis': analysis,
                'articles': articles
            })
            
            # Keep history manageable
            if len(self.sentiment_history) > 1000:
                self.sentiment_history = self.sentiment_history[-500:]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return SentimentAnalysis(
                overall_sentiment=SentimentLevel.NEUTRAL,
                sentiment_score=0.0,
                confidence=0.0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=1.0,
                article_count=0,
                impact_score=0.0,
                recommendation="Error in sentiment analysis"
            )
    
    def _calculate_article_sentiment(self, article: NewsArticle) -> float:
        """Calculate sentiment score for a single article"""
        try:
            text = f"{article.title} {article.content}"
            
            # Use multiple sentiment analysis methods
            scores = []
            
            # NLTK VADER
            if self.sentiment_analyzer:
                vader_score = self.sentiment_analyzer.polarity_scores(text)
                scores.append(vader_score['compound'])
            
            # TextBlob
            try:
                blob = TextBlob(text)
                scores.append(blob.sentiment.polarity)
            except:
                pass
            
            # FinBERT (if available)
            if self.finbert_analyzer:
                try:
                    # Truncate text for transformer model
                    truncated_text = text[:512]
                    result = self.finbert_analyzer(truncated_text)[0]
                    
                    # Convert to sentiment score
                    if result['label'] == 'positive':
                        finbert_score = result['score']
                    elif result['label'] == 'negative':
                        finbert_score = -result['score']
                    else:
                        finbert_score = 0.0
                    
                    scores.append(finbert_score)
                except:
                    pass
            
            # Calculate weighted average
            if scores:
                sentiment_score = np.mean(scores)
            else:
                sentiment_score = 0.0
            
            # Apply keyword-based adjustments
            sentiment_score = self._adjust_sentiment_with_keywords(text, sentiment_score)
            
            return float(np.clip(sentiment_score, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating article sentiment: {e}")
            return 0.0
    
    def _adjust_sentiment_with_keywords(self, text: str, base_sentiment: float) -> float:
        """Adjust sentiment based on keyword analysis"""
        try:
            text_lower = text.lower()
            adjustment = 0.0
            
            # Positive keywords
            positive_keywords = self.config['keywords']['positive']
            positive_matches = sum(1 for keyword in positive_keywords if keyword in text_lower)
            adjustment += positive_matches * 0.1
            
            # Negative keywords
            negative_keywords = self.config['keywords']['negative']
            negative_matches = sum(1 for keyword in negative_keywords if keyword in text_lower)
            adjustment -= negative_matches * 0.1
            
            # High impact keywords (amplify existing sentiment)
            high_impact_keywords = self.config['keywords']['high_impact']
            high_impact_matches = sum(1 for keyword in high_impact_keywords if keyword in text_lower)
            if high_impact_matches > 0:
                adjustment += base_sentiment * 0.2 * high_impact_matches
            
            return base_sentiment + adjustment
            
        except Exception as e:
            logger.error(f"Error adjusting sentiment with keywords: {e}")
            return base_sentiment
    
    def _calculate_impact_score(self, article: NewsArticle) -> float:
        """Calculate market impact score for an article"""
        try:
            impact_score = 0.0
            text = f"{article.title} {article.content}".lower()
            
            # Source credibility weight
            source_weight = self.news_sources.get(article.source, {}).get('weight', 0.5)
            impact_score += source_weight * 0.3
            
            # Category-based impact
            category_impacts = {
                NewsCategory.REGULATORY: 0.8,
                NewsCategory.SECURITY: 0.7,
                NewsCategory.INSTITUTIONAL: 0.6,
                NewsCategory.MARKET: 0.5,
                NewsCategory.ECONOMIC: 0.6,
                NewsCategory.ADOPTION: 0.4,
                NewsCategory.TECHNICAL: 0.3,
                NewsCategory.CRYPTO: 0.4
            }
            impact_score += category_impacts.get(article.category, 0.3) * 0.4
            
            # High-impact keyword presence
            high_impact_keywords = self.config['keywords']['high_impact']
            high_impact_matches = sum(1 for keyword in high_impact_keywords if keyword in text)
            impact_score += min(high_impact_matches * 0.1, 0.3)
            
            # Recency boost (newer articles have higher impact)
            hours_old = (datetime.now() - article.published_at).total_seconds() / 3600
            recency_multiplier = max(0.5, 1.0 - (hours_old / 24))
            impact_score *= recency_multiplier
            
            return float(np.clip(impact_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating impact score: {e}")
            return 0.0
    
    def _score_to_sentiment_level(self, score: float) -> SentimentLevel:
        """Convert numeric sentiment score to sentiment level"""
        if score >= 0.6:
            return SentimentLevel.VERY_POSITIVE
        elif score >= 0.2:
            return SentimentLevel.POSITIVE
        elif score <= -0.6:
            return SentimentLevel.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentLevel.NEGATIVE
        else:
            return SentimentLevel.NEUTRAL
    
    def _calculate_confidence(self, sentiment_scores: List[float], article_count: int) -> float:
        """Calculate confidence in sentiment analysis"""
        try:
            if not sentiment_scores or article_count == 0:
                return 0.0
            
            # Base confidence from article count
            count_confidence = min(1.0, article_count / 10)  # Max confidence at 10+ articles
            
            # Consensus confidence (lower variance = higher confidence)
            if len(sentiment_scores) > 1:
                variance = np.var(sentiment_scores)
                consensus_confidence = 1.0 - min(1.0, variance)
            else:
                consensus_confidence = 0.5
            
            # Combined confidence
            overall_confidence = (count_confidence * 0.6 + consensus_confidence * 0.4)
            
            return float(np.clip(overall_confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _generate_recommendation(self, sentiment_score: float, impact_score: float, confidence: float, article_count: int) -> str:
        """Generate trading recommendation based on analysis"""
        try:
            recommendations = []
            
            # Sentiment-based recommendations
            if sentiment_score >= 0.6 and confidence >= 0.6:
                recommendations.append("Strong positive sentiment detected - consider increasing long positions")
            elif sentiment_score >= 0.3 and confidence >= 0.5:
                recommendations.append("Moderate positive sentiment - maintain or slightly increase exposure")
            elif sentiment_score <= -0.6 and confidence >= 0.6:
                recommendations.append("Strong negative sentiment detected - consider reducing positions or hedging")
            elif sentiment_score <= -0.3 and confidence >= 0.5:
                recommendations.append("Moderate negative sentiment - exercise caution")
            else:
                recommendations.append("Mixed or neutral sentiment - maintain current strategy")
            
            # Impact-based recommendations
            if impact_score >= 0.8:
                recommendations.append("High market impact expected - consider temporary trading halt")
            elif impact_score >= 0.6:
                recommendations.append("Significant market impact possible - reduce position sizes")
            
            # Confidence-based recommendations
            if confidence < 0.3:
                recommendations.append("Low confidence in analysis - wait for more data")
            elif article_count < 3:
                recommendations.append("Limited news data - exercise additional caution")
            
            return "; ".join(recommendations)
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "Unable to generate recommendation due to analysis error"
    
    def _should_recommend_blackout(self, articles: List[NewsArticle], impact_score: float) -> bool:
        """Determine if trading blackout should be recommended"""
        try:
            # High impact score threshold
            if impact_score >= self.config['analysis_settings']['impact_threshold_critical']:
                return True
            
            # Check for high-impact categories
            critical_categories = [NewsCategory.REGULATORY, NewsCategory.SECURITY]
            critical_articles = [a for a in articles if a.category in critical_categories and a.impact_score > 0.7]
            
            if len(critical_articles) >= 2:
                return True
            
            # Check for high-impact keywords in recent articles
            recent_articles = [a for a in articles if (datetime.now() - a.published_at).hours <= 2]
            high_impact_keywords = self.config['keywords']['high_impact']
            
            for article in recent_articles:
                text = f"{article.title} {article.content}".lower()
                high_impact_matches = sum(1 for keyword in high_impact_keywords if keyword in text)
                if high_impact_matches >= 3:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining blackout recommendation: {e}")
            return False
    
    def should_halt_trading(self, trading_mode: str = 'conservative') -> Tuple[bool, str]:
        """
        Determine if trading should be halted based on current sentiment
        
        Args:
            trading_mode: Trading mode ('conservative' or 'aggressive')
            
        Returns:
            Tuple of (should_halt, reason)
        """
        try:
            # Check active blackout periods
            current_time = datetime.now()
            active_blackouts = [bp for bp in self.blackout_periods if bp['end_time'] > current_time]
            
            if active_blackouts:
                return True, f"Active blackout period: {active_blackouts[0]['reason']}"
            
            # Get recent sentiment analysis
            if not self.sentiment_history:
                return False, "No recent sentiment data available"
            
            latest_analysis = self.sentiment_history[-1]['analysis']
            
            # Mode-specific thresholds
            mode_config = self.config['trading_modes'].get(trading_mode, self.config['trading_modes']['conservative'])
            
            # Check for blackout recommendation
            if latest_analysis.blackout_recommended and mode_config['blackout_on_high_impact']:
                return True, f"High-impact news detected: {latest_analysis.recommendation}"
            
            # Check sentiment thresholds
            if abs(latest_analysis.sentiment_score) > mode_config['sentiment_threshold']:
                if mode_config['require_consensus'] and latest_analysis.confidence < 0.6:
                    return False, "Extreme sentiment detected but low confidence"
                
                sentiment_direction = "negative" if latest_analysis.sentiment_score < 0 else "positive"
                return True, f"Extreme {sentiment_direction} sentiment detected (score: {latest_analysis.sentiment_score:.2f})"
            
            return False, "No trading halt conditions met"
            
        except Exception as e:
            logger.error(f"Error determining trading halt: {e}")
            return False, "Error in halt determination"
    
    def add_blackout_period(self, reason: str, duration_minutes: int = None) -> None:
        """Add a manual blackout period"""
        try:
            duration = duration_minutes or self.config['analysis_settings']['blackout_duration_minutes']
            
            blackout = {
                'start_time': datetime.now(),
                'end_time': datetime.now() + timedelta(minutes=duration),
                'reason': reason,
                'manual': True
            }
            
            self.blackout_periods.append(blackout)
            logger.info(f"Added blackout period: {reason} (duration: {duration} minutes)")
            
        except Exception as e:
            logger.error(f"Error adding blackout period: {e}")
    
    def get_sentiment_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get sentiment analysis summary for specified period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_analyses = [
                h for h in self.sentiment_history 
                if h['timestamp'] >= cutoff_time
            ]
            
            if not recent_analyses:
                return {'message': 'No recent sentiment data available'}
            
            # Calculate summary statistics
            sentiment_scores = [a['analysis'].sentiment_score for a in recent_analyses]
            impact_scores = [a['analysis'].impact_score for a in recent_analyses]
            
            summary = {
                'period_hours': hours_back,
                'analysis_count': len(recent_analyses),
                'average_sentiment': float(np.mean(sentiment_scores)),
                'sentiment_volatility': float(np.std(sentiment_scores)),
                'average_impact': float(np.mean(impact_scores)),
                'max_impact': float(np.max(impact_scores)),
                'latest_sentiment': recent_analyses[-1]['analysis'].overall_sentiment.value,
                'latest_score': float(recent_analyses[-1]['analysis'].sentiment_score),
                'latest_confidence': float(recent_analyses[-1]['analysis'].confidence),
                'blackout_recommended': recent_analyses[-1]['analysis'].blackout_recommended,
                'active_blackouts': len([bp for bp in self.blackout_periods if bp['end_time'] > datetime.now()])
            }
            
            # Add trend analysis
            if len(sentiment_scores) >= 3:
                recent_trend = np.polyfit(range(len(sentiment_scores)), sentiment_scores, 1)[0]
                summary['sentiment_trend'] = 'improving' if recent_trend > 0.01 else 'declining' if recent_trend < -0.01 else 'stable'
            else:
                summary['sentiment_trend'] = 'insufficient_data'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {'error': str(e)}
    
    async def run_continuous_analysis(self, update_interval_minutes: int = 15) -> None:
        """Run continuous sentiment analysis in background"""
        try:
            logger.info(f"Starting continuous sentiment analysis (update interval: {update_interval_minutes} minutes)")
            
            while True:
                try:
                    # Fetch latest news
                    articles = await self.fetch_news_articles(hours_back=6)
                    
                    if articles:
                        # Analyze sentiment
                        analysis = self.analyze_sentiment(articles)
                        
                        logger.info(f"Sentiment analysis completed: {analysis.overall_sentiment.value} "
                                  f"(score: {analysis.sentiment_score:.2f}, confidence: {analysis.confidence:.2f})")
                        
                        # Check for automatic blackout triggers
                        if analysis.blackout_recommended:
                            self.add_blackout_period(
                                f"Automatic blackout: {analysis.recommendation}",
                                self.config['analysis_settings']['blackout_duration_minutes']
                            )
                    
                    # Clean up old data
                    self._cleanup_old_data()
                    
                except Exception as e:
                    logger.error(f"Error in continuous analysis cycle: {e}")
                
                # Wait for next update
                await asyncio.sleep(update_interval_minutes * 60)
                
        except Exception as e:
            logger.error(f"Error in continuous analysis: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory issues"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            
            # Clean sentiment history
            self.sentiment_history = [
                h for h in self.sentiment_history 
                if h['timestamp'] >= cutoff_time
            ]
            
            # Clean blackout periods
            self.blackout_periods = [
                bp for bp in self.blackout_periods 
                if bp['end_time'] >= datetime.now() - timedelta(hours=24)
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test the news analyzer
    import asyncio
    
    async def test_news_analyzer():
        print("Testing News Sentiment Analysis System")
        print("=" * 50)
        
        # Initialize analyzer
        analyzer = NewsAnalyzer()
        
        # Test with sample articles
        sample_articles = [
            NewsArticle(
                title="Bitcoin Soars to New All-Time High as Institutional Adoption Accelerates",
                content="Major financial institutions continue to embrace Bitcoin, driving unprecedented demand and price growth. The recent announcement of several Fortune 500 companies adding Bitcoin to their treasury reserves has sparked a new wave of institutional FOMO.",
                source="coindesk",
                url="https://example.com/1",
                published_at=datetime.now() - timedelta(hours=2),
                category=NewsCategory.ADOPTION
            ),
            NewsArticle(
                title="SEC Announces Stricter Cryptocurrency Regulations",
                content="The Securities and Exchange Commission unveiled new regulatory framework for cryptocurrency exchanges, requiring enhanced compliance measures and reporting standards. The announcement has caused concern among crypto traders about potential market impact.",
                source="bloomberg",
                url="https://example.com/2",
                published_at=datetime.now() - timedelta(hours=1),
                category=NewsCategory.REGULATORY
            ),
            NewsArticle(
                title="Major Exchange Suffers Security Breach, $50M in Crypto Stolen",
                content="A leading cryptocurrency exchange reported a significant security breach resulting in the theft of various cryptocurrencies worth approximately $50 million. The exchange has suspended all trading activities while investigating the incident.",
                source="reuters",
                url="https://example.com/3",
                published_at=datetime.now() - timedelta(minutes=30),
                category=NewsCategory.SECURITY
            )
        ]
        
        print(f"\nAnalyzing {len(sample_articles)} sample articles...")
        
        # Analyze sentiment
        analysis = analyzer.analyze_sentiment(sample_articles)
        
        print(f"\nSentiment Analysis Results:")
        print(f"Overall Sentiment: {analysis.overall_sentiment.value}")
        print(f"Sentiment Score: {analysis.sentiment_score:.3f}")
        print(f"Confidence: {analysis.confidence:.3f}")
        print(f"Impact Score: {analysis.impact_score:.3f}")
        print(f"Article Count: {analysis.article_count}")
        
        print(f"\nSentiment Distribution:")
        print(f"Positive: {analysis.positive_ratio:.1%}")
        print(f"Negative: {analysis.negative_ratio:.1%}")
        print(f"Neutral: {analysis.neutral_ratio:.1%}")
        
        print(f"\nRecommendation: {analysis.recommendation}")
        print(f"Blackout Recommended: {analysis.blackout_recommended}")
        
        # Test trading halt decision
        print(f"\nTrading Halt Analysis:")
        for mode in ['conservative', 'aggressive']:
            should_halt, reason = analyzer.should_halt_trading(mode)
            print(f"{mode.title()} Mode: {'HALT' if should_halt else 'CONTINUE'} - {reason}")
        
        # Test individual article analysis
        print(f"\nIndividual Article Analysis:")
        for i, article in enumerate(sample_articles, 1):
            print(f"\nArticle {i}:")
            print(f"  Title: {article.title}")
            print(f"  Category: {article.category.value}")
            print(f"  Sentiment: {article.sentiment_level.value} ({article.sentiment_score:.3f})")
            print(f"  Impact Score: {article.impact_score:.3f}")
        
        # Test sentiment summary
        print(f"\nSentiment Summary:")
        summary = analyzer.get_sentiment_summary(hours_back=24)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print(f"\nNews Sentiment Analysis Testing Complete!")
    
    # Run the test
    asyncio.run(test_news_analyzer())