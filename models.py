#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Models for Naver Cafe Search System
========================================
Defines data structures used throughout the application
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Set
from enum import Enum


class SearchIntent(Enum):
    """Types of search intent"""
    GENERAL = "general"
    COMMUNITY = "community"
    INFORMATION = "information"
    STUDY = "study"
    LOCAL = "local"
    BUSINESS = "business"


class ConfidenceLevel(Enum):
    """Confidence levels for search results"""
    VERY_HIGH = "매우 높음"
    HIGH = "높음"
    MEDIUM = "중간"
    LOW = "낮음"


@dataclass
class SearchQuery:
    """Represents a search query with metadata"""
    text: str
    intent: SearchIntent = SearchIntent.GENERAL
    exclude_keywords: List[str] = field(default_factory=list)
    generated_queries: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CafeInfo:
    """Basic cafe information from search results"""
    name: str
    url: str
    cafe_id: str
    description: str = ""
    member_count: Optional[int] = None
    
    def __hash__(self):
        return hash(self.url)


@dataclass
class SearchResult:
    """Individual search result from Naver"""
    title: str
    link: str
    description: str
    cafe_name: str
    cafe_url: str
    query_origin: str
    
    def to_cafe_info(self) -> CafeInfo:
        """Convert search result to cafe info"""
        # Extract cafe ID from URL
        cafe_id = self.cafe_url.split('/')[-1] if self.cafe_url else ""
        
        return CafeInfo(
            name=self.cafe_name,
            url=self.cafe_url,
            cafe_id=cafe_id,
            description=self.description
        )


@dataclass
class FilteredCafe:
    """Cafe with relevance scoring and analysis"""
    cafe_info: CafeInfo
    relevance_score: float
    matching_keywords: List[str]
    negative_keywords_found: List[str]
    search_keyword_origins: Set[str]
    category: str
    confidence: ConfidenceLevel
    analysis_notes: str = ""
    search_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.cafe_info.name,
            'url': self.cafe_info.url,
            'description': self.cafe_info.description,
            'relevance_score': round(self.relevance_score, 2),
            'matching_keywords': list(self.matching_keywords),
            'negative_keywords_found': list(self.negative_keywords_found),
            'search_keyword_origin': ', '.join(self.search_keyword_origins),
            'category': self.category,
            'confidence': self.confidence.value,
            'analysis_notes': self.analysis_notes,
            'search_timestamp': self.search_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FilteredCafe':
        """Create instance from dictionary"""
        cafe_info = CafeInfo(
            name=data['name'],
            url=data['url'],
            cafe_id=data['url'].split('/')[-1],
            description=data.get('description', '')
        )
        
        return cls(
            cafe_info=cafe_info,
            relevance_score=data['relevance_score'],
            matching_keywords=data['matching_keywords'],
            negative_keywords_found=data.get('negative_keywords_found', []),
            search_keyword_origins=set(data['search_keyword_origin'].split(', ')),
            category=data.get('category', '기타'),
            confidence=ConfidenceLevel(data.get('confidence', ConfidenceLevel.MEDIUM.value)),
            analysis_notes=data.get('analysis_notes', ''),
            search_timestamp=datetime.fromisoformat(data['search_timestamp'])
        )


@dataclass
class SearchProgress:
    """Progress information for real-time updates"""
    current_step: str
    progress_percentage: int
    message: str
    current_keyword: Optional[str] = None
    results_found: int = 0
    cafes_analyzed: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for SSE transmission"""
        return {
            'progress': self.progress_percentage,
            'message': self.message,
            'current_keyword': self.current_keyword,
            'results_found': self.results_found,
            'cafes_analyzed': self.cafes_analyzed
        }


@dataclass
class SearchStatistics:
    """Statistics about a search operation"""
    total_queries_generated: int = 0
    total_results_fetched: int = 0
    unique_cafes_found: int = 0
    cafes_after_filtering: int = 0
    search_duration_seconds: float = 0.0
    keywords_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class CafeAnalysis:
    """Detailed analysis of a cafe"""
    cafe_info: CafeInfo
    content_themes: List[str] = field(default_factory=list)
    activity_level: str = "Unknown"
    primary_topics: List[str] = field(default_factory=list)
    sentiment: str = "Neutral"
    quality_indicators: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize quality indicators if not provided"""
        if not self.quality_indicators:
            self.quality_indicators = {
                'has_rules': False,
                'has_categories': False,
                'active_moderation': False,
                'regular_events': False,
                'quality_content': False
            }


class SearchCache:
    """Simple in-memory cache for search results"""
    
    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache with TTL
        
        Args:
            ttl_seconds: Time to live for cached entries
        """
        self._cache: Dict[str, tuple[datetime, List[SearchResult]]] = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, query: str) -> Optional[List[SearchResult]]:
        """Get cached results if not expired"""
        if query in self._cache:
            timestamp, results = self._cache[query]
            if (datetime.now() - timestamp).total_seconds() < self.ttl_seconds:
                return results
            else:
                # Remove expired entry
                del self._cache[query]
        return None
    
    def set(self, query: str, results: List[SearchResult]):
        """Cache search results"""
        self._cache[query] = (datetime.now(), results)
    
    def clear_expired(self):
        """Remove all expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (timestamp, _) in self._cache.items()
            if (now - timestamp).total_seconds() >= self.ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]
    
    def clear_all(self):
        """Clear entire cache"""
        self._cache.clear()