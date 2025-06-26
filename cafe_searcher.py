#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Naver Cafe Searcher
============================
Core search engine with intelligent query generation and async processing
"""

import asyncio
import logging
import re
import time
from typing import List, Dict, Optional, Callable, Set
from collections import defaultdict
import hashlib

import aiohttp
from selectolax.parser import HTMLParser

from config import CONFIG, NEGATIVE_KEYWORDS, CATEGORY_KEYWORDS
from models import (
    SearchResult, CafeInfo, FilteredCafe, SearchProgress, 
    ConfidenceLevel, SearchStatistics, SearchCache
)
from nlp_processor import get_nlp_processor
from query_generator import get_query_generator

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
                logger.info("Circuit breaker closed - service recovered")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            raise


class EnhancedCafeSearcher:
    """
    Enhanced Naver Cafe Searcher with intelligent query generation
    and improved relevance scoring
    """
    
    def __init__(self, client_id: str, client_secret: str):
        """Initialize the searcher with API credentials"""
        self.client_id = client_id
        self.client_secret = client_secret
        self.headers = {
            'X-Naver-Client-Id': client_id,
            'X-Naver-Client-Secret': client_secret,
            'User-Agent': 'NaverCafeSearcher/3.0'
        }
        
        # Initialize components
        self.nlp = get_nlp_processor()
        self.query_generator = get_query_generator()
        self.circuit_breaker = CircuitBreaker()
        self.cache = SearchCache(ttl_seconds=CONFIG['search']['cache_ttl'])
        
        # Search configuration
        self.negative_keywords = NEGATIVE_KEYWORDS
        self.category_keywords = CATEGORY_KEYWORDS
        
        # Async session (created in context manager)
        self.session = None
        
        # Statistics
        self.stats = SearchStatistics()
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=30,
            limit_per_host=10,
            ttl_dns_cache=300
        )
        timeout = aiohttp.ClientTimeout(
            total=CONFIG['search']['request_timeout'],
            connect=10
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def search_cafes(self,
                          search_keywords: List[str],
                          exclude_keywords: List[str] = None,
                          min_score: float = 40,
                          progress_callback: Optional[Callable] = None) -> List[FilteredCafe]:
        """
        Perform comprehensive cafe search with intelligent query generation
        
        Args:
            search_keywords: List of search keywords
            exclude_keywords: Keywords to exclude from results
            min_score: Minimum relevance score
            progress_callback: Callback for progress updates
            
        Returns:
            List of filtered and scored cafes
        """
        start_time = time.time()
        self.stats = SearchStatistics()  # Reset statistics
        
        # Update progress
        await self._update_progress(progress_callback, "검색을 시작합니다...", 5)
        
        # Generate intelligent queries for each keyword
        all_queries = []
        for keyword in search_keywords:
            queries = self.query_generator.generate_queries(
                keyword, 
                search_intent='community',
                exclude_keywords=exclude_keywords
            )
            all_queries.extend(queries)
            self.stats.keywords_used.append(keyword)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in all_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        self.stats.total_queries_generated = len(unique_queries)
        logger.info(f"Generated {len(unique_queries)} unique queries from {len(search_keywords)} keywords")
        
        await self._update_progress(
            progress_callback, 
            f"{len(unique_queries)}개의 검색 쿼리를 생성했습니다.", 
            10
        )
        
        # Execute searches concurrently
        all_cafes = await self._execute_searches(unique_queries, progress_callback)
        
        self.stats.unique_cafes_found = len(all_cafes)
        
        await self._update_progress(
            progress_callback,
            f"{len(all_cafes)}개의 카페를 발견했습니다. 분석을 시작합니다...",
            50
        )
        
        # Analyze and filter cafes
        filtered_cafes = await self._analyze_cafes(
            all_cafes, 
            search_keywords,
            exclude_keywords,
            min_score,
            progress_callback
        )
        
        self.stats.cafes_after_filtering = len(filtered_cafes)
        self.stats.search_duration_seconds = time.time() - start_time
        
        # Sort by relevance score
        filtered_cafes.sort(key=lambda x: x.relevance_score, reverse=True)
        
        await self._update_progress(
            progress_callback,
            f"검색 완료! {len(filtered_cafes)}개의 관련 카페를 찾았습니다.",
            100,
            results=filtered_cafes
        )
        
        logger.info(f"Search completed: {self.stats.to_dict()}")
        
        return filtered_cafes
    
    async def _execute_searches(self, 
                               queries: List[str], 
                               progress_callback: Optional[Callable]) -> Dict[str, Dict]:
        """Execute search queries concurrently"""
        all_cafes = {}
        semaphore = asyncio.Semaphore(CONFIG['search']['concurrent_requests'])
        
        tasks = []
        for i, query in enumerate(queries):
            task = self._search_single_query(query, semaphore)
            tasks.append((i, query, task))
        
        # Process results as they complete
        for i, query, task in tasks:
            try:
                results = await task
                
                # Process each result
                for result in results:
                    cafe_url = result.cafe_url
                    if cafe_url and cafe_url not in all_cafes:
                        all_cafes[cafe_url] = {
                            'info': result.to_cafe_info(),
                            'descriptions': {result.description},
                            'origin_queries': {query},
                            'titles': {result.title}
                        }
                    elif cafe_url:
                        all_cafes[cafe_url]['descriptions'].add(result.description)
                        all_cafes[cafe_url]['origin_queries'].add(query)
                        all_cafes[cafe_url]['titles'].add(result.title)
                
                # Update progress
                progress = 10 + int(((i + 1) / len(queries)) * 40)
                await self._update_progress(
                    progress_callback,
                    f"검색 진행 중... ({i + 1}/{len(queries)})",
                    progress,
                    current_keyword=query,
                    results_found=len(all_cafes)
                )
                
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")
        
        return all_cafes
    
    async def _search_single_query(self, query: str, semaphore: asyncio.Semaphore) -> List[SearchResult]:
        """Search for a single query with caching and circuit breaker"""
        # Check cache first
        cached_results = self.cache.get(query)
        if cached_results is not None:
            logger.debug(f"Cache hit for query: {query}")
            return cached_results
        
        async with semaphore:
            results = []
            
            for start in range(1, CONFIG['search']['max_results_per_keyword'], 100):
                try:
                    # Use circuit breaker for API calls
                    page_results = await self.circuit_breaker.call(
                        self._fetch_search_page,
                        query,
                        start
                    )
                    
                    if not page_results:
                        break
                    
                    results.extend(page_results)
                    self.stats.total_results_fetched += len(page_results)
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error fetching page for '{query}' at start={start}: {e}")
                    break
            
            # Cache the results
            self.cache.set(query, results)
            
            return results
    
    async def _fetch_search_page(self, query: str, start: int) -> List[SearchResult]:
        """Fetch and parse a single search results page"""
        params = {
            'query': query,
            'display': 100,
            'start': start,
            'sort': 'sim'
        }
        
        url = "https://openapi.naver.com/v1/search/cafearticle.json"
        
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                logger.warning(f"API returned status {response.status} for query: {query}")
                return []
            
            data = await response.json()
            items = data.get('items', [])
            
            results = []
            for item in items:
                cafe_url = self._extract_cafe_url(item.get('link', ''))
                if not cafe_url:
                    continue
                
                result = SearchResult(
                    title=self._clean_text(item.get('title', '')),
                    link=item.get('link', ''),
                    description=self._clean_text(item.get('description', '')),
                    cafe_name=self._clean_text(item.get('cafename', '')),
                    cafe_url=cafe_url,
                    query_origin=query
                )
                results.append(result)
            
            return results
    
    def _extract_cafe_url(self, link: str) -> str:
        """Extract clean cafe URL from article link"""
        match = re.search(r'cafe\.naver\.com/([^/?]+)', link)
        return f"https://cafe.naver.com/{match.group(1)}" if match else ""
    
    def _clean_text(self, text: str) -> str:
        """Clean HTML tags and normalize text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove special characters but keep Korean
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        # Normalize whitespace
        return ' '.join(text.split()).strip()
    
    async def _analyze_cafes(self,
                           all_cafes: Dict[str, Dict],
                           search_keywords: List[str],
                           exclude_keywords: List[str],
                           min_score: float,
                           progress_callback: Optional[Callable]) -> List[FilteredCafe]:
        """Analyze cafes and calculate relevance scores"""
        filtered_cafes = []
        total_cafes = len(all_cafes)
        
        # Extract nouns from search keywords for better matching
        search_nouns = set()
        for keyword in search_keywords:
            nouns = self.nlp.extract_nouns(keyword)
            search_nouns.update(nouns)
        
        for i, (url, data) in enumerate(all_cafes.items()):
            cafe_info = data['info']
            
            # Combine all text for analysis
            full_text = f"{cafe_info.name} {' '.join(data['titles'])} {' '.join(data['descriptions'])}"
            
            # Skip if contains exclude keywords
            if exclude_keywords and self._should_exclude(full_text, exclude_keywords):
                continue
            
            # Calculate relevance score
            score, matching_kw, negative_kw = self._calculate_relevance_score(
                cafe_info.name,
                full_text,
                search_keywords,
                search_nouns
            )
            
            if score >= min_score:
                # Determine category
                category = self._determine_category(full_text)
                
                # Determine confidence
                if score >= 90:
                    confidence = ConfidenceLevel.VERY_HIGH
                elif score >= 70:
                    confidence = ConfidenceLevel.HIGH
                elif score >= 50:
                    confidence = ConfidenceLevel.MEDIUM
                else:
                    confidence = ConfidenceLevel.LOW
                
                # Create filtered cafe
                filtered_cafe = FilteredCafe(
                    cafe_info=cafe_info,
                    relevance_score=score,
                    matching_keywords=matching_kw,
                    negative_keywords_found=negative_kw,
                    search_keyword_origins=data['origin_queries'],
                    category=category,
                    confidence=confidence,
                    analysis_notes=self._generate_analysis_notes(score, matching_kw, category)
                )
                
                filtered_cafes.append(filtered_cafe)
            
            # Update progress
            if progress_callback and i % 10 == 0:
                progress = 50 + int(((i + 1) / total_cafes) * 40)
                await self._update_progress(
                    progress_callback,
                    f"카페 분석 중... ({i + 1}/{total_cafes})",
                    progress,
                    cafes_analyzed=i + 1
                )
        
        return filtered_cafes
    
    def _should_exclude(self, text: str, exclude_keywords: List[str]) -> bool:
        """Check if text contains exclude keywords"""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in exclude_keywords)
    
    def _calculate_relevance_score(self,
                                 cafe_name: str,
                                 full_text: str,
                                 search_keywords: List[str],
                                 search_nouns: Set[str]) -> tuple[float, List[str], List[str]]:
        """Calculate relevance score with improved algorithm"""
        text_lower = full_text.lower()
        name_lower = cafe_name.lower()
        
        matching_keywords = []
        positive_score = 0
        
        # Check full keywords
        for keyword in search_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                matching_keywords.append(keyword)
                positive_score += 50
                
                # Bonus if in cafe name
                if keyword_lower in name_lower:
                    positive_score += 30
        
        # Check individual nouns
        for noun in search_nouns:
            noun_lower = noun.lower()
            if noun_lower in text_lower and noun not in matching_keywords:
                matching_keywords.append(noun)
                positive_score += 20
                
                # Bonus if in cafe name
                if noun_lower in name_lower:
                    positive_score += 15
        
        # Use NLP for additional keyword relevance
        additional_score = self.nlp.calculate_keyword_relevance(full_text, list(search_nouns))
        positive_score += additional_score * 0.5
        
        # Check negative keywords
        negative_keywords_found = []
        negative_score = 0
        
        for kw, penalty in self.negative_keywords.items():
            if kw in text_lower:
                negative_keywords_found.append(kw)
                negative_score += penalty
        
        # Final score
        final_score = max(0, positive_score + negative_score)
        
        return final_score, matching_keywords, negative_keywords_found
    
    def _determine_category(self, text: str) -> str:
        """Determine cafe category based on content"""
        text_lower = text.lower()
        scores = defaultdict(int)
        
        for category, keywords in self.category_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[category] += 1
        
        if scores:
            return max(scores, key=scores.get)
        return "기타"
    
    def _generate_analysis_notes(self, score: float, keywords: List[str], category: str) -> str:
        """Generate analysis notes for the cafe"""
        notes = []
        
        if score >= 90:
            notes.append("매우 관련성 높은 카페")
        elif score >= 70:
            notes.append("관련성 높은 카페")
        
        if len(keywords) >= 3:
            notes.append(f"{len(keywords)}개의 키워드 매칭")
        
        if category != "기타":
            notes.append(f"{category} 관련 카페")
        
        return ". ".join(notes)
    
    async def _update_progress(self,
                             callback: Optional[Callable],
                             message: str,
                             progress: int,
                             **kwargs):
        """Update progress through callback"""
        if callback:
            progress_data = {
                'progress': progress,
                'message': message,
                **kwargs
            }
            callback(progress_data)