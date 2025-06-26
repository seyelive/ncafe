#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AI-Powered Cafe Search Engine
=====================================
Implements semantic search, hidden gems discovery, and diversity algorithms
"""

import asyncio
import logging
import numpy as np
import time
from typing import List, Dict, Optional, Callable, Set, Tuple
from collections import defaultdict, Counter
import random
from dataclasses import dataclass

from config import CONFIG, NEGATIVE_KEYWORDS, CATEGORY_KEYWORDS
from models import (
    SearchResult, CafeInfo, FilteredCafe, SearchProgress, 
    ConfidenceLevel, SearchStatistics, SearchCache
)
from cafe_searcher import EnhancedCafeSearcher
from enhanced_nlp_processor import get_enhanced_nlp

logger = logging.getLogger(__name__)


@dataclass
class AISearchSettings:
    """AI search configuration settings"""
    semantic_search: bool = True
    hidden_gems: bool = True
    diversity_mode: bool = True
    max_semantic_candidates: int = 1000
    diversity_lambda: float = 0.7  # Balance between relevance and diversity
    novelty_boost: float = 1.2  # Boost factor for less popular cafes


class HiddenGemsDiscovery:
    """Algorithm for discovering hidden gem cafes"""
    
    def __init__(self):
        """Initialize hidden gems discovery"""
        self.popularity_cache = {}
        self.freshness_cache = {}
    
    def calculate_hidden_gem_score(self, 
                                 cafe_info: CafeInfo, 
                                 base_relevance: float,
                                 search_keywords: List[str]) -> float:
        """
        Calculate hidden gem score based on multiple factors
        
        Args:
            cafe_info: Basic cafe information
            base_relevance: Base relevance score
            search_keywords: Search keywords
            
        Returns:
            Adjusted score with hidden gem boost
        """
        # Estimate popularity (inverse of hidden gem potential)
        popularity_score = self._estimate_popularity(cafe_info)
        
        # Calculate novelty factor
        novelty_score = self._calculate_novelty(cafe_info, search_keywords)
        
        # Calculate quality indicators
        quality_score = self._assess_quality(cafe_info)
        
        # Hidden gem formula: boost less popular but high-quality cafes
        hidden_gem_factor = 1.0
        
        if popularity_score < 0.3:  # Low popularity
            if quality_score > 0.7:  # High quality
                hidden_gem_factor = 1.5  # Significant boost
            elif quality_score > 0.5:  # Medium quality
                hidden_gem_factor = 1.3  # Moderate boost
        
        # Additional boost for novel content
        if novelty_score > 0.8:
            hidden_gem_factor *= 1.2
        
        adjusted_score = base_relevance * hidden_gem_factor
        
        logger.debug(f"Hidden gem analysis for {cafe_info.name}: "
                    f"popularity={popularity_score:.2f}, "
                    f"novelty={novelty_score:.2f}, "
                    f"quality={quality_score:.2f}, "
                    f"boost={hidden_gem_factor:.2f}")
        
        return min(adjusted_score, 100.0)  # Cap at 100
    
    def _estimate_popularity(self, cafe_info: CafeInfo) -> float:
        """Estimate cafe popularity based on various signals"""
        popularity_indicators = 0
        total_indicators = 0
        
        # URL pattern analysis (simpler URLs often = more popular)
        if cafe_info.url:
            url_simplicity = len(cafe_info.url.split('/'))
            if url_simplicity <= 4:  # Very simple URL
                popularity_indicators += 1
            total_indicators += 1
        
        # Name length (shorter names often = more established)
        if cafe_info.name:
            if len(cafe_info.name) <= 10:  # Short name
                popularity_indicators += 1
            total_indicators += 1
        
        # Description length (longer descriptions might indicate more effort)
        if cafe_info.description:
            if len(cafe_info.description) > 100:  # Detailed description
                popularity_indicators -= 0.5  # Actually reduces popularity score
            total_indicators += 1
        
        popularity_score = popularity_indicators / max(total_indicators, 1)
        return max(0.0, min(1.0, popularity_score))
    
    def _calculate_novelty(self, cafe_info: CafeInfo, search_keywords: List[str]) -> float:
        """Calculate novelty based on unique content patterns"""
        novelty_score = 0.0
        
        # Check for unique keywords in cafe name
        if cafe_info.name:
            name_words = set(cafe_info.name.split())
            search_words = set(' '.join(search_keywords).split())
            unique_words = name_words - search_words
            
            if len(unique_words) > 0:
                novelty_score += 0.3
        
        # Check for detailed descriptions (indicates effort)
        if cafe_info.description and len(cafe_info.description) > 50:
            novelty_score += 0.4
        
        # Random novelty factor to ensure some variety
        novelty_score += random.random() * 0.3
        
        return min(novelty_score, 1.0)
    
    def _assess_quality(self, cafe_info: CafeInfo) -> float:
        """Assess content quality indicators"""
        quality_score = 0.5  # Base quality
        
        # Name quality (not too short, not too long)
        if cafe_info.name:
            name_len = len(cafe_info.name)
            if 5 <= name_len <= 20:  # Optimal name length
                quality_score += 0.2
        
        # Description quality
        if cafe_info.description:
            desc_len = len(cafe_info.description)
            if 30 <= desc_len <= 200:  # Good description length
                quality_score += 0.3
        
        return min(quality_score, 1.0)


class DiversityManager:
    """Manages result diversity to avoid similar cafes"""
    
    def __init__(self, diversity_lambda: float = 0.7):
        """
        Initialize diversity manager
        
        Args:
            diversity_lambda: Balance between relevance (1.0) and diversity (0.0)
        """
        self.diversity_lambda = diversity_lambda
        self.nlp = get_enhanced_nlp()
    
    def diversify_results(self, 
                         cafes: List[FilteredCafe], 
                         max_results: int = 50) -> List[FilteredCafe]:
        """
        Apply Maximal Marginal Relevance (MMR) for diversity
        
        Args:
            cafes: List of filtered cafes
            max_results: Maximum number of results to return
            
        Returns:
            Diversified list of cafes
        """
        if len(cafes) <= max_results:
            return cafes
        
        # Sort by relevance initially
        cafes.sort(key=lambda x: x.relevance_score, reverse=True)
        
        selected = []
        remaining = cafes.copy()
        
        # Select the highest relevance cafe first
        if remaining:
            selected.append(remaining.pop(0))
        
        # Apply MMR algorithm
        while len(selected) < max_results and remaining:
            best_score = -1
            best_cafe = None
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Calculate relevance score (normalized to 0-1)
                relevance = candidate.relevance_score / 100.0
                
                # Calculate maximum similarity to already selected cafes
                max_similarity = 0.0
                for selected_cafe in selected:
                    similarity = self._calculate_cafe_similarity(candidate, selected_cafe)
                    max_similarity = max(max_similarity, similarity)
                
                # MMR formula
                mmr_score = (self.diversity_lambda * relevance - 
                           (1 - self.diversity_lambda) * max_similarity)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_cafe = candidate
                    best_idx = i
            
            if best_cafe:
                selected.append(best_cafe)
                remaining.pop(best_idx)
        
        logger.info(f"Diversified {len(cafes)} cafes to {len(selected)} results")
        return selected
    
    def _calculate_cafe_similarity(self, cafe1: FilteredCafe, cafe2: FilteredCafe) -> float:
        """Calculate similarity between two cafes"""
        # Text similarity
        text1 = f"{cafe1.cafe_info.name} {cafe1.cafe_info.description}"
        text2 = f"{cafe2.cafe_info.name} {cafe2.cafe_info.description}"
        text_similarity = self.nlp.calculate_semantic_similarity(text1, text2)
        
        # Category similarity
        category_similarity = 1.0 if cafe1.category == cafe2.category else 0.0
        
        # Keyword overlap similarity
        keywords1 = set(cafe1.matching_keywords)
        keywords2 = set(cafe2.matching_keywords)
        
        if keywords1 or keywords2:
            keyword_similarity = (len(keywords1.intersection(keywords2)) / 
                                len(keywords1.union(keywords2)))
        else:
            keyword_similarity = 0.0
        
        # Weighted combination
        overall_similarity = (
            text_similarity * 0.5 +
            category_similarity * 0.3 +
            keyword_similarity * 0.2
        )
        
        return overall_similarity


class SemanticSearchEngine:
    """Semantic search engine for Korean cafe discovery"""
    
    def __init__(self):
        """Initialize semantic search engine"""
        self.nlp = get_enhanced_nlp()
        self.hidden_gems = HiddenGemsDiscovery()
        self.diversity_manager = DiversityManager()
        
    async def enhanced_search(self,
                            base_searcher: EnhancedCafeSearcher,
                            search_keywords: List[str],
                            exclude_keywords: List[str] = None,
                            min_score: float = 40,
                            ai_settings: AISearchSettings = None,
                            progress_callback: Optional[Callable] = None) -> List[FilteredCafe]:
        """
        Perform enhanced semantic search with AI features
        
        Args:
            base_searcher: Base cafe searcher instance
            search_keywords: Original search keywords
            exclude_keywords: Keywords to exclude
            min_score: Minimum relevance score
            ai_settings: AI search configuration
            progress_callback: Progress callback function
            
        Returns:
            List of enhanced filtered cafes
        """
        if ai_settings is None:
            ai_settings = AISearchSettings()
        
        start_time = time.time()
        
        # Extract features from search query
        self._update_progress(progress_callback, "🧠 AI가 검색 의도를 분석하고 있습니다...", 10)
        
        search_text = ' '.join(search_keywords)
        search_features = self.nlp.extract_enhanced_features(search_text)
        
        logger.info(f"Search features extracted: {search_features}")
        
        # Expand queries if semantic search is enabled
        if ai_settings.semantic_search:
            self._update_progress(progress_callback, "📝 검색어를 확장하고 있습니다...", 20)
            expanded_queries = self.nlp.expand_query(search_text, max_expansions=8)
            all_keywords = search_keywords.copy()
            
            # Add expanded terms
            for query in expanded_queries[1:]:  # Skip original
                expanded_features = self.nlp.extract_enhanced_features(query)
                all_keywords.extend(expanded_features['keywords'][:2])
            
            # Remove duplicates
            all_keywords = list(dict.fromkeys(all_keywords))
            logger.info(f"Expanded keywords: {all_keywords}")
        else:
            all_keywords = search_keywords
        
        # Perform base search with expanded keywords
        self._update_progress(progress_callback, "🔍 네이버에서 카페 데이터를 수집하고 있습니다...", 30)
        
        base_results = await base_searcher.search_cafes(
            all_keywords,
            exclude_keywords,
            min_score=20,  # Lower threshold for more candidates
            progress_callback=None  # We'll handle progress ourselves
        )
        
        # Enhanced analysis and scoring
        self._update_progress(progress_callback, "🤖 AI가 카페들을 분석하고 있습니다...", 60)
        
        enhanced_results = await self._analyze_cafes_with_ai(
            base_results,
            search_keywords,
            search_features,
            ai_settings,
            progress_callback
        )
        
        # Apply minimum score filter
        filtered_results = [cafe for cafe in enhanced_results if cafe.relevance_score >= min_score]
        
        # Apply diversity if enabled
        if ai_settings.diversity_mode and len(filtered_results) > 20:
            self._update_progress(progress_callback, "🎯 결과 다양성을 보장하고 있습니다...", 85)
            filtered_results = self.diversity_manager.diversify_results(filtered_results, max_results=50)
        
        # Sort by final relevance score
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Final progress update
        duration = time.time() - start_time
        self._update_progress(
            progress_callback,
            f"✅ AI 검색 완료! {len(filtered_results)}개의 최적화된 결과를 발견했습니다.",
            100,
            results=filtered_results
        )
        
        logger.info(f"Enhanced search completed in {duration:.2f}s: "
                   f"{len(base_results)} -> {len(filtered_results)} cafes")
        
        return filtered_results
    
    async def _analyze_cafes_with_ai(self,
                                   cafes: List[FilteredCafe],
                                   search_keywords: List[str],
                                   search_features: Dict,
                                   ai_settings: AISearchSettings,
                                   progress_callback: Optional[Callable]) -> List[FilteredCafe]:
        """Analyze cafes using AI techniques"""
        enhanced_cafes = []
        total_cafes = len(cafes)
        
        # Batch processing for efficiency
        batch_size = 10
        for i in range(0, total_cafes, batch_size):
            batch = cafes[i:i + batch_size]
            
            for j, cafe in enumerate(batch):
                try:
                    # Enhanced relevance scoring
                    enhanced_cafe = await self._enhance_cafe_analysis(
                        cafe,
                        search_keywords,
                        search_features,
                        ai_settings
                    )
                    
                    if enhanced_cafe:
                        enhanced_cafes.append(enhanced_cafe)
                
                except Exception as e:
                    logger.error(f"Error analyzing cafe {cafe.cafe_info.name}: {e}")
                    # Keep original cafe if enhancement fails
                    enhanced_cafes.append(cafe)
            
            # Update progress
            progress = 60 + int(((i + len(batch)) / total_cafes) * 25)
            self._update_progress(
                progress_callback,
                f"🤖 AI 분석 중... ({i + len(batch)}/{total_cafes})",
                progress,
                cafes_analyzed=i + len(batch)
            )
        
        return enhanced_cafes
    
    async def _enhance_cafe_analysis(self,
                                   cafe: FilteredCafe,
                                   search_keywords: List[str],
                                   search_features: Dict,
                                   ai_settings: AISearchSettings) -> Optional[FilteredCafe]:
        """Enhance individual cafe analysis with AI"""
        
        # Combine all cafe text for analysis
        cafe_text = f"{cafe.cafe_info.name} {cafe.cafe_info.description}"
        if hasattr(cafe, 'additional_text'):
            cafe_text += f" {cafe.additional_text}"
        
        # Calculate enhanced relevance score
        relevance_score, score_details = self.nlp.calculate_relevance_score(
            cafe_text,
            search_keywords,
            search_features
        )
        
        # Apply hidden gems boost if enabled
        if ai_settings.hidden_gems:
            relevance_score = self.hidden_gems.calculate_hidden_gem_score(
                cafe.cafe_info,
                relevance_score,
                search_keywords
            )
        
        # Update cafe with enhanced information
        enhanced_cafe = FilteredCafe(
            cafe_info=cafe.cafe_info,
            relevance_score=relevance_score,
            matching_keywords=score_details.get('matching_keywords', cafe.matching_keywords),
            negative_keywords_found=cafe.negative_keywords_found,
            search_keyword_origins=cafe.search_keyword_origins,
            category=self._determine_enhanced_category(cafe_text, search_features),
            confidence=self._calculate_enhanced_confidence(relevance_score, score_details),
            analysis_notes=self._generate_enhanced_notes(score_details, ai_settings),
            search_timestamp=cafe.search_timestamp
        )
        
        return enhanced_cafe
    
    def _determine_enhanced_category(self, cafe_text: str, search_features: Dict) -> str:
        """Determine category using enhanced analysis"""
        # Use NLP-extracted categories if available
        cafe_features = self.nlp.extract_enhanced_features(cafe_text)
        
        if cafe_features['categories']:
            return cafe_features['categories'][0]
        
        # Fallback to original category logic
        text_lower = cafe_text.lower()
        scores = defaultdict(int)
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[category] += 1
        
        if scores:
            return max(scores, key=scores.get)
        
        return "기타"
    
    def _calculate_enhanced_confidence(self, score: float, score_details: Dict) -> ConfidenceLevel:
        """Calculate confidence level based on multiple factors"""
        # Base confidence from score
        base_confidence = ConfidenceLevel.LOW
        if score >= 90:
            base_confidence = ConfidenceLevel.VERY_HIGH
        elif score >= 70:
            base_confidence = ConfidenceLevel.HIGH
        elif score >= 50:
            base_confidence = ConfidenceLevel.MEDIUM
        
        # Adjust based on score details
        semantic_score = score_details.get('semantic_similarity', 0)
        keyword_score = score_details.get('keyword_match', 0)
        
        # High semantic similarity increases confidence
        if semantic_score > 80 and keyword_score > 60:
            if base_confidence == ConfidenceLevel.MEDIUM:
                base_confidence = ConfidenceLevel.HIGH
            elif base_confidence == ConfidenceLevel.HIGH:
                base_confidence = ConfidenceLevel.VERY_HIGH
        
        return base_confidence
    
    def _generate_enhanced_notes(self, score_details: Dict, ai_settings: AISearchSettings) -> str:
        """Generate enhanced analysis notes"""
        notes = []
        
        # Score breakdown
        if score_details.get('semantic_similarity', 0) > 70:
            notes.append("의미적 유사도 높음")
        
        if score_details.get('keyword_match', 0) > 80:
            notes.append("키워드 매칭 우수")
        
        if score_details.get('category_match', 0) > 60:
            notes.append("카테고리 일치")
        
        # AI features used
        if ai_settings.semantic_search:
            notes.append("AI 의미 분석 적용")
        
        if ai_settings.hidden_gems:
            notes.append("숨겨진 보석 알고리즘 적용")
        
        return " • ".join(notes) if notes else "기본 분석"
    
    def _update_progress(self,
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
            try:
                callback(progress_data)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")


class EnhancedAICafeSearcher(EnhancedCafeSearcher):
    """Enhanced cafe searcher with AI capabilities"""
    
    def __init__(self, client_id: str, client_secret: str):
        """Initialize enhanced AI cafe searcher"""
        super().__init__(client_id, client_secret)
        self.semantic_engine = SemanticSearchEngine()
        
    async def search_cafes_with_ai(self,
                                 search_keywords: List[str],
                                 exclude_keywords: List[str] = None,
                                 min_score: float = 40,
                                 ai_settings: AISearchSettings = None,
                                 progress_callback: Optional[Callable] = None) -> List[FilteredCafe]:
        """
        Perform AI-enhanced cafe search
        
        Args:
            search_keywords: List of search keywords
            exclude_keywords: Keywords to exclude from results
            min_score: Minimum relevance score
            ai_settings: AI search configuration
            progress_callback: Callback for progress updates
            
        Returns:
            List of AI-enhanced filtered cafes
        """
        return await self.semantic_engine.enhanced_search(
            base_searcher=self,
            search_keywords=search_keywords,
            exclude_keywords=exclude_keywords,
            min_score=min_score,
            ai_settings=ai_settings,
            progress_callback=progress_callback
        )