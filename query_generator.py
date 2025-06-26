#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Query Generator for Naver Search
===========================================
Generates sophisticated search queries using NLP and Naver's advanced operators
"""

import logging
from typing import List, Dict, Set, Tuple
from itertools import combinations

from nlp_processor import get_nlp_processor

logger = logging.getLogger(__name__)


class QueryStrategy:
    """Base class for query generation strategies"""
    
    def generate(self, nouns: List[str], original_text: str) -> List[str]:
        """Generate queries based on the strategy"""
        raise NotImplementedError


class ExactPhraseStrategy(QueryStrategy):
    """
    Strategy 1: Use exact phrase matching for core concepts
    Uses "" operator for precise matching
    """
    
    def generate(self, nouns: List[str], original_text: str) -> List[str]:
        queries = []
        
        if len(nouns) >= 2:
            # Create exact phrase from first 2-3 nouns
            core_phrase = f'"{nouns[0]} {nouns[1]}"'
            
            # Add remaining nouns as required terms
            if len(nouns) > 2:
                required_terms = ' '.join([f'+{noun}' for noun in nouns[2:5]])  # Limit to avoid too long queries
                query = f'{core_phrase} {required_terms}'
            else:
                query = core_phrase
            
            queries.append(query)
            
            # Alternative: Different noun combinations
            if len(nouns) >= 3:
                alt_phrase = f'"{nouns[1]} {nouns[2]}"'
                required = f'+{nouns[0]}'
                queries.append(f'{alt_phrase} {required}')
        
        return queries


class MandatoryTermsStrategy(QueryStrategy):
    """
    Strategy 2: All terms are mandatory using + operator
    Ensures all key concepts appear in results
    """
    
    def generate(self, nouns: List[str], original_text: str) -> List[str]:
        queries = []
        
        if nouns:
            # All nouns as mandatory
            all_mandatory = ' '.join([f'+{noun}' for noun in nouns[:6]])  # Limit for query length
            queries.append(all_mandatory)
            
            # Core nouns mandatory, others optional
            if len(nouns) > 3:
                core_mandatory = ' '.join([f'+{noun}' for noun in nouns[:3]])
                optional = ' '.join(nouns[3:5])
                queries.append(f'{core_mandatory} {optional}')
        
        return queries


class OrCombinationStrategy(QueryStrategy):
    """
    Strategy 3: Use OR operator for broader results
    Good for finding related content
    """
    
    def generate(self, nouns: List[str], original_text: str) -> List[str]:
        queries = []
        
        if len(nouns) >= 2:
            # OR combination of main terms
            or_query = ' | '.join(nouns[:4])  # Limit to 4 terms
            queries.append(or_query)
            
            # Mix of mandatory and OR
            if len(nouns) > 2:
                mandatory = f'+{nouns[0]}'
                or_terms = ' | '.join(nouns[1:4])
                queries.append(f'{mandatory} ({or_terms})')
        
        return queries


class ProximityStrategy(QueryStrategy):
    """
    Strategy 4: Ensure terms appear close together
    Uses NEAR operator concept (simulated with phrases)
    """
    
    def generate(self, nouns: List[str], original_text: str) -> List[str]:
        queries = []
        
        if len(nouns) >= 2:
            # Create multiple phrase combinations
            for i in range(min(len(nouns) - 1, 3)):
                phrase = f'"{nouns[i]} {nouns[i+1]}"'
                queries.append(phrase)
        
        return queries


class NegativeFilterStrategy(QueryStrategy):
    """
    Strategy 5: Exclude common unwanted terms
    Uses - operator to filter out commercial/spam content
    """
    
    def generate(self, nouns: List[str], original_text: str) -> List[str]:
        queries = []
        
        # Common terms to exclude in cafe searches
        exclude_terms = ['-중고', '-판매', '-구매', '-팝니다', '-삽니다']
        
        if nouns:
            base_query = ' '.join(nouns[:4])
            filtered_query = f'{base_query} {" ".join(exclude_terms[:3])}'
            queries.append(filtered_query)
        
        return queries


class IntelligentQueryGenerator:
    """
    Generates sophisticated search queries using multiple strategies
    and Naver's advanced search operators
    """
    
    def __init__(self):
        """Initialize the query generator with NLP processor and strategies"""
        self.nlp = get_nlp_processor()
        self.strategies = [
            ExactPhraseStrategy(),
            MandatoryTermsStrategy(),
            OrCombinationStrategy(),
            ProximityStrategy(),
            NegativeFilterStrategy()
        ]
        
        # Query templates for specific search intents
        self.intent_templates = {
            'community': ['"{0} 카페"', '"{0} 모임"', '"{0} 동호회"', '+{0} +카페 +가입'],
            'information': ['"{0} 정보"', '+{0} +자료', '{0} 가이드'],
            'study': ['"{0} 스터디"', '"{0} 학습"', '+{0} +공부'],
            'local': ['"{0} {1}"', '+{0} +{1} +지역']  # For location-based
        }
    
    def generate_queries(self, 
                        search_text: str, 
                        search_intent: str = 'general',
                        exclude_keywords: List[str] = None) -> List[str]:
        """
        Generate optimized search queries from natural language input
        
        Args:
            search_text: User's natural language search input
            search_intent: Type of search (general, community, information, etc.)
            exclude_keywords: Additional keywords to exclude
            
        Returns:
            List of optimized search queries
        """
        if not search_text:
            return []
        
        # Extract nouns and key phrases
        nouns = self.nlp.extract_nouns(search_text)
        key_phrases = self.nlp.extract_key_phrases(search_text)
        
        if not nouns:
            # Fallback to original text if no nouns extracted
            logger.warning(f"No nouns extracted from: {search_text}")
            return [search_text]
        
        logger.info(f"Extracted nouns: {nouns}")
        logger.info(f"Extracted phrases: {key_phrases}")
        
        all_queries = set()  # Use set to avoid duplicates
        
        # Apply all strategies
        for strategy in self.strategies:
            try:
                queries = strategy.generate(nouns, search_text)
                all_queries.update(queries)
            except Exception as e:
                logger.error(f"Error in {strategy.__class__.__name__}: {e}")
        
        # Add intent-specific queries
        if search_intent in self.intent_templates and nouns:
            for template in self.intent_templates[search_intent]:
                try:
                    # Fill template with extracted nouns
                    if '{1}' in template and len(nouns) > 1:
                        query = template.format(nouns[0], nouns[1])
                    else:
                        query = template.format(nouns[0])
                    all_queries.add(query)
                except:
                    pass
        
        # Add key phrase queries
        for phrase in key_phrases[:3]:  # Limit to top 3 phrases
            all_queries.add(f'"{phrase}"')
        
        # Apply user-specified exclusions
        if exclude_keywords:
            exclude_str = ' '.join([f'-{kw}' for kw in exclude_keywords[:5]])
            filtered_queries = []
            for query in all_queries:
                # Add exclusions to queries that don't already have them
                if '-' not in query:
                    filtered_queries.append(f'{query} {exclude_str}')
                else:
                    filtered_queries.append(query)
            all_queries = set(filtered_queries)
        
        # Convert to list and sort by complexity (longer = more specific)
        final_queries = sorted(list(all_queries), key=len, reverse=True)
        
        # Limit number of queries to prevent overload
        max_queries = 10
        if len(final_queries) > max_queries:
            # Take a mix of different query types
            final_queries = self._select_diverse_queries(final_queries, max_queries)
        
        logger.info(f"Generated {len(final_queries)} queries for: {search_text}")
        return final_queries
    
    def _select_diverse_queries(self, queries: List[str], max_count: int) -> List[str]:
        """
        Select a diverse set of queries to maximize coverage
        
        Args:
            queries: All generated queries
            max_count: Maximum number to return
            
        Returns:
            Diverse subset of queries
        """
        selected = []
        
        # Categories based on operators used
        exact_phrases = [q for q in queries if '"' in q]
        mandatory = [q for q in queries if '+' in q and '"' not in q]
        with_or = [q for q in queries if '|' in q]
        with_exclude = [q for q in queries if '-' in q]
        simple = [q for q in queries if not any(op in q for op in ['"', '+', '|', '-'])]
        
        # Take some from each category
        categories = [exact_phrases, mandatory, with_or, with_exclude, simple]
        per_category = max_count // len(categories)
        
        for category in categories:
            selected.extend(category[:per_category])
        
        # Fill remaining slots with most specific queries
        remaining = max_count - len(selected)
        if remaining > 0:
            other_queries = [q for q in queries if q not in selected]
            selected.extend(other_queries[:remaining])
        
        return selected[:max_count]
    
    def generate_fallback_queries(self, search_text: str) -> List[str]:
        """
        Generate simple fallback queries when advanced generation fails
        
        Args:
            search_text: Original search text
            
        Returns:
            List of simple queries
        """
        # Simple word-based queries
        words = search_text.split()
        queries = [
            search_text,  # Original
            ' '.join(words[:3]) if len(words) > 3 else search_text,  # First 3 words
            ' '.join([f'+{w}' for w in words[:3]])  # First 3 as mandatory
        ]
        
        return list(set(queries))


# Singleton instance
_query_generator = None

def get_query_generator() -> IntelligentQueryGenerator:
    """Get or create the singleton query generator instance"""
    global _query_generator
    if _query_generator is None:
        _query_generator = IntelligentQueryGenerator()
    return _query_generator