#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Processor using Kiwipiepy
=============================
Handles Korean text processing with superior accuracy and Windows compatibility
"""

import logging
from typing import List, Set, Tuple, Optional
from collections import Counter

try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    logging.warning("Kiwipiepy not available. Install with: pip install kiwipiepy")

logger = logging.getLogger(__name__)


class NLPProcessor:
    """
    Korean NLP processor using Kiwipiepy for morphological analysis
    
    Kiwipiepy advantages over Mecab:
    - Easy installation on Windows (pip install kiwipiepy)
    - 80-90% accuracy in ambiguity resolution (vs 40-50% for others)
    - Pure C++ implementation with Python bindings
    - No external dependencies (Java, etc.)
    """
    
    def __init__(self):
        """Initialize the Kiwi analyzer"""
        self.kiwi = None
        self.stop_words = {
            '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', 
            '도', '로', '으로', '만', '까지', '에서', '에게', '한테',
            '부터', '까지', '마다', '처럼', '같이', '보다', '라고', '하고'
        }
        
        if KIWI_AVAILABLE:
            try:
                self.kiwi = Kiwi()
                # Add user words for better cafe name recognition
                self._add_user_words()
                logger.info("Kiwipiepy NLP engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Kiwi: {e}")
                self.kiwi = None
        else:
            logger.warning("Kiwipiepy not available - using fallback tokenizer")
    
    def _add_user_words(self):
        """Add domain-specific words to improve analysis accuracy"""
        if not self.kiwi:
            return
            
        user_words = [
            ('카페', 'NNP'),  # Cafe as proper noun
            ('네이버', 'NNP'),  # Naver
            ('동호회', 'NNG'),  # Club/group
            ('스터디', 'NNG'),  # Study group
            ('모임', 'NNG'),   # Gathering
        ]
        
        for word, pos in user_words:
            try:
                self.kiwi.add_user_word(word, pos, 0.0)
            except:
                pass  # Ignore errors for individual words
    
    def extract_nouns(self, text: str, min_length: int = 2) -> List[str]:
        """
        Extract meaningful nouns from Korean text
        
        Args:
            text: Input text to analyze
            min_length: Minimum length of nouns to extract
            
        Returns:
            List of extracted nouns in order of appearance
        """
        if not text or not text.strip():
            return []
        
        if self.kiwi:
            return self._extract_nouns_kiwi(text, min_length)
        else:
            return self._extract_nouns_fallback(text, min_length)
    
    def _extract_nouns_kiwi(self, text: str, min_length: int) -> List[str]:
        """Extract nouns using Kiwipiepy"""
        try:
            # Analyze text with Kiwi
            result = self.kiwi.analyze(text)
            
            if not result or not result[0]:
                return []
            
            # Get the best analysis result
            tokens = result[0][0]
            
            nouns = []
            for token in tokens:
                # Extract nouns (NNG: general noun, NNP: proper noun)
                if token.tag.startswith('N') and len(token.form) >= min_length:
                    if token.form not in self.stop_words:
                        nouns.append(token.form)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_nouns = []
            for noun in nouns:
                if noun not in seen:
                    seen.add(noun)
                    unique_nouns.append(noun)
            
            logger.debug(f"Extracted {len(unique_nouns)} nouns from text")
            return unique_nouns
            
        except Exception as e:
            logger.error(f"Error in Kiwi noun extraction: {e}")
            return self._extract_nouns_fallback(text, min_length)
    
    def _extract_nouns_fallback(self, text: str, min_length: int) -> List[str]:
        """Simple fallback noun extraction when Kiwi is not available"""
        import re
        
        # Remove special characters and split
        words = re.findall(r'[가-힣]+', text)
        
        # Filter by length and common patterns
        nouns = []
        for word in words:
            if len(word) >= min_length and word not in self.stop_words:
                # Simple heuristic: words ending in common noun suffixes
                if (word.endswith(('성', '가', '자', '화', '부', '실', '과', '팀')) or
                    len(word) >= 2):
                    nouns.append(word)
        
        return nouns
    
    def extract_key_phrases(self, text: str, max_words: int = 3) -> List[str]:
        """
        Extract key phrases (compound nouns) from text
        
        Args:
            text: Input text
            max_words: Maximum words in a phrase
            
        Returns:
            List of key phrases
        """
        if not self.kiwi or not text:
            return []
        
        try:
            result = self.kiwi.analyze(text)
            if not result or not result[0]:
                return []
            
            tokens = result[0][0]
            phrases = []
            current_phrase = []
            
            for token in tokens:
                if token.tag.startswith('N'):
                    current_phrase.append(token.form)
                    if len(current_phrase) >= max_words:
                        phrases.append(' '.join(current_phrase))
                        current_phrase = current_phrase[1:]  # Sliding window
                else:
                    if len(current_phrase) >= 2:
                        phrases.append(' '.join(current_phrase))
                    current_phrase = []
            
            # Don't forget the last phrase
            if len(current_phrase) >= 2:
                phrases.append(' '.join(current_phrase))
            
            return phrases
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []
    
    def calculate_keyword_relevance(self, text: str, keywords: List[str]) -> float:
        """
        Calculate relevance score based on keyword frequency and position
        
        Args:
            text: Text to analyze
            keywords: Keywords to check
            
        Returns:
            Relevance score (0-100)
        """
        if not text or not keywords:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Count occurrences
            count = text_lower.count(keyword_lower)
            if count > 0:
                # Base score for presence
                score += 10 * min(count, 3)  # Cap at 3 occurrences
                
                # Bonus for early appearance
                first_pos = text_lower.find(keyword_lower)
                if first_pos < len(text) * 0.2:  # In first 20%
                    score += 5
        
        return min(score, 100.0)
    
    def suggest_synonyms(self, word: str) -> List[str]:
        """
        Suggest synonyms or related words (simplified version)
        
        Args:
            word: Input word
            
        Returns:
            List of related words
        """
        # Simple synonym mapping for common search terms
        synonym_map = {
            '개발': ['개발자', '프로그래밍', '코딩', '개발팀'],
            '프로그래밍': ['개발', '코딩', '프로그래머'],
            '창업': ['스타트업', '사업', '비즈니스', '창업자'],
            '모임': ['동호회', '클럽', '커뮤니티', '그룹'],
            '스터디': ['학습', '공부', '스터디그룹'],
            '정보': ['자료', '데이터', '인포', '정보공유']
        }
        
        word_lower = word.lower()
        return synonym_map.get(word_lower, [])


# Singleton instance
_nlp_processor = None

def get_nlp_processor() -> NLPProcessor:
    """Get or create the singleton NLP processor instance"""
    global _nlp_processor
    if _nlp_processor is None:
        _nlp_processor = NLPProcessor()
    return _nlp_processor