#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Korean NLP Processor with Semantic Search
==================================================
Implements Korean Sentence Transformers and advanced semantic similarity
"""

import logging
import pickle
import numpy as np
from typing import List, Set, Tuple, Dict, Optional
from collections import Counter, defaultdict
from pathlib import Path
import hashlib

# Try to import advanced NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    logging.warning("Kiwipiepy not available. Install with: pip install kiwipiepy")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Install with: pip install scikit-learn")

logger = logging.getLogger(__name__)


class SemanticCacheManager:
    """Manages caching for semantic embeddings and similarity computations"""
    
    def __init__(self, cache_dir: str = "semantic_cache"):
        """Initialize cache manager"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embedding_cache = {}
        self.similarity_cache = {}
        
    def get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        key = self.get_cache_key(text)
        if key in self.embedding_cache:
            return self.embedding_cache[key]
        
        # Try to load from disk
        cache_file = self.cache_dir / f"emb_{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                    self.embedding_cache[key] = embedding
                    return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        return None
    
    def save_embedding(self, text: str, embedding: np.ndarray):
        """Save embedding to cache"""
        key = self.get_cache_key(text)
        self.embedding_cache[key] = embedding
        
        # Save to disk
        cache_file = self.cache_dir / f"emb_{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")


class EnhancedKoreanNLP:
    """
    Enhanced Korean NLP processor with semantic search capabilities
    """
    
    def __init__(self, use_gpu: bool = False):
        """Initialize enhanced NLP processor"""
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.cache_manager = SemanticCacheManager()
        
        # Initialize Korean morphological analyzer
        self.kiwi = None
        if KIWI_AVAILABLE:
            try:
                self.kiwi = Kiwi()
                self._add_domain_words()
                logger.info("Kiwi morphological analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Kiwi: {e}")
        
        # Initialize sentence transformer for semantic similarity
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use Korean-optimized model
                model_name = "jhgan/ko-sroberta-multitask"
                device = 'cuda' if self.use_gpu else 'cpu'
                self.sentence_model = SentenceTransformer(model_name, device=device)
                logger.info(f"Korean Sentence Transformer loaded: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
        
        # Initialize TF-IDF for fallback similarity
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=None  # We'll handle Korean stopwords ourselves
            )
        
        # Korean stopwords and patterns
        self.stop_words = {
            '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', 
            '도', '로', '으로', '만', '까지', '에서', '에게', '한테',
            '부터', '까지', '마다', '처럼', '같이', '보다', '라고', '하고'
        }
        
        # Synonym dictionary for query expansion
        self.synonym_dict = {
            '개발': ['개발자', '프로그래밍', '코딩', 'dev', 'developer'],
            '프로그래밍': ['개발', '코딩', '프로그래머', 'programming'],
            '스터디': ['학습', '공부', '스터디그룹', 'study'],
            '모임': ['동호회', '클럽', '커뮤니티', '그룹', '모임'],
            '카페': ['커뮤니티', '모임', '동호회', '그룹'],
            '창업': ['스타트업', '사업', '비즈니스', 'startup'],
            '투자': ['재테크', '주식', '펀드', '투자'],
            '여행': ['여행', '관광', '여행지', 'travel'],
        }
        
        # Category keywords for better classification
        self.category_patterns = {
            "개발/IT": ['개발', '프로그래밍', '코딩', 'it', '개발자', '프로그래머', 'python', 'java', 'javascript'],
            "스터디/교육": ['스터디', '학습', '공부', '교육', '강의', '시험', '자격증', '토익', '토플'],
            "창업/비즈니스": ['창업', '스타트업', '사업', '비즈니스', '마케팅', '투자', '재테크'],
            "취미/문화": ['취미', '문화', '예술', '음악', '사진', '영화', '독서', '요리'],
            "지역/모임": ['지역', '모임', '동호회', '클럽', '커뮤니티', '친목'],
            "육아/가족": ['육아', '맘카페', '육아맘', '가족', '아이', '육아맘'],
            "여행/관광": ['여행', '관광', '여행지', '캠핑', '백패킹', '펜션'],
        }
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for processing"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _add_domain_words(self):
        """Add domain-specific words to Kiwi"""
        if not self.kiwi:
            return
        
        domain_words = [
            ('카페', 'NNP', 0.0),
            ('네이버카페', 'NNP', 0.0),
            ('스터디', 'NNG', 0.0),
            ('스터디그룹', 'NNG', 0.0),
            ('개발자', 'NNG', 0.0),
            ('프로그래머', 'NNG', 0.0),
            ('창업가', 'NNG', 0.0),
            ('스타트업', 'NNG', 0.0),
        ]
        
        for word, pos, score in domain_words:
            try:
                self.kiwi.add_user_word(word, pos, score)
            except Exception:
                pass
    
    def extract_enhanced_features(self, text: str) -> Dict:
        """
        Extract comprehensive features from Korean text
        
        Args:
            text: Input Korean text
            
        Returns:
            Dictionary containing various extracted features
        """
        features = {
            'nouns': [],
            'keywords': [],
            'phrases': [],
            'categories': [],
            'entities': [],
            'semantic_topics': []
        }
        
        if not text or not text.strip():
            return features
        
        # Basic morphological analysis
        if self.kiwi:
            try:
                result = self.kiwi.analyze(text)
                if result and result[0]:
                    tokens = result[0][0]
                    
                    # Extract nouns
                    nouns = []
                    current_compound = []
                    
                    for token in tokens:
                        if token.tag.startswith('N'):  # Nouns
                            if len(token.form) >= 2 and token.form not in self.stop_words:
                                nouns.append(token.form)
                                current_compound.append(token.form)
                        else:
                            # Complete compound noun if exists
                            if len(current_compound) >= 2:
                                compound = ''.join(current_compound)
                                if len(compound) >= 3:
                                    nouns.append(compound)
                            current_compound = []
                    
                    # Final compound check
                    if len(current_compound) >= 2:
                        compound = ''.join(current_compound)
                        if len(compound) >= 3:
                            nouns.append(compound)
                    
                    features['nouns'] = list(dict.fromkeys(nouns))  # Remove duplicates
                    
                    # Extract key phrases (noun combinations)
                    phrases = []
                    for i in range(len(nouns) - 1):
                        phrase = f"{nouns[i]} {nouns[i+1]}"
                        phrases.append(phrase)
                    features['phrases'] = phrases
                    
            except Exception as e:
                logger.error(f"Morphological analysis error: {e}")
        
        # Fallback noun extraction
        if not features['nouns']:
            features['nouns'] = self._extract_nouns_fallback(text)
        
        # Extract keywords using TF-IDF and frequency
        features['keywords'] = self._extract_keywords(text, features['nouns'])
        
        # Categorize content
        features['categories'] = self._categorize_text(text, features['nouns'])
        
        # Extract entities (simple pattern matching)
        features['entities'] = self._extract_entities(text)
        
        return features
    
    def _extract_nouns_fallback(self, text: str) -> List[str]:
        """Fallback noun extraction using regex patterns"""
        import re
        
        # Korean character pattern
        words = re.findall(r'[가-힣]+', text)
        nouns = []
        
        for word in words:
            if (len(word) >= 2 and 
                word not in self.stop_words and
                not re.match(r'^[ㄱ-ㅎㅏ-ㅣ]+$', word)):  # Not just consonants/vowels
                nouns.append(word)
        
        return list(dict.fromkeys(nouns))  # Remove duplicates
    
    def _extract_keywords(self, text: str, nouns: List[str]) -> List[str]:
        """Extract keywords using frequency and importance"""
        # Combine nouns with text words
        all_words = nouns.copy()
        
        # Add high-frequency words from text
        word_counts = Counter(text.split())
        frequent_words = [word for word, count in word_counts.most_common(10) 
                         if len(word) >= 2 and word not in self.stop_words]
        
        all_words.extend(frequent_words)
        
        # Remove duplicates and sort by length (longer words are often more specific)
        keywords = list(dict.fromkeys(all_words))
        keywords.sort(key=len, reverse=True)
        
        return keywords[:10]  # Return top 10 keywords
    
    def _categorize_text(self, text: str, nouns: List[str]) -> List[str]:
        """Categorize text based on content patterns"""
        text_lower = text.lower()
        all_terms = set(nouns + text.split())
        
        categories = []
        
        for category, patterns in self.category_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            
            # Additional scoring for noun matches
            noun_matches = sum(1 for noun in nouns if any(pattern in noun.lower() for pattern in patterns))
            score += noun_matches * 2
            
            if score >= 1:  # At least one match
                categories.append((category, score))
        
        # Sort by score and return top categories
        categories.sort(key=lambda x: x[1], reverse=True)
        return [cat[0] for cat in categories[:3]]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities using simple patterns"""
        import re
        
        entities = []
        
        # Location patterns
        location_patterns = [
            r'[가-힣]+시\s*[가-힣]+구',  # 서울시 강남구
            r'[가-힣]+구\s*[가-힣]+동',  # 강남구 역삼동
            r'[가-힣]+역',              # 강남역
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        return list(dict.fromkeys(entities))
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two Korean texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Try sentence transformer first
        if self.sentence_model:
            try:
                # Check cache first
                emb1 = self.cache_manager.get_embedding(text1)
                if emb1 is None:
                    emb1 = self.sentence_model.encode([text1])[0]
                    self.cache_manager.save_embedding(text1, emb1)
                
                emb2 = self.cache_manager.get_embedding(text2)
                if emb2 is None:
                    emb2 = self.sentence_model.encode([text2])[0]
                    self.cache_manager.save_embedding(text2, emb2)
                
                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                return float(similarity)
                
            except Exception as e:
                logger.error(f"Sentence transformer similarity error: {e}")
        
        # Fallback to TF-IDF similarity
        if self.tfidf_vectorizer and SKLEARN_AVAILABLE:
            try:
                vectors = self.tfidf_vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                return float(similarity)
            except Exception as e:
                logger.error(f"TF-IDF similarity error: {e}")
        
        # Final fallback: Jaccard similarity on words
        return self._jaccard_similarity(text1, text2)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def expand_query(self, query: str, max_expansions: int = 5) -> List[str]:
        """
        Expand query with synonyms and related terms
        
        Args:
            query: Original query
            max_expansions: Maximum number of expanded queries
            
        Returns:
            List of expanded queries including the original
        """
        expanded_queries = [query]
        
        # Extract features from original query
        features = self.extract_enhanced_features(query)
        keywords = features['keywords']
        
        # Generate expanded queries using synonyms
        for keyword in keywords[:3]:  # Top 3 keywords
            if keyword.lower() in self.synonym_dict:
                synonyms = self.synonym_dict[keyword.lower()]
                for synonym in synonyms[:2]:  # Top 2 synonyms
                    expanded_query = query.replace(keyword, synonym)
                    if expanded_query != query and expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
        
        # Add category-based expansions
        if features['categories']:
            main_category = features['categories'][0]
            if main_category in self.category_patterns:
                category_terms = self.category_patterns[main_category][:3]
                for term in category_terms:
                    if term not in query.lower():
                        expanded_query = f"{query} {term}"
                        if expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)
        
        return expanded_queries[:max_expansions]
    
    def calculate_relevance_score(self, 
                                cafe_text: str, 
                                search_keywords: List[str],
                                search_features: Dict) -> Tuple[float, Dict]:
        """
        Calculate enhanced relevance score using multiple factors
        
        Args:
            cafe_text: Text content about the cafe
            search_keywords: Original search keywords
            search_features: Features extracted from search query
            
        Returns:
            Tuple of (relevance_score, details_dict)
        """
        score_details = {
            'keyword_match': 0.0,
            'semantic_similarity': 0.0,
            'category_match': 0.0,
            'feature_match': 0.0,
            'total_score': 0.0
        }
        
        # Extract features from cafe text
        cafe_features = self.extract_enhanced_features(cafe_text)
        
        # 1. Keyword matching score (40% weight)
        keyword_score = 0.0
        matching_keywords = []
        
        for keyword in search_keywords:
            if keyword.lower() in cafe_text.lower():
                keyword_score += 20  # Base score for exact match
                matching_keywords.append(keyword)
                
                # Bonus for keyword in important positions
                if keyword.lower() in cafe_text[:100].lower():  # Early in text
                    keyword_score += 10
        
        score_details['keyword_match'] = min(keyword_score, 100)
        
        # 2. Semantic similarity score (30% weight)
        semantic_score = 0.0
        if search_keywords:
            search_text = ' '.join(search_keywords)
            semantic_score = self.calculate_semantic_similarity(search_text, cafe_text) * 100
        
        score_details['semantic_similarity'] = semantic_score
        
        # 3. Category matching score (20% weight)
        category_score = 0.0
        search_categories = set(search_features.get('categories', []))
        cafe_categories = set(cafe_features.get('categories', []))
        
        if search_categories and cafe_categories:
            category_overlap = len(search_categories.intersection(cafe_categories))
            category_score = (category_overlap / len(search_categories)) * 100
        
        score_details['category_match'] = category_score
        
        # 4. Feature matching score (10% weight)
        feature_score = 0.0
        search_nouns = set(search_features.get('nouns', []))
        cafe_nouns = set(cafe_features.get('nouns', []))
        
        if search_nouns and cafe_nouns:
            noun_overlap = len(search_nouns.intersection(cafe_nouns))
            feature_score = (noun_overlap / len(search_nouns)) * 100
        
        score_details['feature_match'] = feature_score
        
        # Calculate weighted total score
        total_score = (
            score_details['keyword_match'] * 0.4 +
            score_details['semantic_similarity'] * 0.3 +
            score_details['category_match'] * 0.2 +
            score_details['feature_match'] * 0.1
        )
        
        score_details['total_score'] = total_score
        score_details['matching_keywords'] = matching_keywords
        
        return total_score, score_details
    
    def cluster_cafes(self, cafe_texts: List[str], n_clusters: int = 5) -> List[int]:
        """
        Cluster cafes based on content similarity
        
        Args:
            cafe_texts: List of cafe text descriptions
            n_clusters: Number of clusters
            
        Returns:
            List of cluster labels for each cafe
        """
        if not cafe_texts or not SKLEARN_AVAILABLE:
            return [0] * len(cafe_texts)
        
        try:
            if self.sentence_model:
                # Use semantic embeddings for clustering
                embeddings = []
                for text in cafe_texts:
                    emb = self.cache_manager.get_embedding(text)
                    if emb is None:
                        emb = self.sentence_model.encode([text])[0]
                        self.cache_manager.save_embedding(text, emb)
                    embeddings.append(emb)
                
                embeddings = np.array(embeddings)
            else:
                # Use TF-IDF vectors
                vectors = self.tfidf_vectorizer.fit_transform(cafe_texts)
                embeddings = vectors.toarray()
            
            # Perform K-means clustering
            n_clusters = min(n_clusters, len(cafe_texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            return cluster_labels.tolist()
            
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return [0] * len(cafe_texts)


# Singleton instance
_enhanced_nlp = None

def get_enhanced_nlp(use_gpu: bool = False) -> EnhancedKoreanNLP:
    """Get or create the singleton enhanced NLP processor instance"""
    global _enhanced_nlp
    if _enhanced_nlp is None:
        _enhanced_nlp = EnhancedKoreanNLP(use_gpu=use_gpu)
    return _enhanced_nlp