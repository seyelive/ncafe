#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Configuration module for AI-Powered Naver Cafe Search System
====================================================================
Central configuration and logging setup with AI features
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

# --- Paths ---
BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"
LOG_DIR = BASE_DIR / "logs"

# --- Enhanced Default Configuration ---
DEFAULT_CONFIG = {
    "naver_api": {
        "client_id": os.environ.get("NAVER_CLIENT_ID", ""),
        "client_secret": os.environ.get("NAVER_CLIENT_SECRET", "")
    },
    "search": {
        "max_results_per_keyword": 500,
        "concurrent_requests": 5,
        "request_timeout": 30,
        "cache_ttl": 300,
        "min_relevance_score": 40
    },
    "ai": {
        "semantic_search": True,
        "hidden_gems": True,
        "diversity_mode": True,
        "use_gpu": False,
        "max_semantic_candidates": 1000,
        "diversity_lambda": 0.7,
        "novelty_boost": 1.2,
        "cache_embeddings": True,
        "batch_size": 10
    },
    "nlp": {
        "use_kiwipiepy": True,
        "extract_compounds": True,
        "min_noun_length": 2,
        "sentence_transformer_model": "jhgan/ko-sroberta-multitask",
        "fallback_to_tfidf": True,
        "cache_analysis_results": True
    },
    "server": {
        "host": "127.0.0.1",
        "port": 5000,
        "debug": False,
        "enable_health_check": True,
        "cors_origins": ["*"]
    },
    "performance": {
        "enable_query_cache": True,
        "enable_result_cache": True,
        "max_cache_size": 1000,
        "cache_cleanup_interval": 3600,
        "enable_profiling": False
    },
    "ui": {
        "enable_templates": True,
        "default_view": "card",
        "results_per_page": 50,
        "enable_export": True,
        "theme": "modern"
    }
}

# --- Enhanced Negative Keywords with AI-optimized weights ---
NEGATIVE_KEYWORDS = {
    # Commercial/Sales (High penalty)
    '중고': -50, '판매': -30, '구매': -30, '삽니다': -40, 
    '팝니다': -40, '분양': -60, '거래': -35, '양도': -35, '매매': -35,
    
    # Job-related (Medium penalty - might be relevant for some searches)
    '구인': -25, '구직': -25, '알바': -25, '채용': -20,
    
    # Spam indicators (High penalty)
    '광고': -45, '홍보': -40, '마케팅': -30, '프로모션': -35,
    
    # Low quality indicators (Medium penalty)
    '급매': -30, '떨이': -25, '싸게': -20, '덤핑': -40
}

# --- Enhanced Category Keywords with AI-optimized patterns ---
CATEGORY_KEYWORDS = {
    "개발/IT": [
        '개발', '프로그래밍', '코딩', 'IT', 'it', '개발자', '프로그래머',
        'python', 'java', 'javascript', 'react', 'vue', 'angular',
        '백엔드', '프론트엔드', '풀스택', 'ai', '머신러닝', '데이터'
    ],
    "스터디/교육": [
        '스터디', '학습', '공부', '교육', '강의', '학원', '과외',
        '자격증', '시험', '토익', '토플', '공무원', '공시', '고시',
        '취업', '면접', '자소서', '포트폴리오'
    ],
    "창업/비즈니스": [
        '창업', '스타트업', '사업', '비즈니스', '기업', '회사',
        '마케팅', '광고', '브랜딩', '세일즈', '영업', '투자', '재테크',
        '주식', '펀드', '부동산', '경영', '전략', '컨설팅'
    ],
    "취미/문화": [
        '취미', '동호회', '문화', '예술', '음악', '사진', '영화', '드라마',
        '독서', '책', '서평', '요리', '레시피', '맛집', '카페', '여행',
        '캠핑', '등산', '운동', '헬스', '요가', '필라테스'
    ],
    "지역/모임": [
        '지역', '모임', '동호회', '클럽', '커뮤니티', '친목', '소모임',
        '번개', '정모', '오프라인', '만남', '네트워킹', '교류',
        '강남', '홍대', '신촌', '분당', '일산', '수원', '인천'
    ],
    "육아/가족": [
        '육아', '맘카페', '육아맘', '워킹맘', '전업맘', '아이', '유아',
        '신생아', '임신', '출산', '육아용품', '유모차', '분유',
        '가족', '부모', '엄마', '아빠', '형제', '자매'
    ],
    "여행/관광": [
        '여행', '관광', '여행지', '국내여행', '해외여행', '자유여행',
        '패키지', '펜션', '민박', '게스트하우스', '호텔', '리조트',
        '캠핑', '백패킹', '트레킹', '힐링', '휴양', '관광지'
    ],
    "라이프스타일": [
        '라이프스타일', '일상', '소확행', '힐링', '웰빙', '건강',
        '다이어트', '운동', '뷰티', '패션', '인테리어', '홈카페',
        '미니멀', '정리정돈', '살림', '자취', '독립', '셀프'
    ]
}

# --- AI Model Configurations ---
AI_MODEL_CONFIGS = {
    "sentence_transformers": {
        "models": {
            "korean_general": "jhgan/ko-sroberta-multitask",
            "korean_sentiment": "beomi/KcELECTRA-base-v2022",
            "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"
        },
        "default_model": "korean_general",
        "device_preference": ["cuda", "mps", "cpu"],
        "max_seq_length": 512,
        "batch_size": 32
    },
    "clustering": {
        "default_clusters": 5,
        "max_clusters": 10,
        "min_cluster_size": 3,
        "algorithm": "kmeans"
    },
    "similarity": {
        "threshold_high": 0.8,
        "threshold_medium": 0.6,
        "threshold_low": 0.4,
        "method": "cosine"
    }
}

# --- Search Templates Configuration ---
SEARCH_TEMPLATES = {
    "fashion": {
        "clothing_manufacturing": {
            "title": "👕 패션 의류 제조",
            "keywords": ["의류제조", "봉제공장", "OEM", "ODM", "패션제조"],
            "description": "의류 제조업체, OEM/ODM, 봉제공장 운영자들의 정보 공유",
            "ai_boost": 1.3
        },
        "smart_factory": {
            "title": "🏭 스마트팩토리",
            "keywords": ["스마트팩토리", "자동화", "제조혁신", "공장관리"],
            "description": "스마트공장 구축, 자동화 시스템, 제조업 혁신 커뮤니티",
            "ai_boost": 1.4
        },
        "sewing_tech": {
            "title": "✂️ 봉제/재봉 기술",
            "keywords": ["봉제", "재봉", "의류생산", "패턴", "소재"],
            "description": "봉제 기술, 재봉기 정보, 의류 생산 기법 공유",
            "ai_boost": 1.2
        },
        "fashion_brand": {
            "title": "👔 패션 브랜드 창업",
            "keywords": ["패션브랜드", "의류창업", "브랜딩", "패션사업"],
            "description": "패션 브랜드 런칭, 의류 브랜딩, 패션 창업 정보",
            "ai_boost": 1.3
        }
    },
    "beauty": {
        "cosmetics_wholesale": {
            "title": "💄 화장품 도소매",
            "keywords": ["화장품도매", "뷰티도매", "화장품유통", "B2B"],
            "description": "화장품 도매업, 뷰티 유통업체, B2B 거래 정보",
            "ai_boost": 1.3
        },
        "k_beauty": {
            "title": "🌟 K-뷰티",
            "keywords": ["K뷰티", "한국화장품", "뷰티", "스킨케어"],
            "description": "한국 화장품, K-뷰티 트렌드, 한국 뷰티 브랜드 정보",
            "ai_boost": 1.4
        },
        "skincare_haircare": {
            "title": "🧴 스킨케어/헤어케어",
            "keywords": ["스킨케어", "헤어케어", "화장품", "뷰티제품"],
            "description": "스킨케어, 헤어케어 제품 정보 및 업계 동향",
            "ai_boost": 1.2
        },
        "beauty_shop": {
            "title": "🏪 뷰티샵 운영",
            "keywords": ["뷰티샵", "화장품매장", "뷰티소매", "매장운영"],
            "description": "화장품 매장 운영, 뷰티샵 창업, 소매업 정보",
            "ai_boost": 1.2
        }
    },
    "export": {
        "global_trade": {
            "title": "🌍 글로벌 수출입",
            "keywords": ["수출입", "글로벌무역", "해외수출", "국제무역"],
            "description": "해외 수출입, 글로벌 무역, 국제 비즈니스 정보",
            "ai_boost": 1.4
        },
        "buyer_networking": {
            "title": "🏢 바이어 네트워킹",
            "keywords": ["바이어", "B2B", "해외바이어", "글로벌파트너"],
            "description": "해외 바이어 발굴, B2B 네트워킹, 글로벌 파트너십",
            "ai_boost": 1.5
        },
        "middle_east": {
            "title": "🇸🇦 중동 시장",
            "keywords": ["중동시장", "사우디", "UAE", "중동수출"],
            "description": "사우디, UAE 등 중동 시장 진출 및 비즈니스 정보",
            "ai_boost": 1.3
        },
        "north_america": {
            "title": "🇺🇸 북미 시장",
            "keywords": ["미국시장", "캐나다", "북미수출", "USA"],
            "description": "미국, 캐나다 시장 진출 및 북미 비즈니스 정보",
            "ai_boost": 1.3
        }
    },
    "tech": {
        "rfid_tech": {
            "title": "📡 RFID 기술",
            "keywords": ["RFID", "센서", "IoT", "무선인식", "스마트태그"],
            "description": "RFID 시스템, 센서 기술, IoT 응용 기술 정보",
            "ai_boost": 1.4
        },
        "fashion_tech": {
            "title": "🔬 패션테크",
            "keywords": ["패션테크", "스마트의류", "웨어러블", "패션기술"],
            "description": "패션과 기술의 융합, 스마트 의류, 웨어러블 기술",
            "ai_boost": 1.5
        },
        "manufacturing_rd": {
            "title": "🏭 제조업 R&D",
            "keywords": ["제조R&D", "생산기술", "공정혁신", "제조혁신"],
            "description": "제조업 연구개발, 생산기술 혁신, 공정 개선",
            "ai_boost": 1.4
        }
    },
    # 기존 템플릿도 일부 유지 (선택사항)
    "general": {
        "programming": {
            "title": "💻 프로그래밍 스터디",
            "keywords": ["프로그래밍", "개발", "코딩"],
            "description": "개발자들이 모여 기술을 공유하고 함께 성장하는 카페",
            "ai_boost": 1.2
        },
        "startup": {
            "title": "🚀 창업 커뮤니티",
            "keywords": ["창업", "스타트업", "사업"],
            "description": "창업가들이 경험을 나누고 네트워킹하는 공간",
            "ai_boost": 1.3
        }
    }
}

def load_config() -> Dict[str, Any]:
    """
    Load configuration from file or use defaults with AI enhancements
    
    Returns:
        Dict containing configuration settings
    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # Deep merge with defaults
                config = _deep_merge_config(DEFAULT_CONFIG.copy(), user_config)
                
                # Validate AI configuration
                config = _validate_ai_config(config)
                
                return config
        except Exception as e:
            logging.warning(f"Failed to load config file: {e}. Using defaults.")
    
    return DEFAULT_CONFIG.copy()

def _deep_merge_config(default: Dict, user: Dict) -> Dict:
    """Deep merge user config with defaults"""
    result = default.copy()
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_config(result[key], value)
        else:
            result[key] = value
    
    return result

def _validate_ai_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and adjust AI configuration based on system capabilities"""
    ai_config = config.get('ai', {})
    
    # Check GPU availability
    if ai_config.get('use_gpu', False):
        try:
            import torch
            if not torch.cuda.is_available():
                logging.warning("GPU requested but CUDA not available. Falling back to CPU.")
                ai_config['use_gpu'] = False
        except ImportError:
            logging.warning("PyTorch not available. Disabling GPU usage.")
            ai_config['use_gpu'] = False
    
    # Adjust batch size based on memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if memory_gb < 8:
            ai_config['batch_size'] = min(ai_config.get('batch_size', 10), 5)
            ai_config['max_semantic_candidates'] = min(ai_config.get('max_semantic_candidates', 1000), 500)
            logging.info("Limited memory detected. Reducing AI batch sizes.")
        elif memory_gb >= 16:
            ai_config['batch_size'] = max(ai_config.get('batch_size', 10), 20)
            logging.info("Ample memory detected. Increasing AI batch sizes.")
    except ImportError:
        pass
    
    config['ai'] = ai_config
    return config

def save_config(config: Dict[str, Any]):
    """Save configuration to file with pretty formatting"""
    try:
        # Remove sensitive information before saving
        config_to_save = config.copy()
        if 'naver_api' in config_to_save:
            api_config = config_to_save['naver_api'].copy()
            if api_config.get('client_secret'):
                api_config['client_secret'] = "***HIDDEN***"
            config_to_save['naver_api'] = api_config
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Failed to save config: {e}")

def setup_logging(level: str = 'INFO'):
    """
    Setup enhanced logging for the AI application
    """
    # Create logs directory
    LOG_DIR.mkdir(exist_ok=True)
    
    # Configure root logger with enhanced format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(LOG_DIR / 'app.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Create separate loggers for different components
    ai_logger = logging.getLogger('ai')
    ai_handler = logging.FileHandler(LOG_DIR / 'ai.log', encoding='utf-8')
    ai_handler.setFormatter(logging.Formatter(log_format))
    ai_logger.addHandler(ai_handler)
    ai_logger.setLevel(logging.DEBUG)
    
    performance_logger = logging.getLogger('performance')
    perf_handler = logging.FileHandler(LOG_DIR / 'performance.log', encoding='utf-8')
    perf_handler.setFormatter(logging.Formatter(log_format))
    performance_logger.addHandler(perf_handler)
    
    # Reduce noise from external libraries
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

def get_ai_model_config(model_type: str = 'sentence_transformers') -> Dict[str, Any]:
    """Get AI model configuration"""
    return AI_MODEL_CONFIGS.get(model_type, {})

def get_search_templates() -> Dict[str, Any]:
    """Get search templates configuration"""
    return SEARCH_TEMPLATES

def get_device_config() -> str:
    """Determine the best device for AI models"""
    config = load_config()
    
    if not config.get('ai', {}).get('use_gpu', False):
        return 'cpu'
    
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon
    except ImportError:
        pass
    
    return 'cpu'

# Load configuration on module import
CONFIG = load_config()

# Setup logging
setup_logging(CONFIG.get('logging', {}).get('level', 'INFO'))

# Create logger for this module
logger = logging.getLogger(__name__)

# Log configuration info
logger.info(f"Configuration loaded: AI enabled={CONFIG.get('ai', {}).get('semantic_search', False)}")
logger.info(f"Device configuration: {get_device_config()}")