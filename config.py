#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for Naver Cafe Search System
================================================
Central configuration and logging setup
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

# --- Default Configuration ---
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
    "nlp": {
        "use_kiwipiepy": True,
        "extract_compounds": True,
        "min_noun_length": 2
    },
    "server": {
        "host": "127.0.0.1",
        "port": 5000,
        "debug": False
    }
}

# --- Negative Keywords with Weights ---
NEGATIVE_KEYWORDS = {
    '중고': -40, '판매': -20, '구매': -20, '삽니다': -30, 
    '팝니다': -30, '분양': -50, '구인': -30, '구직': -30, 
    '알바': -30, '거래': -25, '양도': -25, '매매': -25
}

# --- Category Keywords ---
CATEGORY_KEYWORDS = {
    "제조/생산": ['제조', '생산', '공장', 'oem', 'odm', '봉제', '프로모션', '가공', '원단', '제작'],
    "도매/유통": ['도매', '사입', '시장', '동대문', '남대문', '유통', '도매가', '떨이', '공급'],
    "창업/비즈니스": ['창업', '비즈니스', '사업', '스타트업', '프랜차이즈', '가맹', '수익', '투자'],
    "교육/학습": ['교육', '학습', '스터디', '강의', '학원', '자격증', '시험', '합격', '과외'],
    "커뮤니티": ['모임', '동호회', '클럽', '친목', '정보공유', '커뮤니티', '회원', '가입'],
    "취미/문화": ['취미', '동호회', '문화', '예술', '음악', '사진', '영화', '공연', '전시'],
    "IT/기술": ['it', '개발', '프로그래밍', '기술', '소프트웨어', '앱', '코딩', '데이터', '개발자'],
    "기타": []
}

def load_config() -> Dict[str, Any]:
    """
    Load configuration from file or use defaults
    
    Returns:
        Dict containing configuration settings
    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # Merge with defaults
                config = DEFAULT_CONFIG.copy()
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in config:
                        config[key].update(value)
                    else:
                        config[key] = value
                return config
        except Exception as e:
            logging.warning(f"Failed to load config file: {e}. Using defaults.")
    
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Failed to save config: {e}")

def setup_logging():
    """
    Setup structured logging for the application
    """
    # Create logs directory
    LOG_DIR.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_DIR / 'app.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

# Load configuration on module import
CONFIG = load_config()

# Setup logging
setup_logging()

# Create logger for this module
logger = logging.getLogger(__name__)