#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Naver Cafe Search System - AI-Powered Application
==========================================================
Version: 4.0.0
Features:
- Korean Sentence Transformers for semantic search
- Local machine learning for relevance scoring
- Hidden gems discovery algorithm
- Diversity-aware result ranking
- Category-based search templates
- Enhanced UI with responsive design
"""

import asyncio
import json
import os
import sys
import time
import webbrowser
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from quart import Quart, request, jsonify, Response, send_file
from quart_cors import cors
import pandas as pd

from config import CONFIG, setup_logging, logger
from enhanced_cafe_searcher import EnhancedAICafeSearcher, AISearchSettings
from models import FilteredCafe

# PyInstaller resource path helper
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Initialize Quart app
app = Quart(__name__)
app.config['PROVIDE_AUTOMATIC_OPTIONS'] = True
app = cors(app, allow_origin="*")

# Global configuration paths
STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

# Global searcher instance and search sessions
searcher = None
search_sessions: Dict[str, dict] = {}

def open_browser():
    """Open browser after server starts"""
    time.sleep(1.5)
    webbrowser.open_new(f"http://{CONFIG['server']['host']}:{CONFIG['server']['port']}")

async def initialize_ai_searcher():
    """Initialize the AI-enhanced searcher with API credentials"""
    global searcher
    
    try:
        client_id = CONFIG['naver_api']['client_id']
        client_secret = CONFIG['naver_api']['client_secret']
        
        if not client_id or not client_secret:
            logger.error("Naver API credentials not found in config")
            return False
        
        searcher = EnhancedAICafeSearcher(client_id, client_secret)
        logger.info("AI-enhanced searcher initialized successfully")
        
        # Test NLP components
        try:
            from enhanced_nlp_processor import get_enhanced_nlp
            nlp = get_enhanced_nlp()
            test_features = nlp.extract_enhanced_features("테스트 프로그래밍 스터디")
            logger.info(f"NLP test successful: {test_features}")
        except Exception as e:
            logger.warning(f"Advanced NLP features may be limited: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize AI searcher: {e}")
        return False

@app.route('/')
async def index():
    """Serve the enhanced main page"""
    return await send_file(str(INDEX_HTML_PATH))

@app.route('/api/status')
async def status():
    """Enhanced API status endpoint with AI feature information"""
    try:
        # Check sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            sentence_transformers_available = True
            sentence_transformers_version = "Available"
        except:
            sentence_transformers_available = False
            sentence_transformers_version = "Not installed"
        
        # Check kiwipiepy
        try:
            from kiwipiepy import Kiwi
            kiwi_available = True
            kiwi_version = "Available"
        except:
            kiwi_available = False
            kiwi_version = "Not installed"
        
        # Check scikit-learn
        try:
            import sklearn
            sklearn_available = True
            sklearn_version = sklearn.__version__
        except:
            sklearn_available = False
            sklearn_version = "Not installed"
        
        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except:
            pass
        
        return jsonify({
            'status': 'ready' if searcher else 'not_initialized',
            'version': '4.0.0',
            'features': {
                'korean_nlp': {
                    'engine': 'kiwipiepy + sentence-transformers',
                    'available': kiwi_available,
                    'version': kiwi_version
                },
                'semantic_search': {
                    'available': sentence_transformers_available,
                    'version': sentence_transformers_version,
                    'model': 'jhgan/ko-sroberta-multitask'
                },
                'machine_learning': {
                    'available': sklearn_available,
                    'version': sklearn_version,
                    'algorithms': ['TF-IDF', 'K-Means', 'Cosine Similarity']
                },
                'ai_algorithms': {
                    'hidden_gems_discovery': True,
                    'diversity_ranking': True,
                    'query_expansion': True,
                    'semantic_similarity': sentence_transformers_available
                },
                'hardware': {
                    'gpu_available': gpu_available,
                    'recommended_ram': '8GB+',
                    'optimized_for': 'CPU inference'
                }
            },
            'config': {
                'min_score': CONFIG['search']['min_relevance_score'],
                'concurrent_requests': CONFIG['search']['concurrent_requests'],
                'cache_ttl': CONFIG['search']['cache_ttl'],
                'ai_enabled': sentence_transformers_available or sklearn_available
            }
        })
    
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/start-enhanced-search', methods=['POST'])
async def start_enhanced_search():
    """Start a new AI-enhanced search and return search ID"""
    if not searcher:
        return jsonify({'error': 'AI searcher not initialized'}), 500
    
    try:
        data = await request.get_json()
        
        # Validate input
        keywords = data.get('keywords', [])
        if not keywords or not isinstance(keywords, list):
            return jsonify({'error': 'Invalid keywords format'}), 400
        
        exclude_keywords = data.get('excludeKeywords', [])
        min_score = data.get('minScore', CONFIG['search']['min_relevance_score'])
        ai_settings_data = data.get('aiSettings', {})
        
        # Create AI settings
        ai_settings = AISearchSettings(
            semantic_search=ai_settings_data.get('semanticSearch', True),
            hidden_gems=ai_settings_data.get('hiddenGems', True),
            diversity_mode=ai_settings_data.get('diversityMode', True)
        )
        
        # Generate unique search ID
        search_id = str(uuid.uuid4())
        
        # Initialize search session
        search_sessions[search_id] = {
            'status': 'starting',
            'progress_queue': asyncio.Queue(),
            'results': None,
            'error': None,
            'ai_settings': ai_settings
        }
        
        logger.info(f"Starting AI search {search_id} with keywords: {keywords}, AI settings: {ai_settings}")
        
        # Start search in background
        asyncio.create_task(
            run_ai_search_background(
                search_id,
                keywords,
                exclude_keywords,
                min_score,
                ai_settings
            )
        )
        
        return jsonify({'search_id': search_id})
        
    except Exception as e:
        logger.error(f"Start enhanced search error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream-progress/<search_id>')
async def stream_progress(search_id):
    """Stream search progress via SSE (same as before but with AI context)"""
    if search_id not in search_sessions:
        return Response("Search ID not found", status=404)
    
    session = search_sessions[search_id]
    
    async def generate():
        """Generate SSE stream"""
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        session['progress_queue'].get(), 
                        timeout=30.0
                    )
                    
                    if 'results' in item:
                        # Final results
                        results_data = [cafe.to_dict() for cafe in item['results']]
                        yield f"event: complete\ndata: {json.dumps(results_data)}\n\n"
                        break
                    elif 'error' in item:
                        # Error occurred
                        yield f"event: error\ndata: {json.dumps(item)}\n\n"
                        break
                    else:
                        # Progress update
                        yield f"data: {json.dumps(item)}\n\n"
                        
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield ": heartbeat\n\n"
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                    break
        finally:
            # Clean up session
            if search_id in search_sessions:
                del search_sessions[search_id]
    
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Transfer-Encoding': 'chunked',
        'X-Accel-Buffering': 'no'
    }
    
    return Response(generate(), headers=headers)

async def run_ai_search_background(search_id, keywords, exclude_keywords, min_score, ai_settings):
    """Run AI-enhanced search in background and update progress"""
    session = search_sessions[search_id]
    
    try:
        session['status'] = 'running'
        
        # Define progress callback
        def progress_callback(progress_data):
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    session['progress_queue'].put_nowait,
                    progress_data
                )
            except Exception as e:
                logger.error(f"Error in AI progress callback: {e}")
        
        # Run AI-enhanced search
        async with searcher:
            results = await searcher.search_cafes_with_ai(
                keywords,
                exclude_keywords,
                min_score,
                ai_settings,
                progress_callback
            )
            
            # Send final results
            session['progress_queue'].put_nowait({'results': results})
            session['status'] = 'completed'
            session['results'] = results
            
    except Exception as e:
        logger.error(f"Background AI search error: {e}", exc_info=True)
        session['status'] = 'error'
        session['error'] = str(e)
        session['progress_queue'].put_nowait({'error': str(e)})

@app.route('/api/analyze-query', methods=['POST'])
async def analyze_query():
    """Analyze search query using enhanced NLP"""
    try:
        data = await request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        from enhanced_nlp_processor import get_enhanced_nlp
        nlp = get_enhanced_nlp()
        
        # Extract comprehensive features
        features = nlp.extract_enhanced_features(query)
        
        # Generate expanded queries
        expanded_queries = nlp.expand_query(query, max_expansions=5)
        
        # Calculate estimated search complexity
        complexity = "simple"
        if len(features['nouns']) > 3:
            complexity = "complex"
        elif len(features['categories']) > 1:
            complexity = "medium"
        
        return jsonify({
            'original_query': query,
            'features': features,
            'expanded_queries': expanded_queries,
            'complexity': complexity,
            'recommendations': {
                'semantic_search': len(features['categories']) > 0,
                'hidden_gems': complexity in ['medium', 'complex'],
                'diversity_mode': len(features['nouns']) > 2
            }
        })
        
    except Exception as e:
        logger.error(f"Query analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-templates')
async def get_templates():
    """Get category-based search templates"""
    templates = {
        "study": [
            {
                "title": "💻 프로그래밍 스터디",
                "description": "개발자들이 모여 기술을 공유하고 함께 성장하는 카페",
                "keywords": ["프로그래밍", "개발", "코딩"],
                "category": "개발/IT"
            },
            {
                "title": "🌍 영어 학습",
                "description": "영어 실력 향상을 위한 스터디 그룹",
                "keywords": ["영어", "토익", "토플"],
                "category": "스터디/교육"
            },
            {
                "title": "📋 공무원 시험",
                "description": "공무원 시험 준비생들의 정보 공유",
                "keywords": ["공무원", "시험", "공시"],
                "category": "스터디/교육"
            }
        ],
        "business": [
            {
                "title": "🚀 창업 커뮤니티",
                "description": "창업가들이 경험을 나누고 네트워킹하는 공간",
                "keywords": ["창업", "스타트업", "사업"],
                "category": "창업/비즈니스"
            },
            {
                "title": "💼 프리랜서",
                "description": "프리랜서들의 정보 공유와 프로젝트 협업",
                "keywords": ["프리랜서", "부업", "사이드"],
                "category": "창업/비즈니스"
            }
        ],
        "hobby": [
            {
                "title": "📸 사진 동호회",
                "description": "사진 촬영 기법과 작품을 공유하는 모임",
                "keywords": ["사진", "포토", "카메라"],
                "category": "취미/문화"
            },
            {
                "title": "🍳 요리",
                "description": "요리 레시피와 맛집 정보를 나누는 카페",
                "keywords": ["요리", "레시피", "맛집"],
                "category": "취미/문화"
            }
        ]
    }
    
    return jsonify(templates)

@app.route('/api/export/csv', methods=['POST'])
async def export_csv():
    """Export search results as CSV with enhanced fields"""
    try:
        data = await request.get_json()
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'No results to export'}), 400
        
        # Convert to DataFrame with enhanced columns
        df = pd.DataFrame(results)
        
        # Enhanced column order
        columns_order = [
            'name', 'url', 'relevance_score', 'confidence', 
            'category', 'matching_keywords', 'negative_keywords_found', 
            'description', 'analysis_notes', 'search_keyword_origin',
            'search_timestamp'
        ]
        
        # Only include columns that exist
        df = df[[col for col in columns_order if col in df.columns]]
        
        # Convert lists to comma-separated strings
        for col in ['matching_keywords', 'negative_keywords_found']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else ''
                )
        
        # Generate filename with timestamp and AI indicator
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ai_cafe_search_{timestamp}.csv"
        filepath = Path(resource_path(filename))
        
        # Save to CSV with UTF-8 BOM for Korean support
        await asyncio.to_thread(
            df.to_csv, 
            filepath, 
            index=False, 
            encoding='utf-8-sig'
        )
        
        return await send_file(str(filepath), as_attachment=True)
        
    except Exception as e:
        logger.error(f"Enhanced CSV export error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/json', methods=['POST'])
async def export_json():
    """Export search results as JSON with AI metadata"""
    try:
        data = await request.get_json()
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'No results to export'}), 400
        
        # Enhanced metadata
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_results': len(results),
                'version': '4.0.0',
                'ai_features_used': [
                    'semantic_search',
                    'hidden_gems_discovery',
                    'diversity_ranking',
                    'enhanced_nlp'
                ],
                'search_stats': {
                    'avg_relevance': sum(r['relevance_score'] for r in results) / len(results) if results else 0,
                    'high_confidence_count': len([r for r in results if r.get('confidence') in ['높음', '매우 높음']]),
                    'categories_found': list(set(r.get('category', '기타') for r in results))
                }
            },
            'results': results
        }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ai_cafe_search_{timestamp}.json"
        filepath = Path(resource_path(filename))
        
        # Save to JSON with Korean support
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return await send_file(str(filepath), as_attachment=True)
        
    except Exception as e:
        logger.error(f"Enhanced JSON export error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if searcher is working
        if not searcher:
            return jsonify({'status': 'unhealthy', 'reason': 'searcher not initialized'}), 503
        
        # Check NLP components
        nlp_status = "healthy"
        try:
            from enhanced_nlp_processor import get_enhanced_nlp
            nlp = get_enhanced_nlp()
            test_result = nlp.extract_enhanced_features("테스트")
            if not test_result:
                nlp_status = "degraded"
        except Exception:
            nlp_status = "failed"
        
        # Check memory usage (basic check)
        import psutil
        memory_percent = psutil.virtual_memory().percent
        
        status = "healthy"
        if memory_percent > 90:
            status = "degraded"
        elif nlp_status == "failed":
            status = "degraded"
        
        return jsonify({
            'status': status,
            'components': {
                'searcher': 'healthy',
                'nlp': nlp_status,
                'memory_usage': f"{memory_percent:.1f}%"
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

# Error handlers
@app.errorhandler(404)
async def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found', 'version': '4.0.0'}), 404

@app.errorhandler(500)
async def server_error(error):
    """Handle 500 errors"""
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error', 'version': '4.0.0'}), 500

# Main execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-Enhanced Naver Cafe Search System')
    parser.add_argument('--no-browser', action='store_true', 
                       help='Do not open browser automatically')
    parser.add_argument('--port', type=int, default=CONFIG['server']['port'], 
                       help='Port to run the server on')
    parser.add_argument('--host', default=CONFIG['server']['host'], 
                       help='Host to bind the server to')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration for AI models')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Initialize AI searcher
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    if loop.run_until_complete(initialize_ai_searcher()):
        logger.info(f"Starting AI-Enhanced Naver Cafe Search System v4.0 on {args.host}:{args.port}")
        
        if args.gpu:
            logger.info("GPU acceleration enabled for AI models")
        
        # Open browser if not disabled
        if not args.no_browser:
            threading.Thread(target=open_browser, daemon=True).start()
        
        # Run the server
        app.run(
            debug=args.debug or CONFIG['server']['debug'],
            host=args.host,
            port=args.port,
            use_reloader=False
        )
    else:
        logger.error("Failed to initialize AI searcher. Check your API credentials and dependencies.")
        print("\n" + "="*60)
        print("ERROR: Failed to initialize AI-enhanced searcher")
        print("Please check:")
        print("1. Naver API credentials in config.json")
        print("2. Required dependencies are installed:")
        print("   pip install sentence-transformers scikit-learn kiwipiepy")
        print("3. Sufficient memory (8GB+ recommended)")
        print("="*60 + "\n")
        input("Press Enter to exit...")