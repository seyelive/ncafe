#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Naver Cafe Search System - Main Application
===================================================
Version: 3.0.0
Features:
- Kiwipiepy for superior Korean NLP
- Intelligent query generation with Naver operators
- Async architecture with aiohttp
- Real-time progress updates via SSE
- Modular design for maintainability
"""

import asyncio
import json
import os
import sys
import time
import webbrowser
import threading
from datetime import datetime
from pathlib import Path

from quart import Quart, request, jsonify, Response, send_file
from quart_cors import cors
import pandas as pd

from config import CONFIG, setup_logging, logger
from cafe_searcher import EnhancedCafeSearcher
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

# Add missing configuration for compatibility
app.config['PROVIDE_AUTOMATIC_OPTIONS'] = True

app = cors(app, allow_origin="*")

# Global configuration paths
STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

# Global searcher instance
searcher = None

async def initialize_searcher():
    """Initialize the searcher with API credentials"""
    global searcher
    
    try:
        client_id = CONFIG['naver_api']['client_id']
        client_secret = CONFIG['naver_api']['client_secret']
        
        if not client_id or not client_secret:
            logger.error("Naver API credentials not found in config")
            return False
        
        searcher = EnhancedCafeSearcher(client_id, client_secret)
        logger.info("Enhanced searcher initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize searcher: {e}")
        return False

@app.route('/')
async def index():
    """Serve the main page"""
    return await send_file(str(INDEX_HTML_PATH))

@app.route('/api/status')
async def status():
    """API status endpoint with feature information"""
    try:
        from kiwipiepy import Kiwi
        kiwi_available = True
        kiwi_version = "Available"
    except:
        kiwi_available = False
        kiwi_version = "Not installed"
    
    return jsonify({
        'status': 'ready' if searcher else 'not_initialized',
        'version': '3.0.0',
        'features': {
            'korean_nlp': {
                'engine': 'kiwipiepy',
                'available': kiwi_available,
                'version': kiwi_version
            },
            'intelligent_queries': True,
            'circuit_breaker': True,
            'caching': True,
            'async_processing': True
        },
        'config': {
            'min_score': CONFIG['search']['min_relevance_score'],
            'concurrent_requests': CONFIG['search']['concurrent_requests'],
            'cache_ttl': CONFIG['search']['cache_ttl']
        }
    })

@app.route('/api/stream-search', methods=['POST'])
async def stream_search():
    """Stream search results using Server-Sent Events"""
    if not searcher:
        return Response("Searcher not initialized", status=500)
    
    try:
        data = await request.get_json()
        
        # Validate input
        keywords = data.get('keywords', [])
        if not keywords or not isinstance(keywords, list):
            return Response("Invalid keywords format", status=400)
        
        exclude_keywords = data.get('excludeKeywords', [])
        min_score = data.get('minScore', CONFIG['search']['min_relevance_score'])
        
        logger.info(f"Starting search with keywords: {keywords}")
        
        async def generate():
            """Generate SSE stream"""
            queue = asyncio.Queue()
            
            async def progress_callback(progress_data):
                """Callback to queue progress updates"""
                await queue.put(progress_data)
            
            # Start search in background task
            search_task = asyncio.create_task(
                search_with_context(
                    keywords,
                    exclude_keywords,
                    min_score,
                    progress_callback
                )
            )
            
            # Stream progress updates
            while True:
                try:
                    # Wait for progress update with timeout
                    item = await asyncio.wait_for(queue.get(), timeout=30.0)
                    
                    if 'results' in item:
                        # Final results
                        results_data = [cafe.to_dict() for cafe in item['results']]
                        yield f"event: complete\ndata: {json.dumps(results_data)}\n\n"
                        break
                    else:
                        # Progress update
                        yield f"data: {json.dumps(item)}\n\n"
                        
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield ": heartbeat\n\n"
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                    break
            
            # Ensure task completes
            try:
                await search_task
            except Exception as e:
                logger.error(f"Search task error: {e}")
        
        headers = {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Transfer-Encoding': 'chunked',
            'X-Accel-Buffering': 'no'
        }
        
        return Response(generate(), headers=headers)
        
    except Exception as e:
        logger.error(f"Stream search error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

async def search_with_context(keywords, exclude_keywords, min_score, progress_callback):
    """Perform search with proper context management"""
    async with searcher:
        results = await searcher.search_cafes(
            keywords,
            exclude_keywords,
            min_score,
            progress_callback
        )
        # Send final results through callback
        await progress_callback({'results': results})

@app.route('/api/export/csv', methods=['POST'])
async def export_csv():
    """Export search results as CSV"""
    try:
        data = await request.get_json()
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'No results to export'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        columns_order = [
            'name', 'url', 'relevance_score', 'confidence', 
            'category', 'matching_keywords', 'negative_keywords_found', 
            'description', 'search_keyword_origin', 'analysis_notes',
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
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"naver_cafe_search_{timestamp}.csv"
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
        logger.error(f"CSV export error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/json', methods=['POST'])
async def export_json():
    """Export search results as JSON"""
    try:
        data = await request.get_json()
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'No results to export'}), 400
        
        # Add metadata
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_results': len(results),
                'version': '3.0.0'
            },
            'results': results
        }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"naver_cafe_search_{timestamp}.json"
        filepath = Path(resource_path(filename))
        
        # Save to JSON with Korean support
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return await send_file(str(filepath), as_attachment=True)
        
    except Exception as e:
        logger.error(f"JSON export error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-text', methods=['POST'])
async def analyze_text():
    """Analyze text and extract keywords using NLP"""
    try:
        data = await request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        from nlp_processor import get_nlp_processor
        nlp = get_nlp_processor()
        
        # Extract nouns and phrases
        nouns = nlp.extract_nouns(text)
        phrases = nlp.extract_key_phrases(text)
        
        return jsonify({
            'nouns': nouns,
            'phrases': phrases,
            'noun_count': len(nouns),
            'phrase_count': len(phrases)
        })
        
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-queries', methods=['POST'])
async def generate_queries():
    """Generate search queries from keywords"""
    try:
        data = await request.get_json()
        keywords = data.get('keywords', [])
        
        if not keywords:
            return jsonify({'error': 'No keywords provided'}), 400
        
        from query_generator import get_query_generator
        generator = get_query_generator()
        
        all_queries = []
        for keyword in keywords:
            queries = generator.generate_queries(
                keyword,
                search_intent='community',
                exclude_keywords=data.get('excludeKeywords', [])
            )
            all_queries.extend(queries)
        
        # Remove duplicates
        unique_queries = list(dict.fromkeys(all_queries))
        
        return jsonify({
            'queries': unique_queries,
            'query_count': len(unique_queries),
            'strategies_applied': [
                'exact_phrase',
                'mandatory_terms',
                'or_combination',
                'proximity',
                'negative_filter'
            ]
        })
        
    except Exception as e:
        logger.error(f"Query generation error: {e}")
        return jsonify({'error': str(e)}), 500

# Browser auto-open helper
def open_browser():
    """Open browser after server starts"""
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open_new(f"http://{CONFIG['server']['host']}:{CONFIG['server']['port']}")

# Error handlers
@app.errorhandler(404)
async def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
async def server_error(error):
    """Handle 500 errors"""
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Main execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Naver Cafe Search System')
    parser.add_argument('--no-browser', action='store_true', 
                       help='Do not open browser automatically')
    parser.add_argument('--port', type=int, default=CONFIG['server']['port'], 
                       help='Port to run the server on')
    parser.add_argument('--host', default=CONFIG['server']['host'], 
                       help='Host to bind the server to')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Initialize searcher
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    if loop.run_until_complete(initialize_searcher()):
        logger.info(f"Starting Enhanced Naver Cafe Search System on {args.host}:{args.port}")
        
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
        logger.error("Failed to initialize searcher. Check your API credentials.")
        print("\n" + "="*50)
        print("ERROR: Failed to initialize searcher")
        print("Please check your Naver API credentials in:")
        print("1. config.json file, or")
        print("2. Environment variables (NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)")
        print("="*50 + "\n")
        input("Press Enter to exit...")