#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Test and Benchmark Script
====================================
Tests and benchmarks AI-enhanced Korean cafe search system
"""

import asyncio
import time
import psutil
import json
from typing import List, Dict, Any
from pathlib import Path
import logging

# Setup minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTester:
    """Performance testing suite for AI cafe search system"""
    
    def __init__(self):
        """Initialize performance tester"""
        self.results = {}
        self.start_memory = psutil.virtual_memory().used
        self.start_time = time.time()
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete performance benchmark"""
        logger.info("🚀 Starting AI-Enhanced Cafe Search Performance Benchmark")
        
        # Test cases
        test_cases = [
            ("simple_search", ["프로그래밍"]),
            ("complex_search", ["프로그래밍", "스터디", "파이썬"]),
            ("category_search", ["창업", "스타트업", "비즈니스"]),
            ("korean_specific", ["육아맘", "강남구", "커뮤니티"]),
            ("mixed_content", ["개발자", "모임", "서울", "IT"])
        ]
        
        benchmark_results = {
            "test_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "system_info": self._get_system_info(),
                "ai_info": self._get_ai_info()
            },
            "tests": {}
        }
        
        for test_name, keywords in test_cases:
            logger.info(f"🧪 Running test: {test_name}")
            test_result = await self._run_search_test(test_name, keywords)
            benchmark_results["tests"][test_name] = test_result
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        # Overall performance summary
        benchmark_results["summary"] = self._calculate_summary(benchmark_results["tests"])
        
        # Save results
        self._save_benchmark_results(benchmark_results)
        
        logger.info("✅ Benchmark completed successfully")
        return benchmark_results
    
    async def _run_search_test(self, test_name: str, keywords: List[str]) -> Dict[str, Any]:
        """Run individual search test"""
        try:
            # Import here to avoid issues if modules aren't available
            from enhanced_nlp_processor import get_enhanced_nlp
            from enhanced_cafe_searcher import AISearchSettings
            
            nlp = get_enhanced_nlp()
            
            # Test NLP processing
            nlp_start = time.time()
            search_text = ' '.join(keywords)
            features = nlp.extract_enhanced_features(search_text)
            expanded_queries = nlp.expand_query(search_text, max_expansions=5)
            nlp_time = time.time() - nlp_start
            
            # Test semantic similarity (if available)
            similarity_start = time.time()
            test_texts = [
                "프로그래밍 개발자 모임",
                "파이썬 스터디 그룹",
                "창업 커뮤니티 네트워킹"
            ]
            similarities = []
            for text in test_texts:
                sim = nlp.calculate_semantic_similarity(search_text, text)
                similarities.append(sim)
            similarity_time = time.time() - similarity_start
            
            # Memory usage
            current_memory = psutil.virtual_memory().used
            memory_used = (current_memory - self.start_memory) / (1024 * 1024)  # MB
            
            return {
                "keywords": keywords,
                "processing_time": {
                    "nlp_processing": nlp_time,
                    "similarity_calculation": similarity_time,
                    "total": nlp_time + similarity_time
                },
                "memory_usage_mb": memory_used,
                "features_extracted": {
                    "nouns_count": len(features.get('nouns', [])),
                    "categories_count": len(features.get('categories', [])),
                    "keywords_count": len(features.get('keywords', []))
                },
                "query_expansion": {
                    "original_query": search_text,
                    "expanded_count": len(expanded_queries),
                    "expansion_factor": len(expanded_queries) / len(keywords)
                },
                "semantic_similarities": {
                    "average": sum(similarities) / len(similarities) if similarities else 0,
                    "max": max(similarities) if similarities else 0,
                    "min": min(similarities) if similarities else 0
                },
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            return {
                "keywords": keywords,
                "status": "failed",
                "error": str(e)
            }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            import platform
            
            return {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "python_version": platform.python_version()
            }
        except Exception as e:
            logger.warning(f"Could not get system info: {e}")
            return {"error": str(e)}
    
    def _get_ai_info(self) -> Dict[str, Any]:
        """Get AI libraries information"""
        ai_info = {}
        
        # Check sentence transformers
        try:
            import sentence_transformers
            ai_info["sentence_transformers"] = sentence_transformers.__version__
        except ImportError:
            ai_info["sentence_transformers"] = "not available"
        
        # Check scikit-learn
        try:
            import sklearn
            ai_info["scikit_learn"] = sklearn.__version__
        except ImportError:
            ai_info["scikit_learn"] = "not available"
        
        # Check PyTorch
        try:
            import torch
            ai_info["torch"] = torch.__version__
            ai_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                ai_info["cuda_version"] = torch.version.cuda
                ai_info["gpu_count"] = torch.cuda.device_count()
        except ImportError:
            ai_info["torch"] = "not available"
        
        # Check Kiwipiepy
        try:
            import kiwipiepy
            ai_info["kiwipiepy"] = "available"
        except ImportError:
            ai_info["kiwipiepy"] = "not available"
        
        return ai_info
    
    def _calculate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance summary"""
        successful_tests = [t for t in test_results.values() if t.get("status") == "success"]
        
        if not successful_tests:
            return {"error": "No successful tests"}
        
        # Calculate averages
        avg_nlp_time = sum(t["processing_time"]["nlp_processing"] for t in successful_tests) / len(successful_tests)
        avg_similarity_time = sum(t["processing_time"]["similarity_calculation"] for t in successful_tests) / len(successful_tests)
        avg_total_time = sum(t["processing_time"]["total"] for t in successful_tests) / len(successful_tests)
        avg_memory = sum(t["memory_usage_mb"] for t in successful_tests) / len(successful_tests)
        
        # Feature extraction summary
        avg_nouns = sum(t["features_extracted"]["nouns_count"] for t in successful_tests) / len(successful_tests)
        avg_categories = sum(t["features_extracted"]["categories_count"] for t in successful_tests) / len(successful_tests)
        
        # Query expansion summary
        avg_expansion = sum(t["query_expansion"]["expansion_factor"] for t in successful_tests) / len(successful_tests)
        
        # Semantic similarity summary
        avg_semantic_sim = sum(t["semantic_similarities"]["average"] for t in successful_tests) / len(successful_tests)
        
        return {
            "total_tests": len(test_results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(test_results) - len(successful_tests),
            "performance_averages": {
                "nlp_processing_seconds": round(avg_nlp_time, 4),
                "similarity_calculation_seconds": round(avg_similarity_time, 4),
                "total_processing_seconds": round(avg_total_time, 4),
                "memory_usage_mb": round(avg_memory, 2)
            },
            "ai_effectiveness": {
                "average_nouns_extracted": round(avg_nouns, 1),
                "average_categories_found": round(avg_categories, 1),
                "average_query_expansion_factor": round(avg_expansion, 2),
                "average_semantic_similarity": round(avg_semantic_sim, 3)
            },
            "performance_rating": self._calculate_performance_rating(avg_total_time, avg_memory)
        }
    
    def _calculate_performance_rating(self, avg_time: float, avg_memory: float) -> str:
        """Calculate overall performance rating"""
        if avg_time < 0.1 and avg_memory < 100:
            return "Excellent"
        elif avg_time < 0.5 and avg_memory < 300:
            return "Good"
        elif avg_time < 1.0 and avg_memory < 500:
            return "Fair"
        else:
            return "Poor"
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 Benchmark results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")

async def quick_performance_test():
    """Run a quick performance test"""
    logger.info("🚀 Running Quick Performance Test")
    
    try:
        from enhanced_nlp_processor import get_enhanced_nlp
        
        nlp = get_enhanced_nlp()
        
        # Quick test
        start_time = time.time()
        features = nlp.extract_enhanced_features("프로그래밍 스터디 모임")
        end_time = time.time()
        
        logger.info(f"✅ NLP Processing Time: {end_time - start_time:.4f} seconds")
        logger.info(f"📊 Features Extracted: {len(features.get('nouns', []))} nouns, {len(features.get('categories', []))} categories")
        
        # Test semantic similarity if available
        try:
            start_time = time.time()
            similarity = nlp.calculate_semantic_similarity("프로그래밍 스터디", "개발자 모임")
            end_time = time.time()
            
            logger.info(f"🔗 Semantic Similarity Time: {end_time - start_time:.4f} seconds")
            logger.info(f"📈 Similarity Score: {similarity:.3f}")
        except Exception as e:
            logger.warning(f"Semantic similarity test failed: {e}")
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")

def print_system_requirements():
    """Print system requirements and recommendations"""
    print("\n" + "="*60)
    print("🖥️  SYSTEM REQUIREMENTS & RECOMMENDATIONS")
    print("="*60)
    
    # Current system info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    print(f"Current System:")
    print(f"  • RAM: {memory_gb:.1f} GB")
    print(f"  • CPU Cores: {cpu_count}")
    
    print(f"\nMinimum Requirements:")
    print(f"  • RAM: 8 GB")
    print(f"  • CPU: 4 cores")
    print(f"  • Storage: 2 GB free space")
    
    print(f"\nRecommended for Optimal Performance:")
    print(f"  • RAM: 16 GB+")
    print(f"  • CPU: 8+ cores")
    print(f"  • GPU: CUDA-capable (optional, 3-5x speedup)")
    
    # Performance predictions
    if memory_gb >= 16 and cpu_count >= 8:
        print(f"  ✅ Your system exceeds recommendations - Excellent performance expected")
    elif memory_gb >= 8 and cpu_count >= 4:
        print(f"  ⚡ Your system meets minimum requirements - Good performance expected")
    else:
        print(f"  ⚠️  Your system is below minimum requirements - Performance may be limited")
    
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Cafe Search Performance Testing")
    parser.add_argument('--quick', action='store_true', help='Run quick performance test')
    parser.add_argument('--full', action='store_true', help='Run full benchmark suite')
    parser.add_argument('--system-info', action='store_true', help='Show system information')
    
    args = parser.parse_args()
    
    if args.system_info:
        print_system_requirements()
    
    if args.quick:
        asyncio.run(quick_performance_test())
    
    if args.full:
        tester = PerformanceTester()
        results = asyncio.run(tester.run_full_benchmark())
        
        print("\n📊 BENCHMARK SUMMARY")
        print("="*50)
        summary = results.get("summary", {})
        if "performance_averages" in summary:
            perf = summary["performance_averages"]
            print(f"Average Processing Time: {perf['total_processing_seconds']:.4f}s")
            print(f"Average Memory Usage: {perf['memory_usage_mb']:.1f}MB")
            print(f"Performance Rating: {summary['performance_rating']}")
        
        if "ai_effectiveness" in summary:
            ai = summary["ai_effectiveness"]
            print(f"Query Expansion Factor: {ai['average_query_expansion_factor']:.2f}x")
            print(f"Semantic Similarity: {ai['average_semantic_similarity']:.3f}")
    
    if not any([args.quick, args.full, args.system_info]):
        print("Usage: python performance_test.py [--quick] [--full] [--system-info]")
        print_system_requirements()