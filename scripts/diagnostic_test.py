"""
RAG System Diagnostics Script
Run this to test and tune your RAG system after compilation
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.services.llamaindex_service import rag_service

def test_retrieval_thresholds():
    """Test different similarity thresholds to find optimal settings"""
    print("🔧 Testing retrieval thresholds...")
    
    test_queries = [
        "What is DEVCON?",
        "event schedule",
        "speakers",
        "technologies",
        "programming",
        "Who is Dom? What is his role and achievements?",
        "How to apply in DEVCON Internship program?",
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing query: '{query}'")
        
        # Test raw retrieval scores
        result = rag_service.test_retrieval_scores(query)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
            
        stats = result["score_stats"]
        print(f"📊 Retrieved {result['num_results']} results")
        print(f"📊 Score range: {stats['min']:.3f} to {stats['max']:.3f} (avg: {stats['mean']:.3f})")
        
        # Show sample results
        print("📄 Top results:")
        for i, sample in enumerate(result["sample_results"], 1):
            print(f"  {i}. Score: {sample['score']:.3f} - {sample['text_preview']}")

def test_queries_with_different_cutoffs():
    """Test the same queries with different similarity cutoffs"""
    print("\n🔧 Testing different similarity cutoffs...")
    
    test_query = "Tell me about the event schedule"
    cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for cutoff in cutoffs:
        print(f"\n🎯 Testing with similarity_cutoff={cutoff}")
        
        # Temporarily modify the query engine
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.postprocessor import SimilarityPostprocessor
        from llama_index.core.query_engine import RetrieverQueryEngine
        
        retriever = VectorIndexRetriever(
            index=rag_service.index,
            similarity_top_k=10,
        )
        postprocessor = SimilarityPostprocessor(similarity_cutoff=cutoff)
        temp_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor],
        )
        
        try:
            response = temp_engine.query(test_query)
            if response.response and response.response.strip() != "Empty Response":
                print(f"✅ Got response: {response.response[:100]}...")
                if hasattr(response, 'source_nodes'):
                    print(f"📚 Used {len(response.source_nodes)} source nodes")
            else:
                print("❌ Empty response")
        except Exception as e:
            print(f"❌ Error: {e}")

def show_sample_documents():
    """Show sample documents in the index"""
    print("\n📄 Sample documents in the index:")
    
    try:
        stats = rag_service.get_index_stats()
        print(f"Total documents: {stats.get('document_count', 0)}")
        
        # Get some sample nodes
        from llama_index.core.retrievers import VectorIndexRetriever
        
        retriever = VectorIndexRetriever(
            index=rag_service.index,
            similarity_top_k=5,
        )
        
        # Use a very general query to get diverse results
        nodes = retriever.retrieve("information")
        
        print(f"Sample content from {len(nodes)} nodes:")
        for i, node in enumerate(nodes, 1):
            preview = node.text[:200].replace('\n', ' ') + "..."
            source = node.metadata.get('source', node.metadata.get('title', 'Unknown'))
            print(f"\n{i}. Source: {source}")
            print(f"   Content: {preview}")
            
    except Exception as e:
        print(f"❌ Error showing documents: {e}")

def main():
    """Run all diagnostics"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    RAG DIAGNOSTICS                           ║
    ║              Troubleshooting Empty Responses                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if system is initialized
    stats = rag_service.get_index_stats()
    if stats.get('status') != 'ready':
        print("❌ RAG system not ready. Please run compilation first.")
        return
    
    print(f"✅ RAG system ready with {stats.get('document_count', 0)} documents")
    
    # Run diagnostics
    show_sample_documents()
    test_retrieval_thresholds()
    test_queries_with_different_cutoffs()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                     RECOMMENDATIONS                          ║
    ║                                                              ║
    ║  1. Use similarity_cutoff between 0.2-0.3 for BGE          ║
    ║  2. Increase top_k retrieval to 10-15                       ║
    ║  3. Check if document content matches query language         ║
    ║  4. Consider query preprocessing/expansion                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()