#!/usr/bin/env python3
"""
Test script to verify source deduplication in the RAG service
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llamaindex_service import LlamaIndexRAGService
from llama_index.core import Document
from typing import List, Dict

def test_source_deduplication():
    """Test that sources are properly deduplicated"""
    print("🧪 Testing source deduplication...")
    
    # Initialize the service
    rag_service = LlamaIndexRAGService()
    
    # Create test documents with duplicate sources
    test_documents = [
        Document(
            text="This is test content 1",
            metadata={
                "title": "Test Document 1",
                "source": "test1.pdf",
                "file_id": "12345"
            }
        ),
        Document(
            text="This is test content 2",
            metadata={
                "title": "Test Document 1",  # Same title as above
                "source": "test1.pdf",       # Same source as above
                "file_id": "12345"           # Same file_id as above
            }
        ),
        Document(
            text="This is test content 3",
            metadata={
                "title": "Test Document 2",
                "source": "test2.pdf",
                "file_id": "67890"
            }
        ),
        Document(
            text="This is test content 4",
            metadata={
                "title": "Test Document 1",  # Same title as first
                "source": "test1.pdf",       # Same source as first
                "file_id": "12345"           # Same file_id as first
            }
        )
    ]
    
    try:
        # Add documents to the index
        success = rag_service.add_documents(test_documents)
        if not success:
            print("❌ Failed to add documents to index")
            return False
        
        # Test query that should return sources
        test_query = "What is the content?"
        
        # Test the main query method
        print("\n📝 Testing main query method...")
        response = rag_service.query(test_query)
        
        # Check for duplicate sources in the response
        if "📚 **Sources Referenced:**" in response:
            sources_section = response.split("📚 **Sources Referenced:**")[1].split("\n\n")[0]
            source_lines = [line.strip() for line in sources_section.split("\n") if line.strip().startswith(("1.", "2.", "3."))]
            
            print(f"📊 Found {len(source_lines)} source references")
            
            # Check for duplicates
            unique_sources = set()
            for line in source_lines:
                if line.startswith(("1.", "2.", "3.")):
                    source_text = line[2:].strip()  # Remove numbering
                    unique_sources.add(source_text)
            
            print(f"📊 Unique sources: {len(unique_sources)}")
            print(f"📊 Total source lines: {len(source_lines)}")
            
            if len(unique_sources) < len(source_lines):
                print("❌ Duplicate sources detected!")
                return False
            else:
                print("✅ No duplicate sources detected!")
        
        # Test the reference control method
        print("\n📝 Testing reference control method...")
        response_with_control = rag_service.query_with_reference_control(test_query, force_show_references=True)
        
        if "📚 **Sources Referenced:**" in response_with_control:
            sources_section = response_with_control.split("📚 **Sources Referenced:**")[1].split("\n\n")[0]
            source_lines = [line.strip() for line in sources_section.split("\n") if line.strip().startswith(("1.", "2.", "3."))]
            
            print(f"📊 Found {len(source_lines)} source references")
            
            # Check for duplicates
            unique_sources = set()
            for line in source_lines:
                if line.startswith(("1.", "2.", "3.")):
                    source_text = line[2:].strip()  # Remove numbering
                    unique_sources.add(source_text)
            
            print(f"📊 Unique sources: {len(unique_sources)}")
            print(f"📊 Total source lines: {len(source_lines)}")
            
            if len(unique_sources) < len(source_lines):
                print("❌ Duplicate sources detected in reference control!")
                return False
            else:
                print("✅ No duplicate sources detected in reference control!")
        
        print("\n✅ All deduplication tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        try:
            rag_service.clear_index()
        except:
            pass

if __name__ == "__main__":
    success = test_source_deduplication()
    if success:
        print("\n🎉 Source deduplication is working correctly!")
    else:
        print("\n💥 Source deduplication test failed!")
        sys.exit(1) 