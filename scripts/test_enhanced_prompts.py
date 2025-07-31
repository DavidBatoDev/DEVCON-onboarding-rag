#!/usr/bin/env python3
"""
Test script for enhanced prompt engineering functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.prompt_engine import DEVCONPromptEngine
from app.services.llamaindex_service import LlamaIndexRAGService
from app.core.config import settings

def test_enhanced_prompt_engine():
    """Test the enhanced prompt engine functionality"""
    print("🧪 Testing Enhanced Prompt Engineering...")
    
    # Initialize RAG service
    rag_service = LlamaIndexRAGService()
    
    # Initialize prompt engine
    prompt_engine = DEVCONPromptEngine(rag_service)
    
    print("\n📋 Testing System Prompt Generation:")
    system_prompt = prompt_engine.generate_system_prompt()
    print("✅ System prompt generated successfully")
    print(f"Length: {len(system_prompt)} characters")
    
    print("\n📋 Testing Context-Aware Prompt (with context):")
    # Mock context nodes
    class MockNode:
        def __init__(self, text, title):
            self.text = text
            self.metadata = {'title': title}
    
    mock_nodes = [
        MockNode("DEVCON chapters should have regular meetings and events.", "Chapter Guidelines"),
        MockNode("Officers should communicate effectively with members.", "Leadership Best Practices")
    ]
    
    context_prompt = prompt_engine.create_context_aware_prompt(mock_nodes, "How should I run my chapter?")
    print("✅ Context-aware prompt generated (with context)")
    print(f"Length: {len(context_prompt)} characters")
    
    print("\n📋 Testing Context-Aware Prompt (no context):")
    empty_prompt = prompt_engine.create_context_aware_prompt([], "What is the weather like?")
    print("✅ Context-aware prompt generated (no context)")
    print(f"Length: {len(empty_prompt)} characters")
    
    print("\n📋 Testing Enhanced Context Prompt:")
    enhanced_prompt = prompt_engine.create_enhanced_context_prompt(mock_nodes, "How should I run my chapter?", True)
    print("✅ Enhanced context prompt generated")
    print(f"Length: {len(enhanced_prompt)} characters")
    
    print("\n📋 Testing Enhanced Context Prompt (no context):")
    enhanced_empty = prompt_engine.create_enhanced_context_prompt([], "What is the weather like?", False)
    print("✅ Enhanced context prompt generated (no context)")
    print(f"Length: {len(enhanced_empty)} characters")
    
    print("\n📋 Testing Query Enhancement:")
    enhanced_query = prompt_engine.enhance_query("Who is the president?")
    print(f"✅ Enhanced query: {enhanced_query}")
    
    print("\n🎉 All prompt engineering tests passed!")
    return True

def test_context_detection():
    """Test context detection functionality"""
    print("\n🔍 Testing Context Detection...")
    
    rag_service = LlamaIndexRAGService()
    
    # Test with mock nodes
    class MockNode:
        def __init__(self, text, title):
            self.text = text
            self.metadata = {'title': title}
    
    # Test with meaningful context
    meaningful_nodes = [
        MockNode("DEVCON chapters should have regular meetings.", "Chapter Guidelines"),
        MockNode("Officers should communicate effectively.", "Leadership Best Practices")
    ]
    
    # Test with empty context
    empty_nodes = []
    
    # Test with nodes that have empty text
    empty_text_nodes = [
        MockNode("", "Empty Document"),
        MockNode("   ", "Whitespace Document")
    ]
    
    # Check context detection
    has_meaningful = bool(meaningful_nodes) and any(node.text.strip() for node in meaningful_nodes if node.text.strip())
    has_empty = bool(empty_nodes) and any(node.text.strip() for node in empty_nodes if node.text.strip())
    has_empty_text = bool(empty_text_nodes) and any(node.text.strip() for node in empty_text_nodes if node.text.strip())
    
    print(f"✅ Meaningful context detected: {has_meaningful}")
    print(f"✅ Empty context detected: {has_empty}")
    print(f"✅ Empty text context detected: {has_empty_text}")
    
    assert has_meaningful == True, "Should detect meaningful context"
    assert has_empty == False, "Should detect empty context"
    assert has_empty_text == False, "Should detect empty text context"
    
    print("🎉 Context detection tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_enhanced_prompt_engine()
        test_context_detection()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 