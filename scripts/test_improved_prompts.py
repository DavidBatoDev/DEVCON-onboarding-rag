#!/usr/bin/env python3
"""
Test script to verify improved prompt engineering
"""

class MockNode:
    """Mock node class for testing"""
    def __init__(self, title, source, file_id, text):
        self.metadata = {
            'title': title,
            'source': source,
            'file_id': file_id
        }
        self.text = text

class MockRAGService:
    """Mock RAG service for testing"""
    def get_index_state(self):
        return {
            "doc1": {"title": "Chapter Guidelines", "file_name": "guidelines.pdf"},
            "doc2": {"title": "Leadership Handbook", "file_name": "handbook.pdf"}
        }

def test_new_conversation_prompt():
    """Test that new conversations get proper introduction"""
    print("🧪 Testing new conversation prompt...")
    
    # Mock the prompt engine
    from app.services.prompt_engine import DEVCONPromptEngine
    
    rag_service = MockRAGService()
    prompt_engine = DEVCONPromptEngine(rag_service)
    
    # Test with no conversation history (new conversation)
    mock_nodes = [MockNode("Test Doc", "test.pdf", "123", "This is test content")]
    
    prompt = prompt_engine.create_enhanced_context_prompt(
        mock_nodes, 
        "How do I run my chapter?", 
        has_relevant_context=True,
        conversation_history=None
    )
    
    print("📝 Generated prompt for new conversation:")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    # Check that it includes DEBBIE introduction
    if "🤖 You are DEBBIE" in prompt:
        print("✅ New conversation includes proper DEBBIE introduction")
        return True
    else:
        print("❌ New conversation missing DEBBIE introduction")
        return False

def test_continuing_conversation_prompt():
    """Test that continuing conversations don't reintroduce DEBBIE"""
    print("\n🧪 Testing continuing conversation prompt...")
    
    # Mock the prompt engine
    from app.services.prompt_engine import DEVCONPromptEngine
    
    rag_service = MockRAGService()
    prompt_engine = DEVCONPromptEngine(rag_service)
    
    # Test with conversation history (continuing conversation)
    mock_nodes = [MockNode("Test Doc", "test.pdf", "123", "This is test content")]
    conversation_history = "User: How do I start a chapter?\nAssistant: Great question! Here's how to get started..."
    
    prompt = prompt_engine.create_enhanced_context_prompt(
        mock_nodes, 
        "What about leadership roles?", 
        has_relevant_context=True,
        conversation_history=conversation_history
    )
    
    print("📝 Generated prompt for continuing conversation:")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    # Check that it doesn't reintroduce DEBBIE
    if "🤖 You are DEBBIE" not in prompt and "🤖 Welcome" not in prompt:
        print("✅ Continuing conversation avoids DEBBIE reintroduction")
        return True
    else:
        print("❌ Continuing conversation still reintroduces DEBBIE")
        return False

def test_conversation_history_inclusion():
    """Test that conversation history is properly included"""
    print("\n🧪 Testing conversation history inclusion...")
    
    # Mock the prompt engine
    from app.services.prompt_engine import DEVCONPromptEngine
    
    rag_service = MockRAGService()
    prompt_engine = DEVCONPromptEngine(rag_service)
    
    # Test with conversation history
    mock_nodes = [MockNode("Test Doc", "test.pdf", "123", "This is test content")]
    conversation_history = "User: How do I start a chapter?\nAssistant: Great question! Here's how to get started..."
    
    prompt = prompt_engine.create_enhanced_context_prompt(
        mock_nodes, 
        "What about leadership roles?", 
        has_relevant_context=True,
        conversation_history=conversation_history
    )
    
    # Check that conversation history is included
    if "Recent conversation:" in prompt and "How do I start a chapter?" in prompt:
        print("✅ Conversation history properly included")
        return True
    else:
        print("❌ Conversation history not included")
        return False

def test_short_conversation_threshold():
    """Test that short conversations still get introduction"""
    print("\n🧪 Testing short conversation threshold...")
    
    # Mock the prompt engine
    from app.services.prompt_engine import DEVCONPromptEngine
    
    rag_service = MockRAGService()
    prompt_engine = DEVCONPromptEngine(rag_service)
    
    # Test with very short conversation history
    mock_nodes = [MockNode("Test Doc", "test.pdf", "123", "This is test content")]
    conversation_history = "Hi"  # Very short, should still get introduction
    
    prompt = prompt_engine.create_enhanced_context_prompt(
        mock_nodes, 
        "How do I run my chapter?", 
        has_relevant_context=True,
        conversation_history=conversation_history
    )
    
    # Check that short conversation still gets introduction
    if "🤖 You are DEBBIE" in prompt:
        print("✅ Short conversation gets proper introduction")
        return True
    else:
        print("❌ Short conversation missing introduction")
        return False

def test_no_context_scenarios():
    """Test scenarios with no relevant context"""
    print("\n🧪 Testing no context scenarios...")
    
    # Mock the prompt engine
    from app.services.prompt_engine import DEVCONPromptEngine
    
    rag_service = MockRAGService()
    prompt_engine = DEVCONPromptEngine(rag_service)
    
    # Test new conversation with no context
    prompt_new = prompt_engine.create_enhanced_context_prompt(
        [], 
        "What's the weather like?", 
        has_relevant_context=False,
        conversation_history=None
    )
    
    # Test continuing conversation with no context
    prompt_continue = prompt_engine.create_enhanced_context_prompt(
        [], 
        "What's the weather like?", 
        has_relevant_context=False,
        conversation_history="User: How do I start a chapter?\nAssistant: Here's how..."
    )
    
    # Check that new conversation gets introduction
    if "🤖 You are DEBBIE" in prompt_new:
        print("✅ New conversation with no context gets introduction")
    else:
        print("❌ New conversation with no context missing introduction")
        return False
    
    # Check that continuing conversation doesn't reintroduce
    if "🤖 You are DEBBIE" not in prompt_continue:
        print("✅ Continuing conversation with no context avoids reintroduction")
        return True
    else:
        print("❌ Continuing conversation with no context still reintroduces")
        return False

if __name__ == "__main__":
    print("🚀 Testing improved prompt engineering...")
    
    tests = [
        test_new_conversation_prompt,
        test_continuing_conversation_prompt,
        test_conversation_history_inclusion,
        test_short_conversation_threshold,
        test_no_context_scenarios
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All prompt engineering tests passed!")
    else:
        print("💥 Some prompt engineering tests failed!")
        exit(1) 