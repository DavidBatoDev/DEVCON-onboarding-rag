#!/usr/bin/env python3
"""
Test to verify how backend consumes history
"""

def test_history_consumption():
    """Test the history consumption logic"""
    print("🧪 Testing history consumption...")
    
    # Simulate the frontend sending history
    frontend_history = [
        {"role": "user", "content": "How do I start a chapter?"},
        {"role": "assistant", "content": "Great question! Here's how to get started..."},
        {"role": "user", "content": "What about leadership roles?"},
        {"role": "assistant", "content": "Leadership roles are important..."},
        {"role": "user", "content": "Can you tell me more about events?"},
        {"role": "assistant", "content": "Events are a key part of chapter management..."},
        {"role": "user", "content": "What's the best way to engage members?"},
        {"role": "assistant", "content": "Member engagement is crucial..."},
        {"role": "user", "content": "How do I handle conflicts?"},
        {"role": "assistant", "content": "Conflict resolution is an important skill..."},
    ]
    
    print(f"📊 Frontend sends: {len(frontend_history)} messages")
    
    # Simulate backend processing (old way - only 5 exchanges)
    def old_format_history(history):
        if not history:
            return ""
        
        formatted_history = []
        for msg in history[-5:]:  # Only last 5 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_history)
    
    # Simulate backend processing (new way - 10 exchanges)
    def new_format_history(history):
        if not history:
            return ""
        
        formatted_history = []
        for msg in history[-10:]:  # Use last 10 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_history)
    
    old_result = old_format_history(frontend_history)
    new_result = new_format_history(frontend_history)
    
    print(f"\n📝 Old backend processing (5 exchanges):")
    print(f"Length: {len(old_result)} characters")
    print(f"User messages: {old_result.count('User:')}")
    print(f"Assistant messages: {old_result.count('Assistant:')}")
    
    print(f"\n📝 New backend processing (10 exchanges):")
    print(f"Length: {len(new_result)} characters")
    print(f"User messages: {new_result.count('User:')}")
    print(f"Assistant messages: {new_result.count('Assistant:')}")
    
    # Test conversation detection
    def is_new_conversation(history_str):
        return (
            not history_str or 
            len(history_str.strip()) < 50 or
            "User:" not in history_str or
            "Assistant:" not in history_str or
            history_str.count("User:") < 1 or
            history_str.count("Assistant:") < 1
        )
    
    print(f"\n🔍 Conversation detection:")
    print(f"Old result is new conversation: {is_new_conversation(old_result)}")
    print(f"New result is new conversation: {is_new_conversation(new_result)}")
    
    if not is_new_conversation(new_result):
        print("✅ New processing correctly detects continuing conversation!")
    else:
        print("❌ New processing still thinks it's a new conversation")
    
    return True

if __name__ == "__main__":
    test_history_consumption() 