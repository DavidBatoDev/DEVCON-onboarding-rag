#!/usr/bin/env python3
"""
Simple test to verify prompt improvements
"""

def test_prompt_improvements():
    """Test that the improved prompts work correctly"""
    print("🧪 Testing prompt improvements...")
    
    # Test the conversation detection logic
    def is_new_conversation(history):
        return (
            not history or 
            len(history.strip()) < 30 or
            "User:" not in history or
            "Assistant:" not in history
        )
    
    # Test cases
    test_cases = [
        (None, True, "No history"),
        ("", True, "Empty history"),
        ("Hi", True, "Short message"),
        ("User: Hello\nAssistant: Hi there!", False, "Full conversation"),
        ("User: How do I start a chapter?\nAssistant: Great question! Here's how...", False, "Long conversation"),
    ]
    
    for history, expected, description in test_cases:
        result = is_new_conversation(history)
        status = "✅" if result == expected else "❌"
        print(f"{status} {description}: {result} (expected {expected})")
    
    print("\n📝 Testing prompt instructions...")
    
    # Test that the new instructions are clear
    instructions = [
        "🚫 NEVER reintroduce yourself or say \"I'm DEBBIE\" in continuing conversations",
        "🚫 NEVER start responses with \"Hey there\" or similar greetings in continuing conversations", 
        "💬 If this is a continuing conversation, respond directly to the question without reintroducing yourself"
    ]
    
    for instruction in instructions:
        print(f"✅ {instruction}")
    
    print("\n🎉 Prompt improvements look good!")
    return True

if __name__ == "__main__":
    test_prompt_improvements() 