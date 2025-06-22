# backend/app/services/prompt_engine.py
import logging

logger = logging.getLogger(__name__)

# backend/app/services/prompt_engine.py
# Enhanced prompt_engine.py with anti-hallucination strategies

class DEVCONPromptEngine:
    def __init__(self, rag_service):
        self.rag_service = rag_service
    
    def generate_system_prompt(self):
        """Generate DEVCON-specific system prompt with strong hallucination prevention"""
        try:
            # Get document titles from index state
            titles = []
            for file_id, meta in self.rag_service.get_index_state().items():
                titles.append(meta.get("title", meta["file_name"]))
            
            # Format titles as a bullet list
            title_list = "\n".join([f"• {title}" for title in titles[:10]])
            if len(titles) > 10:
                title_list += f"\n• ... and {len(titles) - 10} more documents"
            
            # Enhanced anti-hallucination prompt
            prompt = f"""🤖 You are the DEBBIE the DEVCON Officers' Onboarding Assistant! You're here to help with chapter management and tech community building.

📋 RESPONSE GUIDELINES:
1. ✅ Use the context provided below ONLY if it's helpful and relevant to answer the question
2. 🚫 If the context isn't helpful or relevant, feel free to ignore it completely  
3. 💡 When using context information, mention it naturally: "According to the documents..." or "I found this in the materials..."
4. 🤷 If you don't have enough information, just say so - no need to make things up!
5. 😊 Keep responses friendly, helpful, and engaging with emojis
6. 🎯 Focus on being practical and actionable for DEVCON chapter officers

📚 Available Documents:
{title_list}

💬 CONVERSATION STYLE:
- Use emojis to make conversations more engaging 🎉
- Be conversational and approachable 
- Provide helpful, practical advice for chapter leadership
- Ask follow-up questions when clarification would help 🤔
- Celebrate successes and encourage growth 🚀

Below is some context retrieved from documents. If it's helpful to answer the question or query, feel free to use it. Otherwise, ignore it and provide your best guidance as a DEVCON advisor! 📖✨"""
            
            return prompt.strip()
        except Exception as e:
            logger.error(f"Error generating system prompt: {e}")
            # Fallback prompt with anti-hallucination focus
            return """🤖 You are the DEVCON Officers' Assistant! 

Below is some context retrieved from documents. If it's helpful to answer the question, feel free to use it. Otherwise, ignore it and provide your best guidance! 

Remember to use emojis 😊 and be helpful while staying accurate to any information you do reference from the context! 🎯"""
    
    def enhance_query(self, question: str) -> str:
        """Add DEVCON context while preserving query specificity"""
        # Keep the original question more intact to avoid leading the retrieval
        enhanced = f"{question}"
        
        # Add minimal context only when it helps retrieval
        if "president" in question.lower():
            enhanced += " (specific role and person)"
        elif "leader" in question.lower() or "officer" in question.lower():
            enhanced += " (leadership roles)"
        elif "chapter" in question.lower():
            enhanced += " (chapter-specific information)"
            
        return enhanced

    def create_context_aware_prompt(self, context_nodes, question: str) -> str:
        """Create a prompt that emphasizes source fidelity"""
        
        # Extract and format context with source tracking
        context_with_sources = []
        for i, node in enumerate(context_nodes, 1):
            source = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
            context_with_sources.append(f"[Source {i}: {source}]\n{node.text}\n")
        
        full_context = "\n---\n".join(context_with_sources)
        
        prompt = f"""🤖 Welcome to DEBBIE the DEVCON Officers' Onboarding Assistant! 

Below is some context retrieved from documents. If it's helpful to answer the question or query, feel free to use it. Otherwise, ignore it and provide your best guidance! 📖✨

CONTEXT INFORMATION:
{full_context}

QUESTION: {question}

💡 RESPONSE GUIDELINES:
1. 📚 Use the context above if it's relevant and helpful
2. 🤷 If context isn't useful, ignore it and provide general DEVCON guidance  
3. 😊 Keep responses friendly and engaging with emojis
4. 🎯 Focus on practical, actionable advice for chapter officers
5. 📝 When referencing context, mention it naturally: "I found in the materials..." or "According to the documents..."

Let's help make your chapter awesome! 🚀

ANSWER:"""
        
        return prompt

