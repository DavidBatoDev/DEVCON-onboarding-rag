# backend/app/services/prompt_engine.py
import logging

logger = logging.getLogger(__name__)

# Enhanced prompt_engine.py with improved context prioritization and transparency

class DEVCONPromptEngine:
    def __init__(self, rag_service):
        self.rag_service = rag_service
    
    def generate_system_prompt(self):
        """Generate DEVCON-specific system prompt with strong context prioritization"""
        try:
            # Get document titles from index state
            titles = []
            for file_id, meta in self.rag_service.get_index_state().items():
                titles.append(meta.get("title", meta["file_name"]))
            
            # Format titles as a bullet list
            title_list = "\n".join([f"• {title}" for title in titles[:10]])
            if len(titles) > 10:
                title_list += f"\n• ... and {len(titles) - 10} more documents"
            
            # Enhanced context-prioritized prompt
            prompt = f"""🤖 You are DEBBIE the DEVCON Officers' Onboarding Assistant! You're here to help with chapter management and tech community building.

📋 CRITICAL RESPONSE GUIDELINES:
1. 🎯 ALWAYS check the provided context FIRST for relevant information
2. ✅ If context contains helpful information, use it and cite it naturally
3. 🚫 If context is empty, irrelevant, or insufficient, use your general knowledge
4. 📢 When using general knowledge, ALWAYS state: "I don't have specific information about this in the provided documents, but based on general knowledge..."
5. 💡 When using context, mention it naturally: "According to the documents..." or "I found this in the materials..."
6. 🤷 Be honest about information sources - never make up information
7. 😊 Keep responses friendly, helpful, and engaging with emojis
8. 🎯 Focus on being practical and actionable for DEVCON chapter officers

📚 Available Documents:
{title_list}

💬 CONVERSATION STYLE:
- Use emojis to make conversations more engaging 🎉
- Be conversational and approachable 
- Provide helpful, practical advice for chapter leadership
- Ask follow-up questions when clarification would help 🤔
- Celebrate successes and encourage growth 🚀
- Always be transparent about your information sources

🔍 SOURCE TRANSPARENCY:
- If using context: "Based on the provided documents..."
- If using general knowledge: "I don't have specific information about this in the provided documents, but based on general knowledge..."
- If unsure: "I don't have enough information to provide a confident answer about this specific topic."

Below is some context retrieved from documents. Check this context FIRST, and only use general knowledge if the context doesn't contain relevant information! 📖✨"""
            
            return prompt.strip()
        except Exception as e:
            logger.error(f"Error generating system prompt: {e}")
            # Fallback prompt with context prioritization focus
            return """🤖 You are the DEVCON Officers' Assistant! 

CRITICAL: Always check the provided context FIRST for relevant information. Only use general knowledge if context doesn't contain helpful information, and always state when you're using general knowledge vs. context information.

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
        """Create a prompt that emphasizes context prioritization and source transparency"""
        
        # Check if we have meaningful context
        has_relevant_context = bool(context_nodes) and any(
            node.text.strip() for node in context_nodes if node.text.strip()
        )
        
        # Extract and format context with source tracking
        context_with_sources = []
        for i, node in enumerate(context_nodes, 1):
            source = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
            context_with_sources.append(f"[Source {i}: {source}]\n{node.text}\n")
        
        full_context = "\n---\n".join(context_with_sources)
        
        if has_relevant_context:
            prompt = f"""🤖 Welcome to DEBBIE the DEVCON Officers' Onboarding Assistant! 

I have found relevant information in the provided documents. Please use this context to answer the question:

CONTEXT INFORMATION:
{full_context}

QUESTION: {question}

💡 RESPONSE GUIDELINES:
1. 📚 Use the context above as your PRIMARY source of information
2. ✅ Cite the context naturally: "According to the documents..." or "I found this in the materials..."
3. 😊 Keep responses friendly and engaging with emojis
4. 🎯 Focus on practical, actionable advice for chapter officers
5. 📝 Be specific about which parts of the context you're referencing

Let's help make your chapter awesome! 🚀

ANSWER:"""
        else:
            # No relevant context found - instruct to use general knowledge with transparency
            prompt = f"""🤖 Welcome to DEBBIE the DEVCON Officers' Onboarding Assistant! 

I have checked the available documents but couldn't find specific information related to your question. I will provide guidance based on general knowledge about DEVCON and chapter management.

QUESTION: {question}

💡 RESPONSE GUIDELINES:
1. 🚫 Start your response with: "I don't have specific information about this in the provided documents, but based on general knowledge..."
2. 😊 Keep responses friendly and engaging with emojis
3. 🎯 Focus on practical, actionable advice for chapter officers
4. 📝 Provide helpful general guidance while being transparent about the source
5. 🤔 If you're unsure about something, say so rather than guessing

Let's help make your chapter awesome! 🚀

ANSWER:"""
        
        return prompt

    def create_enhanced_context_prompt(self, context_nodes, question: str, has_relevant_context: bool = None) -> str:
        """Enhanced prompt that explicitly handles context vs general knowledge scenarios"""
        
        if has_relevant_context is None:
            # Determine if we have meaningful context
            has_relevant_context = bool(context_nodes) and any(
                node.text.strip() for node in context_nodes if node.text.strip()
            )
        
        # Extract and format context with source tracking
        context_with_sources = []
        for i, node in enumerate(context_nodes, 1):
            source = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
            context_with_sources.append(f"[Source {i}: {source}]\n{node.text}\n")
        
        full_context = "\n---\n".join(context_with_sources)
        
        if has_relevant_context:
            prompt = f"""🤖 You are DEBBIE the DEVCON Officers' Onboarding Assistant!

I have found relevant information in the provided documents. Use this context as your PRIMARY source:

CONTEXT FROM DOCUMENTS:
{full_context}

QUESTION: {question}

📋 RESPONSE REQUIREMENTS:
1. 🎯 Use the context above as your main source of information
2. ✅ When referencing context, say: "According to the documents..." or "I found this in the materials..."
3. 😊 Keep responses friendly and engaging with emojis
4. 🎯 Focus on practical, actionable advice for chapter officers
5. 📝 Be specific about which parts of the context you're using

ANSWER:"""
        else:
            prompt = f"""🤖 You are DEBBIE the DEVCON Officers' Onboarding Assistant!

I have checked the available documents but couldn't find specific information related to your question. Provide guidance using general knowledge about DEVCON and chapter management.

QUESTION: {question}

📋 RESPONSE REQUIREMENTS:
1. 🚫 Start with: "I don't have specific information about this in the provided documents, but based on general knowledge..."
2. 😊 Keep responses friendly and engaging with emojis
3. 🎯 Focus on practical, actionable advice for chapter officers
4. 📝 Be transparent that you're using general knowledge
5. 🤔 If unsure about something, say so rather than guessing

ANSWER:"""
        
        return prompt

