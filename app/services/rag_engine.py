import google.generativeai as genai
from app.services.vector_store import load_from_store
from app.services.embedder import embed_texts
from app.core.config import settings

# Configure Gemini
genai.configure(api_key=settings.GEMINI_API_KEY)

# Initialize vector store
vector_store = load_from_store()

def retrieve_relevant_chunks(query: str, top_k: int = 3):
    query_emb = embed_texts([query])[0]
    return vector_store.similarity_search(query_emb, top_k=top_k)

def get_gemini_response(query: str, context: str) -> str:
    # Initialize Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""You are a helpful assistant answering questions based on internal DEVCON documents.

Below is some context retrieved from documents. If it's helpful to answer the question or query, feel free to use it. Otherwise, ignore it. Add Emojis in chats to make it better.

📚 Context:
{context}

🧠 Query:
{query}

Please provide a helpful and accurate response based on the context provided."""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=1024,
            )
        )
        return response.text
    except Exception as e:
        print(f"Error generating response with Gemini: {e}")
        return "Sorry, I encountered an error while generating the response."

def ask_with_rag(query: str) -> str:
    chunks = retrieve_relevant_chunks(query)

    print(f"Retrieved {len(chunks)} relevant chunks for query: {query}")
    print("Chunks:", chunks)

    if not chunks:
        return "Sorry, I couldn't find relevant information from the documents."

    context = "\n\n".join([chunk.get("text") or chunk.get("content", "") for chunk in chunks])
    
    # Limit context to avoid token limits
    context = context[:48000]
    
    return get_gemini_response(query, context).strip()