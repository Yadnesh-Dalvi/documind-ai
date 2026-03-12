# backend/test_llm.py
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"  # Free, fast, very capable

# ============================================================
# LESSON: Groq uses OpenAI-compatible API format.
# Messages are a list of {role, content} dicts.
# Roles: "system" (behavior), "user" (input), "assistant" (history)
# This is the industry standard format — same in OpenAI, Claude API etc.
# ============================================================
print("--- TEST 1: Basic prompt ---")
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "What is RAG in AI? Explain in 3 bullet points."}
    ],
    temperature=0.2,
    max_tokens=500
)
print(response.choices[0].message.content)
print(f"\nTokens used: {response.usage.total_tokens}")

# ============================================================
# LESSON: System prompt goes as the FIRST message with role="system"
# This is the RAG pattern — the system prompt constrains the model
# to only answer from context you provide.
# ============================================================
print("\n--- TEST 2: System prompt (the RAG pattern) ---")

fake_context = """
DocuMind AI is a document intelligence platform built with FastAPI and React.
It uses ChromaDB as a vector database and Groq LLM for fast inference.
The system supports PDF uploads and semantic search.
"""

response2 = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": """You are a helpful document assistant.
ONLY answer questions based on the context provided.
If the answer is not in the context, say 'I cannot find this in the documents.'
Never make up information."""
        },
        {
            "role": "user",
            "content": f"Context:\n{fake_context}\n\nQuestion: What database does DocuMind use?"
        }
    ],
    temperature=0.2
)
print(response2.choices[0].message.content)

# ============================================================
# LESSON: Streaming with Groq — set stream=True
# You get chunks back one by one, just like ChatGPT.
# chunk.choices[0].delta.content is each token as it arrives.
# ============================================================
print("\n--- TEST 3: Streaming response ---")
print("Streaming: ", end="", flush=True)

stream = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Count from 1 to 5, one number per line."}
    ],
    stream=True
)

for chunk in stream:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="", flush=True)

print("\n\n✅ Module 1 complete! Your LLM connection works.")