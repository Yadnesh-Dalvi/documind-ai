# backend/test_embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

# ============================================================
# LESSON: SentenceTransformer loads the model locally.
# First run downloads ~80MB to your machine and caches it.
# 'all-MiniLM-L6-v2' is the most popular embedding model
# for RAG projects — fast, small, and very accurate.
# ============================================================
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")

# ============================================================
# LESSON: .encode() converts text → vector (list of numbers)
# The output is always 384 numbers for this model.
# Every sentence, paragraph, or word → same size vector.
# This fixed size is what makes math operations possible.
# ============================================================
print("--- TEST 1: What does an embedding look like? ---")
sentence = "Machine learning is a subset of artificial intelligence."
vector = model.encode(sentence)

print(f"Original text: {sentence}")
print(f"Vector dimensions: {len(vector)}")
print(f"First 10 numbers: {vector[:10].round(4)}")
print(f"Data type: {type(vector)}\n")

# ============================================================
# LESSON: Cosine similarity measures how similar two vectors
# are — regardless of their size. Score ranges:
#   1.0  = identical meaning
#   0.7+ = very similar
#   0.0  = completely unrelated
#  -1.0  = opposite meaning
# This is the math that powers semantic search in ChromaDB.
# ============================================================
print("--- TEST 2: Semantic similarity (the magic of embeddings) ---")

def cosine_similarity(v1, v2):
    # Dot product divided by product of magnitudes
    dot = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / magnitude

sentences = [
    "How do I reset my password?",           # query
    "Steps to recover your account access.", # same meaning, different words
    "The weather today is very sunny.",      # completely unrelated
    "Password recovery instructions.",       # related
    "I love eating pizza on weekends.",      # unrelated
]

query = sentences[0]
query_vector = model.encode(query)

print(f"Query: '{query}'\n")
for sentence in sentences[1:]:
    vector = model.encode(sentence)
    score = cosine_similarity(query_vector, vector)
    bar = "█" * int(score * 20)
    print(f"  {score:.4f} {bar}")
    print(f"  '{sentence}'\n")

# ============================================================
# LESSON: Batch encoding is how we'll process PDF chunks.
# Instead of encoding one by one, encode all chunks at once.
# This is 10x faster than a loop — important for large docs.
# ============================================================
print("--- TEST 3: Batch encoding (how we'll process PDFs) ---")

pdf_chunks = [
    "DocuMind AI is a document intelligence platform.",
    "It uses ChromaDB as a vector database for storing embeddings.",
    "The backend is built with FastAPI and Python.",
    "Users can upload PDF files and ask questions about them.",
    "Groq provides fast LLM inference using Llama 3.3.",
]

# Encode all chunks at once
chunk_vectors = model.encode(pdf_chunks)
print(f"Encoded {len(pdf_chunks)} chunks")
print(f"Shape: {chunk_vectors.shape}  ← (num_chunks, dimensions)")
print(f"Each chunk → {chunk_vectors.shape[1]} numbers\n")

# Simulate a user query against these chunks
user_query = "What LLM does DocuMind use?"
query_vec = model.encode(user_query)

print(f"User query: '{user_query}'")
print("\nRanked results by similarity:")
scores = [(cosine_similarity(query_vec, cv), chunk) for cv, chunk in zip(chunk_vectors, pdf_chunks)]
scores.sort(reverse=True)

for rank, (score, chunk) in enumerate(scores, 1):
    print(f"  #{rank} ({score:.4f}) {chunk}")

print("\n✅ Module 2 complete! You understand embeddings.")