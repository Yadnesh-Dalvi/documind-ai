# backend/test_vectordb.py
import chromadb
from sentence_transformers import SentenceTransformer
import json

# ============================================================
# LESSON: PersistentClient saves your vector DB to disk.
# Every time you run this, data is saved to ./chroma_db folder
# This is your AI's long term memory — survives restarts.
# In production: Pinecone/Weaviate replace this. Same concept.
# ============================================================
print("--- SETUP: Initialize ChromaDB ---")
client = chromadb.PersistentClient(path="./chroma_db")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Collection = like a table in SQL, or an index in Elasticsearch
# Each collection stores chunks from your documents
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
)
print(f"Collection ready: '{collection.name}'\n")

# ============================================================
# LESSON: This simulates what happens when a user uploads a PDF.
# Real flow: PDF → extract text → split into chunks → embed → store
# Each chunk gets:
#   - id: unique identifier
#   - embedding: the 384-number vector
#   - document: the original text (stored for retrieval)
#   - metadata: source info (filename, page number etc.)
# ============================================================
print("--- TEST 1: Indexing documents (the ingestion pipeline) ---")

# Simulated PDF chunks — in Module 4 these come from real PDFs
chunks = [
    "DocuMind AI is a document intelligence platform built with FastAPI and React.",
    "ChromaDB is used as the vector database for storing and retrieving embeddings.",
    "Groq provides fast LLM inference using the Llama 3.3 70B model.",
    "Users can upload PDF files through the React frontend interface.",
    "The RAG pipeline retrieves relevant chunks before generating answers.",
    "Embeddings are created using the all-MiniLM-L6-v2 sentence transformer model.",
    "FastAPI handles file uploads, document processing, and query endpoints.",
    "The system uses cosine similarity to find the most relevant document chunks.",
]

metadatas = [
    {"source": "documind_overview.pdf", "page": 1},
    {"source": "documind_overview.pdf", "page": 1},
    {"source": "documind_overview.pdf", "page": 2},
    {"source": "documind_overview.pdf", "page": 2},
    {"source": "technical_docs.pdf",    "page": 1},
    {"source": "technical_docs.pdf",    "page": 1},
    {"source": "technical_docs.pdf",    "page": 2},
    {"source": "technical_docs.pdf",    "page": 2},
]

# Generate embeddings for all chunks at once
print(f"Embedding {len(chunks)} chunks...")
embeddings = model.encode(chunks).tolist()  # ChromaDB needs list, not numpy array

# Store in ChromaDB
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)
print(f"✅ Stored {collection.count()} chunks in ChromaDB\n")

# ============================================================
# LESSON: This is the RETRIEVAL step of RAG.
# We embed the user's question → search for similar vectors
# ChromaDB returns the closest chunks using cosine similarity.
# n_results=3 means "give me the 3 most relevant chunks"
# ============================================================
print("--- TEST 2: Semantic search (the retrieval step) ---")

queries = [
    "What database does DocuMind use?",
    "How are embeddings generated?",
    "What does the frontend look like?",
]

for query in queries:
    print(f"Query: '{query}'")
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        similarity = 1 - dist  # ChromaDB returns distance, not similarity
        print(f"  #{i+1} ({similarity:.4f}) [{meta['source']} p{meta['page']}]")
        print(f"       {doc}")
    print()

# ============================================================
# LESSON: Metadata filtering lets you search within specific docs.
# In our app: "only search chunks from THIS uploaded PDF"
# This is how multi-document RAG works — you filter by filename.
# ============================================================
print("--- TEST 3: Metadata filtering (search within specific docs) ---")

query = "What model is used for inference?"
query_embedding = model.encode(query).tolist()

# Only search within technical_docs.pdf
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    where={"source": "technical_docs.pdf"},  # filter by metadata
    include=["documents", "metadatas"]
)

print(f"Query: '{query}'")
print(f"Searching only in: technical_docs.pdf\n")
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f"  [{meta['source']} p{meta['page']}] {doc}")

print("\n✅ Module 3 complete! You can store and retrieve vectors.")