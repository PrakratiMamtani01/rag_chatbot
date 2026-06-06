import os
from time import time

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path


load_dotenv(find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-system")

def load_documents(docs_dir):
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Documents folder not found: {docs_dir}")

    documents = []

    for file_path in docs_path.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        if text:
            documents.append({
                "filename": file_path.name,
                "text": text
            })

    return documents

def chunk_documents(documents, chunk_size=500):
    chunks = []
    for doc in documents:
        for i in range(0, len(doc["text"]), chunk_size):
            chunks.append({
                "filename": doc["filename"],
                "text": doc["text"][i:i+chunk_size],
                "id": f"{doc['filename']}_{i//chunk_size}"
            })
    return chunks

def embeddings(model, texts):
    embeddings = model.encode(texts)
    return embeddings.tolist()


def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index["name"] for index in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Wait until index is ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(PINECONE_INDEX_NAME)

def upsert_chunks(index, embedding_model, chunks):
    texts = [chunk["text"] for chunk in chunks]
    vectors = embeddings(embedding_model, texts)

    pinecone_vectors = []

    for chunk, vector in zip(chunks, vectors):
        pinecone_vectors.append({
            "id": chunk["id"],
            "values": vector,
            "metadata": {
                "filename": chunk["filename"],
                "text": chunk["text"]
            }
        })

    index.upsert(vectors=pinecone_vectors)

    print(f"Uploaded {len(pinecone_vectors)} chunks to Pinecone.")


def retrieve_relevant_chunks(index, embedding_model, question, top_k=3):
    question_vector = embeddings(embedding_model, [question])[0]

    results = index.query(
        vector=question_vector,
        top_k=top_k,
        include_metadata=True
    )

    retrieved_chunks = []

    for match in results["matches"]:
        retrieved_chunks.append({
            "score": match["score"],
            "filename": match["metadata"]["filename"],
            "text": match["metadata"]["text"]
        })

    return retrieved_chunks

def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY
    )

    documents = load_documents("policies/")

    model = SentenceTransformer(model_name="all-MiniLM-L6-v2")

    chunks = chunk_documents(documents)

    texts = [chunk["text"] for chunk in chunks]
    embeddings_list = embeddings(model, texts)
