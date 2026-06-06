import os 

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def load_documents(doc_path):
    documents = []
    for filename in os.listdir(doc_path):
        if filename.endswith(".txt"):
            with open(os.path.join(doc_path, filename), "r", encoding="utf-8") as file:
                text = file.read().strip()
                if text:
                    documents.append({
                        "filename": filename,
                        "text": text
                    })
    return documents

def chunk_documents(documents, chunk_size=150, overlap=50):
    chunks = []
    for doc in documents:
        if len(doc["text"]) > chunk_size:
            for i in range(0, len(doc["text"]), chunk_size - overlap):
                chunks.append({
                    "filename": doc["filename"],
                    "text": doc["text"][i:i+chunk_size],
                    "id": f"{doc['filename']}_{i//chunk_size}"
                })
        else:
            chunks.append({
                    "filename": doc["filename"],
                    "text": doc["text"],
                    "id": f"{doc['filename']}_0"
            })
    return chunks

def embed_text(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return embeddings.tolist()

def upsert_data(index, chunks, embeddings):
    meta_vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        meta_vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
                "filename": chunk["filename"],
                "text": chunk["text"]
            }
        })
    print(index.describe_index_stats())
    index.upsert(vectors=meta_vectors)

def retrieve_data(index, query):
    retrieved_chunks = []
    query_embedding = embed_text([query])[0]
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    for match in results["matches"]:
        retrieved_chunks.append({
            "score": match["score"],
            "filename": match["metadata"]["filename"],
            "text": match["metadata"]["text"]
        })
    return retrieved_chunks

def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=GEMINI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    documents = load_documents("policies/")
    chunks = chunk_documents(documents)
    chunk_texts = [chunk['text'] for chunk in chunks]
    embeddings = embed_text(chunk_texts)
    upsert_data(index, chunks, embeddings)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers query based on the context given ONLY. If you don't know the answer, I dont have enough information."),
        ("human", "{context}\n\nQuestion: {question}")
    ])

    chain = prompt | llm

    question = input("Ask a question: ")

    context = retrieve_data(index, question)
    context_str = "\n\n".join([chunk["text"] for chunk in context])

    results = chain.invoke({
        "context": context_str,
        "question": question
    })

    print("Answer:", results.content)

if __name__ == "__main__":
    main()





