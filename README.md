## Retrival Augmented Chatbot System

### Description
This chatbot is scalable system using policy documents as domain-specific knowledge base. For efficient computing the documents are chunked with overlap so that the semantic meaning between 2 chunks is not lost. It is then saved into Pinecone vector database so when retrieving an answer it can compute cosine similarity between embeddings and give top_k answer. 

## Techstack
- Python
- Langchain
- Google Gemini API
- PineconeAPI

## How to Run
- Ensure you have Python installed
- Create a virtual requirement
- ``` pip install langchain, langchain_google_genai, pinecone, sentence_transformers, dotenv ```
- Create a .env file with your GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
- ``` python practice2.py ```
