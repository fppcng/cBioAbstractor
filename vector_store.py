import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

VECTORSTORE_DIR  = "./vector_store"
collection_name  = "pdf_chunks"
embeddings_model = OpenAIEmbeddings()

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings_model,
    persist_directory=VECTORSTORE_DIR,
)


def add_embeddings(chunks):
    texts    = [chunk.page_content for chunk in chunks]
    metadata = [{"source": str(chunk.metadata.get("source", ""))} for chunk in chunks]
    vector_store.add_texts(texts=texts, metadatas=metadata)
    return len(texts)


def search_vector_store(question, k=5):
    query_embedding = embeddings_model.embed_query(question)
    results = vector_store.similarity_search_by_vector(query_embedding, k=k)
    return [doc.page_content for doc in results]


def clear_vector_store():
    global vector_store          # must be first line before any use of the name
    vector_store.delete_collection()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings_model,
        persist_directory=VECTORSTORE_DIR,
    )