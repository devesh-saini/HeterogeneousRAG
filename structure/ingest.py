from langchain_ollama import OllamaEmbeddings
import chromadb


chroma_client = chromadb.PersistentClient(path="../vectordb/")


all_data = chroma_client.get_or_create_collection("all_collections")


embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

