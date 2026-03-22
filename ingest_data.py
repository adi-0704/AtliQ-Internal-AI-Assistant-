import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

DATA_DIR = "data"
DB_DIR = "qdrant_db"

def ingest_data():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Ingestion will create its own client using the path
    
    documents = []
    
    # Iterate through each department folder
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory not found.")
        return

    departments = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for dept in departments:
        dept_path = os.path.join(DATA_DIR, dept)
        files = glob.glob(os.path.join(dept_path, "*"))
        
        for file_path in files:
            print(f"Loading {file_path} for department: {dept}")
            
            try:
                if file_path.endswith(".csv"):
                    loader = CSVLoader(file_path)
                elif file_path.endswith(".md"):
                    loader = TextLoader(file_path, encoding="utf-8")
                else:
                    continue
                
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata["department"] = dept
                    doc.metadata["source"] = os.path.basename(file_path)
                
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    if not documents:
        print("No documents found to ingest.")
        return

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    print(f"Total chunks to ingest: {len(splits)}")
    
    # Create vector store using langchain-qdrant
    vectorstore = QdrantVectorStore.from_documents(
        splits,
        embeddings,
        path=DB_DIR,
        collection_name="company_data",
        force_recreate=True
    )
    
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
