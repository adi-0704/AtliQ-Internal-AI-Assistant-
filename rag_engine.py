import os
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_DIR = "qdrant_db"
COLLECTION_NAME = "company_data"

# Role permissions mapping
ROLE_PERMISSIONS = {
    "finance": ["finance", "marketing", "general"],
    "hr": ["hr", "general"],
    "engineering": ["engineering", "general"],
    "marketing": ["marketing", "general"],
    "c-level": ["finance", "hr", "engineering", "marketing", "general"]
}

# Singleton for vectorstore to avoid concurrent access errors
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            path=DB_DIR
        )
    return _vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain_manual(role):
    vectorstore = get_vectorstore()
    
    # Define metadata filter based on role
    allowed_depts = ROLE_PERMISSIONS.get(role, ["general"])
    
    # Qdrant filtering
    from qdrant_client.http import models as rest
    
    filter_obj = rest.Filter(
        must=[
            rest.FieldCondition(
                key="metadata.department",
                match=rest.MatchAny(any=allowed_depts)
            )
        ]
    )
    
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": filter_obj
        }
    )
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Define prompt
    system_prompt = (
        "You are an internal company assistant at AtliQ. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer, say that you don't know. "
        "Maintain a professional and helpful tone. "
        "Your current access role is: {role}. You can only see data for: {allowed_depts}.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the RAG chain using LCEL
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "input": RunnablePassthrough(),
            "role": lambda x: role,
            "allowed_depts": lambda x: ", ".join(allowed_depts)
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def check_pii(text):
    # Basic PII detection (emails, phone numbers)
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    
    if re.search(email_pattern, text) or re.search(phone_pattern, text):
        return True
    return False

def mask_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    
    masked_text = re.sub(email_pattern, "[EMAIL_REDACTED]", text)
    masked_text = re.sub(phone_pattern, "[PHONE_REDACTED]", masked_text)
    
    return masked_text
