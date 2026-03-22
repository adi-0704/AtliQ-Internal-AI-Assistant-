import streamlit as st
import os
from dotenv import load_dotenv
from rag_engine import get_rag_chain_manual, check_pii, mask_pii
from ingest_data import ingest_data

load_dotenv()

st.set_page_config(page_title="AtliQ Internal AI Assistant", layout="wide")

# Mock user database
USERS = {
    "adi": {"password": "admin123", "role": "c-level"},
    "admin": {"password": "password123", "role": "c-level"},
    "finance_user": {"password": "finance_pass", "role": "finance"},
    "hr_user": {"password": "hr_pass", "role": "hr"},
    "eng_user": {"password": "eng_pass", "role": "engineering"},
    "marketing_user": {"password": "marketing_pass", "role": "marketing"}
}

def login_page():
    st.title("🔐 AtliQ Internal - Login")
    with st.container():
        st.markdown("### Please sign in to access the AI Assistant")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in USERS and USERS[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = USERS[username]["role"]
                st.success(f"Logged in as {username} ({USERS[username]['role']})")
                st.rerun()
            else:
                st.error("Invalid username or password")

def main_app():
    role = st.session_state.role
    username = st.session_state.username
    
    st.title("AtliQ Internal AI Assistant 🚀")
    # Ensure data is ingested on startup if DB is missing
    if not os.path.exists("qdrant_db"):
        with st.spinner("Initializing Vector Database... This may take a minute on first run."):
            ingest_data()
        st.success("Database initialized!")
    st.markdown(f"### Secure RAG with RBAC and Guardrails - Logged in as: **{username}**")

    # Sidebar for logout and info
    st.sidebar.title("🔐 Account Info")
    st.sidebar.write(f"User: **{username}**")
    st.sidebar.write(f"Role: **{role}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.sidebar.divider()

    # Initialize session state for chat history and cost
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about company data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Input Guardrail: PII Detection
        if check_pii(prompt):
            with st.chat_message("assistant"):
                st.error("🚨 Input Guardrail Triggered: Sensitive information (PII) detected in your query. Please do not share emails or phone numbers.")
            st.session_state.messages.append({"role": "assistant", "content": "Input Guardrail Triggered: Sensitive information (PII) detected."})
        else:
            # Generate response using RAG engine
            with st.chat_message("assistant"):
                with st.spinner(f"Retrieving information for {role} role..."):
                    try:
                        rag_chain, retriever = get_rag_chain_manual(role)
                        
                        # Manually handle retrieval for UI display of sources
                        context_docs = retriever.invoke(prompt)
                        
                        # Invoke manual RAG chain
                        answer = rag_chain.invoke(prompt)
                        
                        # Output Guardrail: Mask PII if any (double check)
                        answer = mask_pii(answer)
                        
                        st.markdown(answer)
                        
                        # Estimate tokens
                        tokens = len(prompt.split()) + len(answer.split()) + 500
                        st.session_state.total_tokens += tokens
                        st.session_state.total_cost += (tokens / 1000) * 0.0001
                        
                        # Store response
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Display sources in expander
                        with st.expander("View Sources"):
                            for doc in context_docs:
                                st.write(f"- **{doc.metadata.get('source', 'Unknown')}** (Dept: {doc.metadata.get('department', 'Unknown')})")
                                st.caption(doc.page_content[:200] + "...")
                                
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

    # Bottom info
    st.sidebar.subheader("📊 Usage Statistics")
    st.sidebar.write(f"Total Tokens: **{st.session_state.total_tokens}**")
    st.sidebar.write(f"Estimated Cost: **${st.session_state.total_cost:.4f}**")

    st.sidebar.divider()
    st.sidebar.caption("System Status:")
    st.sidebar.success("✅ RAG Engine: Online")
    st.sidebar.success("✅ Guardrails: Active")
    st.sidebar.success("✅ Monitoring: Active")

# Main execution logic
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:
    main_app()
