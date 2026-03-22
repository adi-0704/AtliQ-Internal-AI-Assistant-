import streamlit as st
import os
import json
from dotenv import load_dotenv
from rag_engine import get_rag_chain_manual, check_pii, mask_pii
from ingest_data import ingest_data

load_dotenv()

st.set_page_config(page_title="AtliQ Internal AI Assistant", layout="wide")

USER_DB_FILE = "users.json"

def load_users():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    return {
        "adi": {"password": "admin123", "role": "c-level"},
        "admin": {"password": "password123", "role": "c-level"}
    }

def save_users(users):
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)

def login_signup_page():
    users = load_users()
    
    st.title("🔐 AtliQ Internal - Authentication")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.markdown("### Sign in to access the AI Assistant")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login_user in users and users[login_user]["password"] == login_pass:
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.session_state.role = users[login_user]["role"]
                st.success(f"Logged in as {login_user} ({users[login_user]['role']})")
                st.rerun()
            else:
                st.error("Invalid username or password")
                
    with tab2:
        st.markdown("### Create a new internal account")
        new_user = st.text_input("Choose Username", key="new_user")
        new_pass = st.text_input("Choose Password", type="password", key="new_pass")
        confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
        role_choice = st.selectbox("Select Role", ["finance", "hr", "engineering", "marketing"], key="role_choice")
        
        if st.button("Create Account"):
            if new_user in users:
                st.error("Username already exists")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match")
            elif len(new_pass) < 6:
                st.error("Password must be at least 6 characters")
            elif new_user == "":
                st.error("Username cannot be empty")
            else:
                users[new_user] = {"password": new_pass, "role": role_choice}
                save_users(users)
                st.success("Account created successfully! Please log in.")

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
                st.error("🚨 Input Guardrail Triggered: Sensitive information (PII) detected in your query.")
            st.session_state.messages.append({"role": "assistant", "content": "Input Guardrail Triggered: Sensitive information (PII) detected."})
        else:
            # Generate response using RAG engine
            with st.chat_message("assistant"):
                with st.spinner(f"Retrieving information for {role} role..."):
                    try:
                        rag_chain, retriever = get_rag_chain_manual(role)
                        context_docs = retriever.invoke(prompt)
                        answer = rag_chain.invoke(prompt)
                        answer = mask_pii(answer)
                        st.markdown(answer)
                        
                        tokens = len(prompt.split()) + len(answer.split()) + 500
                        st.session_state.total_tokens += tokens
                        st.session_state.total_cost += (tokens / 1000) * 0.0001
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.expander("View Sources"):
                            for doc in context_docs:
                                st.write(f"- **{doc.metadata.get('source', 'Unknown')}** (Dept: {doc.metadata.get('department', 'Unknown')})")
                                st.caption(doc.page_content[:200] + "...")
                                
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

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
    login_signup_page()
else:
    main_app()
