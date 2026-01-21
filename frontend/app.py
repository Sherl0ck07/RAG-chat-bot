"""
Streamlit Frontend for Veltris Intelligent Doc-Bot
"""

import streamlit as st
import time
from typing import Dict, List, Optional
from datetime import datetime

from utils.api_client import APIClient, APIError
from utils.ui_components import (
    display_message,
    display_sources,
    display_confidence_badge,
    display_loading_spinner
)

# Page configuration
st.set_page_config(
    page_title="Veltris Doc-Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 3px solid #667eea;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 Veltris Doc-Bot")
    st.markdown("---")
    
    # Health check
    with st.spinner("Checking system status..."):
        health = st.session_state.api_client.health_check()
    
    if health and health.get("status") == "healthy":
        st.success("✅ System Online")
        st.metric("Documents Loaded", health.get("total_documents", 0))
        st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")
    else:
        st.error("❌ System Offline")
        st.warning("Backend not responding. Please start the API server.")
    
    st.markdown("---")
    
    # Documentation subset info
    st.markdown("### 📚 Knowledge Base")
    st.info("**Active Dataset**: HuggingFace Accelerate Documentation")
    
    st.markdown("---")
    
    # Settings
    with st.expander("⚙️ Settings"):
        top_k = st.slider("Sources to retrieve", 1, 10, 5)
        st.caption("Number of documents to search")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Built with ❤️ for Veltris")

# Main content
st.markdown('<p class="main-header">🤖 Veltris Intelligent Doc-Bot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about HuggingFace Accelerate documentation</p>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            st.markdown("---")
            display_sources(message["sources"])
        
        # Display confidence if available
        if message["role"] == "assistant" and "confidence" in message:
            display_confidence_badge(message["confidence"])

# Chat input
if prompt := st.chat_input("Ask me anything about Accelerate..."):
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call API
                response = st.session_state.api_client.query(prompt, top_k=top_k)
                
                if response:
                    answer = response.get("answer", "I couldn't generate a response.")
                    sources = response.get("sources", [])
                    confidence = response.get("confidence", 0.0)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        st.markdown("---")
                        display_sources(sources)
                    
                    # Display confidence
                    display_confidence_badge(confidence)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "confidence": confidence
                    })
                else:
                    st.error("Failed to get response from the bot.")
                    
            except APIError as e:
                st.error(f"API Error: {e.message}")
                st.caption(f"Status Code: {e.status_code}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# Welcome message for empty chat
if len(st.session_state.messages) == 0:
    st.markdown("### 👋 Welcome!")
    st.markdown("""
    I'm your intelligent documentation assistant. I can help you with questions about:
    
    - **Installation & Setup**: How to install and configure Accelerate
    - **Training**: Distributed training, mixed precision, and optimization
    - **Configuration**: Setting up training environments and hardware
    - **Best Practices**: Tips and recommendations for using Accelerate
    
    Just type your question below to get started!
    """)
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("What is Accelerate?", use_container_width=True):
            st.session_state.example_query = "What is Accelerate?"
            st.rerun()
        if st.button("How do I use mixed precision?", use_container_width=True):
            st.session_state.example_query = "How do I use mixed precision training?"
            st.rerun()
    
    with col2:
        if st.button("How to configure for multiple GPUs?", use_container_width=True):
            st.session_state.example_query = "How do I configure Accelerate for multiple GPUs?"
            st.rerun()
        if st.button("What is gradient accumulation?", use_container_width=True):
            st.session_state.example_query = "What is gradient accumulation in Accelerate?"
            st.rerun()

# Handle example query clicks
if "example_query" in st.session_state:
    # Trigger the query
    prompt = st.session_state.example_query
    del st.session_state.example_query
    
    # Add to messages and process
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.api_client.query(prompt, top_k=top_k)
                
                if response:
                    answer = response.get("answer", "I couldn't generate a response.")
                    sources = response.get("sources", [])
                    confidence = response.get("confidence", 0.0)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "confidence": confidence
                    })
                    
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")