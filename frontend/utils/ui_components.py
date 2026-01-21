"""
Reusable UI Components for Streamlit
"""

import streamlit as st
from typing import List, Dict


def display_sources(sources: List[Dict]):
    """
    Display source citations in an organized format
    
    Args:
        sources: List of source citation dictionaries
    """
    if not sources:
        return
    
    st.markdown("**📚 Sources:**")
    
    for idx, source in enumerate(sources, 1):
        section = source.get("section", "Unknown Section")
        filename = source.get("filename", "Unknown File")
        similarity = source.get("similarity_score", 0.0)
        excerpt = source.get("excerpt", "")
        
        with st.expander(f"📄 Source {idx}: {section} (Similarity: {similarity:.2%})"):
            st.markdown(f"**File:** `{filename}`")
            if excerpt:
                st.markdown(f"**Excerpt:**")
                st.caption(excerpt)


def display_confidence_badge(confidence: float):
    """
    Display confidence score as a colored badge
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
    """
    if confidence >= 0.7:
        badge_class = "confidence-high"
        label = "High Confidence"
    elif confidence >= 0.4:
        badge_class = "confidence-medium"
        label = "Medium Confidence"
    else:
        badge_class = "confidence-low"
        label = "Low Confidence"
    
    st.markdown(
        f'<span class="confidence-badge {badge_class}">🎯 {label}: {confidence:.0%}</span>',
        unsafe_allow_html=True
    )


def display_message(role: str, content: str):
    """
    Display a chat message with appropriate styling
    
    Args:
        role: 'user' or 'assistant'
        content: Message content
    """
    with st.chat_message(role):
        st.markdown(content)


def display_loading_spinner(message: str = "Processing..."):
    """
    Display a loading spinner with message
    
    Args:
        message: Loading message to display
    """
    with st.spinner(message):
        pass


def display_error(message: str, details: str = None):
    """
    Display an error message
    
    Args:
        message: Main error message
        details: Optional detailed error information
    """
    st.error(message)
    if details:
        with st.expander("Error Details"):
            st.code(details)


def display_info_box(title: str, content: str):
    """
    Display an information box
    
    Args:
        title: Box title
        content: Box content
    """
    st.info(f"**{title}**\n\n{content}")