# streamlit_app.py
import streamlit as st
import requests
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Deep Researcher - RAG Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API = "http://127.0.0.1:8000/query"
API_BACKEND = "http://127.0.0.1:8000/backend"
API_SET_BACKEND = "http://127.0.0.1:8000/set_backend"

# Backend Labels
BACKEND_LABELS = {
    "lexrank": "📝 LexRank (Extractive, lightweight)",
    "t5": "🤖 DistilBART (Abstractive, HuggingFace)",
    "ollama": "🦙 Ollama LLM (LLaMA/Mistral, quantized)"
}

# Function to set backend
def set_backend(backend_key):
    """Set the backend via API call"""
    try:
        resp = requests.post(API_SET_BACKEND, params={"backend": backend_key}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return True, data.get("message", "Backend updated successfully")
        else:
            return False, f"Failed to set backend: HTTP {resp.status_code}"
    except Exception as e:
        return False, f"Error setting backend: {str(e)[:50]}..."

# Custom CSS for better styling
st.markdown("""
<style>
    /* SIDEBAR FIXES */
    .stSidebar > div:first-child {
        background-color: #f8f9fa !important;
    }
    
    .stSidebar .stMarkdown {
        color: #333333 !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    .stSidebar p {
        color: #333333 !important;
    }
    
    .stSidebar .stAlert {
        background-color: #e3f2fd !important;
        color: #1565c0 !important;
        border: 1px solid #bbdefb !important;
    }
    
    .stSidebar label {
        color: #333333 !important;
        font-weight: 500 !important;
    }
    
    /* FORM ELEMENTS FIXES */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: #333333 !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        caret-color: #333333 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        caret-color: #333333 !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #999999 !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: white !important;
        color: #333333 !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        caret-color: #333333 !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        caret-color: #333333 !important;
    }
    
    .stNumberInput > div > div > div > button {
        background-color: #f8f9fa !important;
        color: #333333 !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* Sidebar form elements */
    .stSidebar .stTextInput > div > div > input,
    .stSidebar .stNumberInput > div > div > input {
        background-color: white !important;
        color: #333333 !important;
        border: 1px solid #dee2e6 !important;
    }
    
    .stSidebar .stSlider > div > div > div > div {
        background-color: #e9ecef !important;
    }
    
    .stSidebar .stCheckbox > label > div {
        color: #333333 !important;
    }
    
    /* Diverse mode highlight */
    .diverse-mode-enabled {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4) !important;
        color: white !important;
        padding: 8px 12px !important;
        border-radius: 20px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        display: inline-block !important;
        margin-left: 10px !important;
    }
    
    /* Sidebar collapse button */
    .stSidebar .css-1rs6os, .stSidebar .css-17eq0hr {
        color: #333333 !important;
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
    }
    
    button[kind="secondary"] {
        color: #333333 !important;
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* Expander styling */
    .stSidebar .streamlit-expanderHeader {
        background-color: #e9ecef !important;
        color: #495057 !important;
        border-radius: 5px !important;
    }
    
    .stSidebar .streamlit-expanderContent {
        background-color: #f8f9fa !important;
    }
    
    /* MAIN CONTENT */
    .stApp {
        background-color: white !important;
    }
    
    .main .block-container {
        color: #333333 !important;
    }
    
    /* Stats visibility */
    [data-testid="metric-container"] {
        background-color: white !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        color: #333333 !important;
    }
    
    [data-testid="metric-container"] > div > div {
        color: #333333 !important;
    }
    
    [data-testid="metric-container"] > div:first-child {
        color: #667eea !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    [data-testid="metric-container"] > div:last-child {
        color: #666666 !important;
        font-size: 0.9rem !important;
    }
    
    /* Headers */
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666666 !important;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .backend-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .backend-status * {
        color: white !important;
    }
    
    /* Text content visibility */
    div[data-testid="stMarkdownContainer"] {
        color: #333333 !important;
    }
    
    div[data-testid="stMarkdownContainer"] p {
        color: #333333 !important;
    }
    
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stMarkdownContainer"] h6 {
        color: #2c3e50 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #667eea !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background-color: #5a67d8 !important;
    }
    
    /* Override any problematic styling */
    .element-container {
        color: #333333 !important;
    }
    
    /* Ensure all regular text is visible */
    p, div, span, label {
        color: #333333 !important;
    }
    
    /* Keep specific elements with intended colors */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Function to get backend status with loading state
def get_backend_status_sync():
    """Non-cached version for immediate loading feedback"""
    try:
        resp = requests.get(API_BACKEND, timeout=5)
        if resp.status_code == 200:
            backend_key = resp.json().get("backend", "lexrank").lower()
            model_name = resp.json().get("model", "")
            backend_display = BACKEND_LABELS.get(backend_key, "❓ Unknown backend")
            
            # Add model info for Ollama
            if backend_key == "ollama" and model_name:
                backend_display += f" — Model: {model_name}"
            
            return backend_display, True, None
        else:
            return "❌ API error", False, f"HTTP {resp.status_code}"
    except Exception as e:
        return "❌ Connection failed", False, str(e)[:50]

# Cached version for performance
@st.cache_data(ttl=30, show_spinner=False)
def get_backend_status_cached():
    return get_backend_status_sync()

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Document Upload Section
    st.markdown("## 📄 Upload Documents")
    uploaded_files = st.file_uploader("Upload one or more PDFs",
                                      type=["pdf"],
                                      accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            files = []
            for f in uploaded_files:
                files.append(("files", (f.name, f, "application/pdf")))
            try:
                resp = requests.post("http://127.0.0.1:8000/upload", files=files)
                if resp.status_code == 200:
                    st.success(resp.json().get("message", "✅ Uploaded successfully!"))
                else:
                    st.error(f"Upload failed: {resp.text}")
            except Exception as e:
                st.error(f"Upload error: {e}")
    
    st.markdown("---")  # Add separator
    
    # Backend status with proper loading state
    st.markdown("**Backend Status:**")
    
    # Create a placeholder for the status
    status_placeholder = st.empty()
    
    # Check if we have cached status first
    if 'backend_status_cache' in st.session_state:
        # Use cached status
        backend_status, is_connected, error_msg = st.session_state.backend_status_cache
        status_color = "🟢" if is_connected else "🔴"
        
        with status_placeholder:
            st.markdown(f"{status_color}")
            if is_connected:
                st.success(backend_status)
            else:
                st.error(backend_status)
                if error_msg:
                    with st.expander("Connection Details"):
                        st.caption(f"Error: {error_msg}")
    else:
        # Show loading and fetch status
        with status_placeholder:
            with st.spinner("🔄 Checking backend status..."):
                time.sleep(0.5)  # Brief delay to ensure spinner shows
                backend_status, is_connected, error_msg = get_backend_status_sync()
                
                # Cache the result
                st.session_state.backend_status_cache = (backend_status, is_connected, error_msg)
        
        # Clear and show final status
        status_placeholder.empty()
        status_color = "🟢" if is_connected else "🔴"
        
        with status_placeholder:
            st.markdown(f"{status_color}")
            if is_connected:
                st.success(backend_status)
            else:
                st.error(backend_status)
                if error_msg:
                    with st.expander("Connection Details"):
                        st.caption(f"Error: {error_msg}")
    
    # Add refresh button
    if st.button("🔄 Refresh Status", help="Check backend status again"):
        if 'backend_status_cache' in st.session_state:
            del st.session_state.backend_status_cache
        st.rerun()
    
    # Backend selector
    st.markdown("### 🔧 Backend Selection")
    
    # Get current backend from session state or default
    if 'current_backend' not in st.session_state:
        # Try to get current backend from API
        try:
            current_status, _, _ = get_backend_status_sync()
            # Extract backend from status message
            for key, label in BACKEND_LABELS.items():
                if key.lower() in current_status.lower():
                    st.session_state.current_backend = key
                    break
            else:
                st.session_state.current_backend = "lexrank"  # Default
        except:
            st.session_state.current_backend = "lexrank"
    
    # Get the index of current backend for selectbox
    current_index = list(BACKEND_LABELS.keys()).index(st.session_state.current_backend)
    
    backend_choice = st.selectbox(
        "Choose Summarizer Backend",
        options=list(BACKEND_LABELS.keys()),
        index=current_index,
        format_func=lambda x: BACKEND_LABELS[x],
        help="Select which backend to use for generating answers",
        key="backend_selector"
    )
    
    # Check if backend changed and update it
    if backend_choice != st.session_state.current_backend:
        with st.spinner(f"🔄 Switching to {BACKEND_LABELS[backend_choice]}..."):
            success, message = set_backend(backend_choice)
            
            if success:
                st.session_state.current_backend = backend_choice
                # Clear backend status cache to force refresh
                if 'backend_status_cache' in st.session_state:
                    del st.session_state.backend_status_cache
                st.success(f"✅ {message}")
                # Add a small delay to show the success message
                time.sleep(1.5)
                st.rerun()
            else:
                st.error(f"❌ {message}")
                # Reset selectbox to previous value on error
                st.session_state.backend_selector = st.session_state.current_backend
    
    # Show current backend confirmation
    if 'current_backend' in st.session_state:
        st.info(f"🎯 Currently using: **{BACKEND_LABELS[st.session_state.current_backend]}**")
    
    # Search parameters - ENHANCED WITH DIVERSE MODE
    st.markdown("### 🔧 Search Parameters")
    top_k = st.slider("Number of results to retrieve:", 1, 20, 5, help="More results may provide better context but slower response")
    word_limit = st.number_input("Summarize in ~N words (optional):", min_value=0, step=10, value=0, help="Set to 0 for no word limit")
    
    # DIVERSE MODE FEATURE
    diverse_mode = st.checkbox(
        "🔥 Enable Diverse Mode (FlowRL-style)", 
        False,
        help="Enhance retrieval diversity using FlowRL-inspired techniques for more comprehensive coverage"
    )
    
    # Show diverse mode info when enabled
    if diverse_mode:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #ff6b6b, #4ecdc4); color: white; padding: 10px; border-radius: 8px; margin: 10px 0;">
                <strong>🔥 Diverse Mode Active</strong><br>
                <small>Using FlowRL-inspired diversity enhancement for better topic coverage</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Advanced options
    with st.expander("🔬 Advanced Options"):
        timeout_duration = st.number_input("Request timeout (seconds):", 30, 300, 120)
        show_scores = st.checkbox("Show relevance scores", True)
        show_chunk_ids = st.checkbox("Show chunk IDs", True)
        
        # Diverse mode parameters (only show when diverse mode is enabled)
        if diverse_mode:
            st.markdown("**🔥 Diverse Mode Settings:**")
            diversity_threshold = st.slider(
                "Diversity threshold:", 
                0.1, 1.0, 0.7, 0.1,
                help="Higher values promote more diversity in retrieved results"
            )
            diversity_penalty = st.slider(
                "Diversity penalty:", 
                0.0, 1.0, 0.3, 0.1,
                help="Penalty applied to similar documents to promote diversity"
            )
        else:
            # Set default values when diverse mode is disabled
            diversity_threshold = 0.7
            diversity_penalty = 0.3
    
    # About section
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) demo searches through documents 
    to provide contextual answers to your questions.
    
    **Features:**
    - Multiple backend support
    - Real-time search
    - Source transparency
    - 🔥 **Diverse Mode**: FlowRL-inspired diversity enhancement
    """)

# Main content
st.markdown('<h1 class="main-header">🧠 Deep Researcher</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Document Search & Question Answering</p>', unsafe_allow_html=True)

# Backend status display with diverse mode indicator
current_backend = st.session_state.get('current_backend', 'lexrank')
current_backend_label = BACKEND_LABELS.get(current_backend, "Unknown")
diverse_indicator = ""

if 'diverse_mode' in locals() and diverse_mode:
    diverse_indicator = '<span class="diverse-mode-enabled">🔥 Diverse Mode</span>'

st.markdown(
    f'<div class="backend-status">Active Backend: {current_backend_label}{diverse_indicator}</div>', 
    unsafe_allow_html=True
)

# Main search interface
with st.container():
    # Search input section
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "🔍 What would you like to research?",
            placeholder="e.g., What are the latest developments in artificial intelligence?",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)

# Search results
if search_button:
    if not query.strip():
        st.warning("⚠️ Please enter a research question to get started.")
    else:
        # Create placeholders for dynamic updates
        status_placeholder = st.empty()
        results_placeholder = st.empty()
        
        with status_placeholder:
            diverse_status = "with Diverse Mode 🔥" if diverse_mode else ""
            with st.spinner(f"🔍 Searching through documents {diverse_status}..."):
                start_time = time.time()
                
                try:
                    # ENHANCED PAYLOAD WITH DIVERSE MODE
                    payload = {
                        "query": query.strip(), 
                        "top_k": max(1, min(top_k, 20))
                    }
                    
                    # Add word limit if specified
                    if word_limit > 0:
                        payload["max_words"] = max(10, min(word_limit, 1000))
                    
                    # Make API request with current backend choice
                    current_backend = st.session_state.get('current_backend', 'lexrank')
                    # Add backend to the payload
                    payload['backend'] = current_backend
                    resp = requests.post(
                        API, 
                        json=payload, 
                        timeout=timeout_duration,
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    processing_time = time.time() - start_time
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        
                        # Clear status and show results
                        status_placeholder.empty()
                        
                        with results_placeholder.container():
                            # Statistics with diverse mode info
                            sources_count = len(data.get("sources", []))
                            avg_score = sum(s.get('score', 0) for s in data.get("sources", [])) / max(sources_count, 1)
                            
                            # Calculate actual word count
                            answer = data.get("answer", "No answer provided")
                            actual_word_count = len(answer.split()) if answer else 0
                            
                            # Enhanced metrics with diverse mode
                            col1, col2, col3, col4, col5, col6 = st.columns(6)
                            with col1:
                                st.metric("⏱️ Response Time", f"{processing_time:.2f}s")
                            with col2:
                                st.metric("📄 Sources Found", sources_count)
                            with col3:
                                st.metric("📊 Avg. Relevance", f"{avg_score:.3f}")
                            with col4:
                                if word_limit > 0:
                                    st.metric("🎯 Word Limit", f"~{word_limit}")
                                else:
                                    st.metric("🎯 Word Limit", "None")
                            with col5:
                                st.metric("📝 Actual Words", actual_word_count)
                            with col6:
                                diversity_status = "🔥 Enabled" if diverse_mode else "❌ Disabled"
                                st.metric("🔥 Diverse Mode", diversity_status)

                            # Answer section
                            st.markdown("## 💡 Answer")
                            answer = data.get("answer", "No answer provided")
                            
                            # Use Streamlit's native container with styling
                            with st.container():
                                answer_style = "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)" if not diverse_mode else "linear-gradient(135deg, #fff5f5 0%, #ffeaa7 100%)"
                                border_color = "#667eea" if not diverse_mode else "#ff6b6b"
                                
                                st.markdown(
                                    f"""
                                    <div style="
                                        background: {answer_style};
                                        border-left: 5px solid {border_color};
                                        padding: 2rem;
                                        border-radius: 10px;
                                        margin: 1rem 0;
                                    ">
                                        <div style="color: #333; font-size: 1.1rem; line-height: 1.6;">
                                            {answer}
                                        </div>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                            
                            # Enhanced diverse mode information
                            if diverse_mode:
                                st.markdown("### 🔥 Diverse Mode Results")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.info(f"**Diversity Threshold:** {diversity_threshold}")
                                    
                                with col2:
                                    st.info(f"**Diversity Penalty:** {diversity_penalty}")
                                
                                # Calculate diversity metrics if available
                                sources = data.get("sources", [])  # Define sources here
                                if sources:
                                    score_std = pd.DataFrame(sources)["score"].std()
                                    st.success(f"**Score Diversity (Std Dev):** {score_std:.3f}")
                            
                            # Sources section
                            sources = data.get("sources", [])
                            if sources:
                                st.markdown("## 📚 Retrieved Sources")
                                diverse_caption = " (Enhanced with Diverse Mode 🔥)" if diverse_mode else ""
                                st.caption(f"Showing top {len(sources)} most relevant sources{diverse_caption}")
                                
                                for idx, source in enumerate(sources, 1):
                                    # Create a proper source card container
                                    with st.container():
                                        # Enhanced card styling for diverse mode
                                        card_border = "2px solid #dee2e6" if not diverse_mode else "2px solid #ff6b6b"
                                        
                                        st.markdown(f"""
                                        <div style="
                                            background: white; 
                                            border: {card_border}; 
                                            border-radius: 10px; 
                                            padding: 1.5rem; 
                                            margin: 1rem 0; 
                                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                                        ">
                                        """, unsafe_allow_html=True)
                                        
                                        # Header row with source info and score
                                        col1, col2 = st.columns([3, 1])
                                        
                                        with col1:
                                            st.markdown(f"**📖 Source {idx}: {source.get('doc_id', 'Unknown')}**")
                                            if show_chunk_ids:
                                                st.caption(f"Chunk ID: {source.get('chunk_id', 'N/A')}")
                                        
                                        with col2:
                                            if show_scores:
                                                score = source.get('score', 0)
                                                score_color = "#28a745" if score > 0.7 else "#ffc107" if score > 0.5 else "#dc3545"
                                                if diverse_mode:
                                                    score_color = "#ff6b6b" if score > 0.7 else "#4ecdc4" if score > 0.5 else "#fd79a8"
                                                
                                                st.markdown(
                                                    f'<span style="background:{score_color};color:white;padding:6px 12px;border-radius:20px;font-size:0.85rem;font-weight:600;display:inline-block;">Score: {score:.3f}</span>', 
                                                    unsafe_allow_html=True
                                                )
                                        
                                        st.markdown("---")
                                        
                                        # Source content with enhanced styling for diverse mode
                                        snippet = source.get('snippet', 'No content available')
                                        content_bg = "#f8f9fa" if not diverse_mode else "#fff5f5"
                                        content_border = "#667eea" if not diverse_mode else "#ff6b6b"
                                        
                                        st.markdown(
                                            f'<div style="color: #444; font-style: italic; line-height: 1.6; padding: 1rem; background: {content_bg}; border-radius: 6px; border-left: 4px solid {content_border};">{snippet}</div>', 
                                            unsafe_allow_html=True
                                        )
                                        
                                        # Close the card div
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        
                                        # Add spacing between cards
                                        st.markdown("<br>", unsafe_allow_html=True)
                                
                                # Add visualization section with enhanced diverse mode styling
                                st.markdown("## 📊 Source Relevance Analysis")
                                
                                # Create relevance bar chart
                                if sources:
                                    df = pd.DataFrame(sources)
                                    
                                    # Create the chart with diverse mode colors
                                    fig, ax = plt.subplots(figsize=(12, 7))
                                    
                                    if diverse_mode:
                                        # Use diverse mode color palette
                                        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f39c12', '#e74c3c', '#9b59b6'] * (len(df) // 6 + 1)
                                        colors = colors[:len(df)]
                                    else:
                                        # Use original color scheme
                                        colors = ['#28a745' if score > 0.7 else '#ffc107' if score > 0.5 else '#dc3545' 
                                                 for score in df["score"]]
                                    
                                    bars = ax.bar(df["chunk_id"].astype(str), df["score"], color=colors)
                                    
                                    ax.set_xlabel("Chunk ID", fontsize=12)
                                    ax.set_ylabel("Relevance Score", fontsize=12)
                                    
                                    title = "Retrieved Sources Relevance Scores"
                                    if diverse_mode:
                                        title += " (🔥Diverse Mode)"
                                    
                                    ax.set_title(title, fontsize=14, fontweight='bold')
                                    ax.set_ylim(0, max(1.0, df["score"].max() * 1.1))
                                    
                                    # Add value labels on bars
                                    for bar, score in zip(bars, df["score"]):
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                               f'{score:.3f}', ha='center', va='bottom', fontsize=10)
                                    
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                                
                                # Export functionality with diverse mode info
                                st.markdown("## 📥 Export Results")
                                
                                # Create enhanced export content with diverse mode info
                                export_md = f"""# Deep Researcher Results
                                
**Query:** {query}
**Backend:** {BACKEND_LABELS.get(backend_choice, backend_choice)}
**Diverse Mode:** {"🔥 Enabled" if diverse_mode else "❌ Disabled"}
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Processing Time:** {processing_time:.2f}s
**Word Limit:** {word_limit if word_limit > 0 else 'None'}
**Actual Words:** {actual_word_count}

## Configuration Details

- **Top K Results:** {top_k}
- **Diverse Mode:** {"Enabled" if diverse_mode else "Disabled"}
{f"- **Diversity Threshold:** {diversity_threshold}" if diverse_mode else ""}
{f"- **Diversity Penalty:** {diversity_penalty}" if diverse_mode else ""}
- **Show Scores:** {"Yes" if show_scores else "No"}
- **Show Chunk IDs:** {"Yes" if show_chunk_ids else "No"}

## Answer

{answer}

## Retrieved Sources

"""
                                for idx, source in enumerate(sources, 1):
                                    export_md += f"""### Source {idx}: {source.get('doc_id', 'Unknown')}
- **Chunk ID:** {source.get('chunk_id', 'N/A')}
- **Relevance Score:** {source.get('score', 0):.3f}
- **Content:** {source.get('snippet', 'No content available')}

"""
                                
                                # Add enhanced statistics with diverse mode metrics
                                export_md += f"""## Statistics

- **Total Sources:** {sources_count}
- **Average Relevance:** {avg_score:.3f}
- **Response Time:** {processing_time:.2f} seconds
- **Backend Used:** {BACKEND_LABELS.get(backend_choice, backend_choice)}
- **Diverse Mode:** {"🔥 Enabled" if diverse_mode else "❌ Disabled"}
{f"- **Score Diversity (Std Dev):** {pd.DataFrame(sources)['score'].std():.3f}" if sources else ""}
{f"- **Diversity Threshold:** {diversity_threshold}" if diverse_mode else ""}
{f"- **Diversity Penalty:** {diversity_penalty}" if diverse_mode else ""}

## Methodology

This search used {"diverse mode with FlowRL-inspired techniques" if diverse_mode else "standard retrieval"} to find the most relevant sources. {"Diverse mode enhances topic coverage by promoting variety in retrieved documents while maintaining relevance." if diverse_mode else "Standard mode prioritizes the highest relevance scores for optimal precision."}
"""
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    download_filename = f"deep_researcher_results_{'diverse_' if diverse_mode else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                                    st.download_button(
                                        "⬇️ Download Results (Markdown)",
                                        export_md,
                                        file_name=download_filename,
                                        mime="text/markdown",
                                        help="Download the complete search results as a Markdown file"
                                    )
                                
                                with col2:
                                    # Create enhanced CSV export with diverse mode info
                                    if sources:
                                        sources_df = pd.DataFrame(sources)
                                        # Add diverse mode metadata
                                        sources_df['diverse_mode'] = diverse_mode
                                        if diverse_mode:
                                            sources_df['diversity_threshold'] = diversity_threshold
                                            sources_df['diversity_penalty'] = diversity_penalty
                                        
                                        csv_data = sources_df.to_csv(index=False)
                                        csv_filename = f"deep_researcher_sources_{'diverse_' if diverse_mode else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                        st.download_button(
                                            "⬇️ Download Sources (CSV)",
                                            csv_data,
                                            file_name=csv_filename,
                                            mime="text/csv",
                                            help="Download source data as CSV for further analysis"
                                        )
                            else:
                                st.warning("📭 No relevant sources found for your query.")
                                if diverse_mode:
                                    st.info("💡 Try adjusting the diversity threshold or disabling Diverse Mode for potentially different results.")
                    
                    else:
                        status_placeholder.empty()
                        st.error(f"❌ API Error: {resp.status_code}")
                        with st.expander("Error Details"):
                            st.code(resp.text)
                        
                        if diverse_mode:
                            st.info("💡 If using Diverse Mode, ensure your backend API supports the 'diverse' parameter.")
                
                except requests.exceptions.Timeout:
                    status_placeholder.empty()
                    timeout_msg = f"⏰ Request timed out after {timeout_duration} seconds."
                    if diverse_mode:
                        timeout_msg += " Diverse Mode may require additional processing time."
                    st.error(timeout_msg)
                    st.info("💡 Try reducing the number of results, increasing the timeout, or disabling Diverse Mode.")
                
                except requests.exceptions.ConnectionError:
                    status_placeholder.empty()
                    st.error("🔌 Cannot connect to the API server. Please ensure the backend is running on http://127.0.0.1:8000")
                    if diverse_mode:
                        st.info("💡 Ensure your backend API supports Diverse Mode functionality.")
                
                except Exception as e:
                    status_placeholder.empty()
                    st.error(f"❌ Unexpected error: {str(e)}")
                    if diverse_mode:
                        st.info("💡 This error might be related to Diverse Mode. Try disabling it and searching again.")

# Enhanced footer with diverse mode info
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    footer_text = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if 'diverse_mode' in locals() and diverse_mode:
        footer_text += " • 🔥 Diverse Mode Active"
    
    st.markdown(
        f"<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        f"{footer_text}"
        f"</div>", 
        unsafe_allow_html=True
    )