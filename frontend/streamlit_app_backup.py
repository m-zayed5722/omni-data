import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import base64
from PIL import Image
import io

# Configuration
API_BASE_URL = "http://localhost:8002"

# Page config
st.set_page_config(
    page_title="GenAI Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.upload-section {
    border: 2px dashed #1f77b4;
    padding: 2rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    background-color: #f0f8ff;
}
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_file(file):
    """Upload CSV file to the API"""
    try:
        files = {"file": (file.name, file, "text/csv")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def get_dataset_summary():
    """Get dataset summary from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/dataset/summary")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def get_suggestions():
    """Get visualization suggestions from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/dataset/suggestions")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def send_query(query):
    """Send query to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"query": query}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending query: {str(e)}")
        return None

def get_conversation_history():
    """Get conversation history from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/history?limit=20")
        if response.status_code == 200:
            return response.json()["history"]
        else:
            return []
    except:
        return []

def display_plotly_from_base64(img_base64):
    """Display matplotlib image from base64"""
    if img_base64:
        try:
            img_data = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_data))
            st.image(img, use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 GenAI Data Visualization Dashboard</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("🚨 API is not running! Please start the FastAPI server first.")
        st.code("python -m uvicorn app.main_viz:app --reload --host 0.0.0.0 --port 8000")
        return
    
    st.success("✅ API is running!")
    
    # Sidebar
    st.sidebar.header("📁 Data Management")
    
    # File upload section
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=['csv'],
        help="Upload a CSV file to start creating visualizations"
    )
    
    if uploaded_file:
        if st.sidebar.button("📤 Upload File"):
            with st.spinner("Uploading file..."):
                upload_result = upload_file(uploaded_file)
                if upload_result:
                    st.sidebar.success(f"✅ Uploaded: {upload_result['filename']}")
                    st.sidebar.info(f"📊 {upload_result['rows']} rows × {upload_result['columns']} columns")
                    st.rerun()
    
    # Get dataset info
    dataset_summary = get_dataset_summary()
    
    if dataset_summary:
        # Sidebar dataset info
        st.sidebar.subheader("📊 Current Dataset")
        st.sidebar.write(f"**Rows:** {dataset_summary['total_rows']}")
        st.sidebar.write(f"**Columns:** {dataset_summary['total_columns']}")
        
        if dataset_summary['numeric_columns']:
            st.sidebar.write("**Numeric Columns:**")
            for col in dataset_summary['numeric_columns'][:5]:  # Show first 5
                st.sidebar.write(f"• {col}")
            if len(dataset_summary['numeric_columns']) > 5:
                st.sidebar.write(f"... and {len(dataset_summary['numeric_columns']) - 5} more")
        
        if dataset_summary['categorical_columns']:
            st.sidebar.write("**Categorical Columns:**")
            for col in dataset_summary['categorical_columns'][:5]:  # Show first 5
                st.sidebar.write(f"• {col}")
            if len(dataset_summary['categorical_columns']) > 5:
                st.sidebar.write(f"... and {len(dataset_summary['categorical_columns']) - 5} more")
        
        # Clear dataset button
        if st.sidebar.button("🗑️ Clear Dataset"):
            try:
                requests.delete(f"{API_BASE_URL}/dataset")
                st.sidebar.success("Dataset cleared!")
                st.rerun()
            except:
                st.sidebar.error("Error clearing dataset")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬 Natural Language Queries")
            
            # Query input
            query_input = st.text_area(
                "Ask questions about your data:",
                placeholder="Examples:\n• Show me a histogram of Age\n• Create a scatter plot of Salary vs Experience\n• Compare Sales across different Regions",
                height=100
            )
            
            if st.button("🚀 Generate Visualization", type="primary"):
                if query_input.strip():
                    with st.spinner("Analyzing your request..."):
                        result = send_query(query_input.strip())
                        if result:
                            st.markdown(f'<div class="chat-message"><strong>You:</strong> {query_input}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="chat-message"><strong>AI:</strong> {result["answer"]}</div>', unsafe_allow_html=True)
                            
                            # Display visualization if available
                            if result.get("visualization"):
                                st.subheader("📈 Generated Visualization")
                                viz_data = result["visualization"]
                                
                                if viz_data.get("plotly_json"):
                                    try:
                                        fig = go.Figure(viz_data["plotly_json"])
                                        st.plotly_chart(fig, use_container_width=True)
                                    except:
                                        pass
                                
                                if viz_data.get("image_base64"):
                                    display_plotly_from_base64(viz_data["image_base64"])
                else:
                    st.warning("Please enter a query!")
            
            # Conversation History
            st.subheader("📋 Recent Conversations")
            history = get_conversation_history()
            
            if history:
                for i, item in enumerate(reversed(history[-5:])):  # Show last 5
                    with st.expander(f"Query {len(history)-i}: {item['query'][:50]}..."):
                        st.write(f"**Query:** {item['query']}")
                        st.write(f"**Response:** {item['response']}")
                        st.write(f"**Time:** {item['timestamp']}")
                
                if st.button("🗑️ Clear History"):
                    try:
                        requests.delete(f"{API_BASE_URL}/history")
                        st.success("History cleared!")
                        st.rerun()
                    except:
                        st.error("Error clearing history")
            else:
                st.info("No conversation history yet. Start by asking a question!")
        
        with col2:
            st.header("💡 Suggestions")
            
            # Get and display suggestions
            suggestions = get_suggestions()
            if suggestions and suggestions.get("suggestions"):
                st.subheader("🎯 Recommended Visualizations")
                
                for i, suggestion in enumerate(suggestions["suggestions"][:6], 1):
                    with st.expander(f"{i}. {suggestion['type'].title()}: {suggestion['description'][:30]}..."):
                        st.write(f"**Type:** {suggestion['type'].title()}")
                        st.write(f"**Description:** {suggestion['description']}")
                        st.code(suggestion['example_query'])
                        
                        if st.button(f"Try this", key=f"suggestion_{i}"):
                            # Auto-fill the query
                            st.session_state.suggested_query = suggestion['example_query']
                            st.rerun()
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            
            quick_queries = [
                "What's in the dataset?",
                "Show me data summary",
                "Give me visualization suggestions",
                "What columns are available?"
            ]
            
            for query in quick_queries:
                if st.button(query, key=f"quick_{query}"):
                    with st.spinner("Processing..."):
                        result = send_query(query)
                        if result:
                            st.success("Query sent! Check the main area for results.")
                            st.rerun()
    
    else:
        # No dataset uploaded
        st.info("👆 Please upload a CSV dataset to get started!")
        
        st.markdown("""
        ### 🚀 How to use this dashboard:
        
        1. **Upload your CSV file** using the sidebar
        2. **Ask questions** in natural language about your data
        3. **Get instant visualizations** powered by AI
        
        ### 📊 Supported visualizations:
        - **Histograms** - Distribution of numeric columns
        - **Scatter plots** - Relationships between two numeric columns
        - **Bar charts** - Categorical vs numeric comparisons
        - **Box plots** - Distribution across categories
        
        ### 💡 Example queries:
        - "Show me a histogram of sales"
        - "Create a scatter plot of price vs quantity"
        - "Compare revenue across different regions"
        - "What's the distribution of age by department?"
        """)

if __name__ == "__main__":
    main()