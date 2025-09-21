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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import re

# Configuration - Update this for Streamlit Cloud deployment
API_BASE_URL = "http://localhost:8002"  # Change to your deployed backend URL when deploying

# Page config
st.set_page_config(
    page_title="GenAI Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache expensive functions
@st.cache_data
def load_css():
    """Cache CSS to avoid reloading"""
    return """
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
.viz-container {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.error-message {
    background-color: #ffe6e6;
    border-left: 4px solid #ff4444;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.success-message {
    background-color: #e6ffe6;
    border-left: 4px solid #44ff44;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.sidebar-content {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.template-button {
    width: 100%;
    margin: 0.25rem 0;
    padding: 0.5rem;
    background-color: #f0f8ff;
    border: 1px solid #1f77b4;
    border-radius: 5px;
    text-align: left;
}
.template-button:hover {
    background-color: #e6f3ff;
}
.query-history-item {
    background-color: #f8f9fa;
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-radius: 5px;
    border-left: 3px solid #1f77b4;
    cursor: pointer;
}
.query-history-item:hover {
    background-color: #e9ecef;
}
</style>
"""

# Apply CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

# Enhanced Smart Parser Functions
def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def normalize_column_name(col_name: str) -> str:
    """Normalize column name for better matching"""
    return re.sub(r'[^a-zA-Z0-9]', '', col_name.lower())

def has_synonym_match(query_word: str, col_name: str, synonyms: Dict[str, List[str]]) -> bool:
    """Check if query word matches any synonym for the column"""
    normalized_col = normalize_column_name(col_name)
    normalized_query = normalize_column_name(query_word)
    
    # Check direct synonym matches
    for key, syn_list in synonyms.items():
        if normalized_col in [normalize_column_name(s) for s in syn_list]:
            if normalized_query in [normalize_column_name(s) for s in syn_list]:
                return True
    
    return False

def is_acronym_match(query_word: str, col_name: str) -> bool:
    """Check if query word could be an acronym of column name"""
    if len(query_word) < 2:
        return False
    
    # Extract first letters of words in column name
    words = re.findall(r'[A-Za-z]+', col_name)
    if len(words) >= len(query_word):
        acronym = ''.join([word[0].lower() for word in words])
        return acronym.startswith(query_word.lower())
    
    return False

def find_best_column_match(query: str, columns: List[str], threshold: float = 0.6) -> List[str]:
    """Find best matching columns using fuzzy matching with synonyms and advanced techniques"""
    
    # Define synonyms for common terms
    synonyms = {
        'sales': ['revenue', 'income', 'earnings', 'sales', 'profit', 'turnover'],
        'date': ['time', 'timestamp', 'created_at', 'updated_at', 'date', 'datetime'],
        'name': ['title', 'label', 'description', 'name', 'id', 'identifier'],
        'amount': ['value', 'price', 'cost', 'sum', 'total', 'amount', 'quantity'],
        'category': ['type', 'class', 'group', 'segment', 'category', 'classification'],
        'customer': ['client', 'user', 'buyer', 'customer', 'account'],
        'product': ['item', 'goods', 'service', 'product', 'merchandise'],
        'region': ['location', 'area', 'territory', 'region', 'zone', 'country', 'city']
    }
    
    query_words = re.findall(r'\b\w+\b', query.lower())
    matched_columns = []
    
    for col in columns:
        col_normalized = normalize_column_name(col)
        col_score = 0
        
        for query_word in query_words:
            query_normalized = normalize_column_name(query_word)
            
            # Exact match (highest score)
            if query_normalized == col_normalized:
                col_score += 1.0
                continue
            
            # Substring match
            if query_normalized in col_normalized or col_normalized in query_normalized:
                col_score += 0.8
                continue
            
            # Synonym match
            if has_synonym_match(query_word, col, synonyms):
                col_score += 0.7
                continue
            
            # Acronym match
            if is_acronym_match(query_word, col):
                col_score += 0.6
                continue
            
            # Fuzzy match using Levenshtein distance
            max_len = max(len(query_normalized), len(col_normalized))
            if max_len > 0:
                distance = levenshtein_distance(query_normalized, col_normalized)
                similarity = 1 - (distance / max_len)
                if similarity >= threshold:
                    col_score += similarity * 0.5
        
        # Normalize score by number of query words
        if len(query_words) > 0:
            col_score = col_score / len(query_words)
        
        if col_score >= threshold:
            matched_columns.append((col, col_score))
    
    # Sort by score and return column names
    matched_columns.sort(key=lambda x: x[1], reverse=True)
    return [col for col, score in matched_columns[:5]]  # Return top 5 matches

# Enhanced pattern recognition with more visualization types
@st.cache_data
def get_enhanced_patterns():
    """Get enhanced patterns for Smart Parser"""
    return {
        # Chart type patterns
        'scatter': [
            r'scatter\s+plot',
            r'scatter\s+chart',
            r'plot\s+.*\s+vs\s+.*',
            r'.*\s+vs\s+.*\s+scatter',
            r'correlation\s+between',
            r'relationship\s+between'
        ],
        'line': [
            r'line\s+chart',
            r'line\s+plot',
            r'trend\s+over\s+time',
            r'time\s+series',
            r'.*\s+over\s+time',
            r'trend\s+analysis'
        ],
        'bar': [
            r'bar\s+chart',
            r'bar\s+plot',
            r'.*\s+by\s+.*',
            r'count\s+of\s+.*',
            r'sum\s+of\s+.*\s+by',
            r'total\s+.*\s+by'
        ],
        'grouped_bar': [
            r'grouped\s+bar',
            r'group\s+by\s+.*\s+and\s+.*',
            r'.*\s+by\s+.*\s+and\s+.*',
            r'compare\s+.*\s+across\s+.*',
            r'breakdown\s+by\s+.*\s+and'
        ],
        'histogram': [
            r'histogram',
            r'distribution\s+of',
            r'frequency\s+of',
            r'.*\s+distribution'
        ],
        'pie': [
            r'pie\s+chart',
            r'proportion\s+of',
            r'percentage\s+of',
            r'share\s+of\s+.*'
        ],
        'box': [
            r'box\s+plot',
            r'boxplot',
            r'outliers\s+in',
            r'distribution\s+summary'
        ],
        'heatmap': [
            r'heatmap',
            r'correlation\s+matrix',
            r'heat\s+map'
        ],
        'timeseries': [
            r'time\s+series',
            r'timeseries',
            r'.*\s+over\s+time',
            r'temporal\s+analysis',
            r'trend\s+over\s+.*'
        ],
        'summary_stats': [
            r'summary\s+statistics',
            r'descriptive\s+stats',
            r'statistical\s+summary',
            r'basic\s+stats',
            r'data\s+summary'
        ],
        # Aggregation patterns
        'sum': [r'total\s+.*', r'sum\s+of\s+.*'],
        'count': [r'count\s+.*', r'number\s+of\s+.*'],
        'avg': [r'average\s+.*', r'mean\s+.*'],
        'max': [r'maximum\s+.*', r'highest\s+.*', r'max\s+.*'],
        'min': [r'minimum\s+.*', r'lowest\s+.*', r'min\s+.*'],
        
        # Grouping patterns
        'group_by': [r'.*\s+by\s+.*', r'group\s+by\s+.*', r'breakdown\s+by\s+.*']
    }

def smart_parse_query(query: str, columns: List[str]) -> Dict[str, Any]:
    """Enhanced smart parser with fuzzy matching and advanced patterns"""
    patterns = get_enhanced_patterns()
    query_lower = query.lower()
    
    # Find best matching columns using fuzzy matching
    matched_columns = find_best_column_match(query, columns)
    
    result = {
        'chart_type': 'bar',  # default
        'columns': matched_columns[:2] if matched_columns else columns[:2],
        'aggregation': None,
        'group_by': None,
        'confidence': 0.5
    }
    
    # Determine chart type with enhanced patterns
    chart_confidence = {}
    for chart_type, pattern_list in patterns.items():
        if chart_type in ['sum', 'count', 'avg', 'max', 'min', 'group_by']:
            continue
            
        for pattern in pattern_list:
            if re.search(pattern, query_lower):
                chart_confidence[chart_type] = chart_confidence.get(chart_type, 0) + 1
    
    if chart_confidence:
        result['chart_type'] = max(chart_confidence.items(), key=lambda x: x[1])[0]
        result['confidence'] = min(max(chart_confidence.values()) * 0.2, 1.0)
    
    # Handle special chart types
    if result['chart_type'] == 'grouped_bar':
        # Try to find two grouping columns
        group_words = re.findall(r'by\s+(\w+)(?:\s+and\s+(\w+))?', query_lower)
        if group_words and group_words[0]:
            group_cols = [col for col in matched_columns if any(g and g.lower() in col.lower() for g in group_words[0])]
            result['columns'] = group_cols[:2] if len(group_cols) >= 2 else matched_columns[:2]
    
    elif result['chart_type'] == 'timeseries':
        # Try to find time column
        time_cols = [col for col in columns if any(time_word in col.lower() 
                    for time_word in ['date', 'time', 'timestamp', 'created', 'updated'])]
        if time_cols and matched_columns:
            result['columns'] = [time_cols[0], matched_columns[0]]
    
    elif result['chart_type'] == 'summary_stats':
        # For summary stats, we want numeric columns
        numeric_cols = [col for col in matched_columns if col in columns]
        result['columns'] = numeric_cols[:3] if numeric_cols else matched_columns[:3]
    
    # Detect aggregation
    for agg_type in ['sum', 'count', 'avg', 'max', 'min']:
        for pattern in patterns[agg_type]:
            if re.search(pattern, query_lower):
                result['aggregation'] = agg_type
                break
    
    # Detect grouping
    for pattern in patterns['group_by']:
        if re.search(pattern, query_lower):
            group_match = re.search(r'by\s+(\w+)', query_lower)
            if group_match:
                group_col = find_best_column_match(group_match.group(1), columns)
                if group_col:
                    result['group_by'] = group_col[0]
            break
    
    return result

def make_api_request(endpoint: str, data: dict) -> dict:
    """Make API request with error handling"""
    try:
        response = requests.post(f"{API_BASE_URL}/{endpoint}", json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return {"error": str(e)}

def upload_data():
    """Handle file upload with enhanced error handling"""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📁 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file to get started with data visualization"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Store in session state
            st.session_state.uploaded_file = uploaded_file
            st.session_state.df = df
            
            # Show success message
            st.markdown('<div class="success-message">✅ File uploaded successfully!</div>', unsafe_allow_html=True)
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", f"{len(df.columns):,}")
            with col3:
                st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Show column info
            with st.expander("📊 Dataset Overview", expanded=False):
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Null %': ((df.isnull().sum() / len(df)) * 100).round(2)
                })
                st.dataframe(col_info, use_container_width=True)
                
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
            
            return df
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    st.markdown('</div>', unsafe_allow_html=True)
    return None

def render_query_templates():
    """Render quick query templates"""
    st.markdown("### 🚀 Quick Templates")
    
    if st.session_state.df is not None:
        columns = list(st.session_state.df.columns)
        
        # Create templates based on available columns
        templates = []
        
        # Find potential numeric columns
        numeric_cols = [col for col in columns if st.session_state.df[col].dtype in ['int64', 'float64']]
        
        # Find potential categorical columns
        categorical_cols = [col for col in columns if st.session_state.df[col].dtype == 'object']
        
        # Find potential date columns
        date_cols = [col for col in columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if len(numeric_cols) >= 2:
            templates.extend([
                f"Create scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}",
                f"Show correlation between {numeric_cols[0]} and {numeric_cols[1]}",
                f"Plot line chart of {numeric_cols[0]} over time" if date_cols else f"Create histogram of {numeric_cols[0]}"
            ])
        
        if numeric_cols and categorical_cols:
            templates.extend([
                f"Show {numeric_cols[0]} by {categorical_cols[0]} as bar chart",
                f"Create pie chart of {categorical_cols[0]}",
                f"Display {numeric_cols[0]} grouped by {categorical_cols[0]}"
            ])
        
        if len(columns) >= 2:
            templates.extend([
                f"Show summary statistics for all columns",
                f"Create heatmap of correlations",
                f"Display data distribution analysis"
            ])
        
        # Render templates as a form to handle button clicks properly
        with st.form("templates_form", clear_on_submit=False):
            for i, template in enumerate(templates[:6]):  # Show up to 6 templates
                if st.form_submit_button(template, key=f"template_{i}"):
                    st.session_state.current_query = template
                    st.rerun()
    else:
        st.info("Upload data to see available templates")

def render_query_history():
    """Render query history in sidebar"""
    st.markdown("### 📝 Query History")
    
    if st.session_state.query_history:
        for i, query in enumerate(reversed(st.session_state.query_history[-10:])):  # Show last 10
            if st.button(f"📊 {query[:50]}{'...' if len(query) > 50 else ''}", key=f"history_{i}"):
                st.session_state.current_query = query
                st.rerun()
    else:
        st.info("No queries yet")

def chat_interface():
    """Enhanced chat interface with smart parsing"""
    st.markdown('<div class="viz-container">', unsafe_allow_html=True)
    st.subheader("🤖 AI Data Visualization Assistant")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first to start creating visualizations.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Use form to handle query submission properly
    with st.form("chat_form", clear_on_submit=True):
        # Query input
        query = st.text_area(
            "What would you like to visualize?",
            value=st.session_state.current_query,
            placeholder="E.g., 'Create a scatter plot of sales vs profit by region'",
            height=100,
            key="query_input"
        )
        
        # Submit button
        submit_button = st.form_submit_button("🎨 Create Visualization", type="primary")
        
        # Clear query from session state after using it
        if st.session_state.current_query:
            st.session_state.current_query = ""
    
    if submit_button and query.strip():
        # Add to query history
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)
        
        # Show user query
        with st.chat_message("user"):
            st.write(query)
        
        # Process with smart parser
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing your request..."):
                
                # Smart parse the query
                columns = list(st.session_state.df.columns)
                parsed_result = smart_parse_query(query, columns)
                
                # Show parsing results
                with st.expander("🔍 Query Analysis", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Chart Type:** {parsed_result['chart_type']}")
                        st.write(f"**Confidence:** {parsed_result['confidence']:.1%}")
                    with col2:
                        st.write(f"**Columns:** {', '.join(parsed_result['columns'])}")
                        if parsed_result['group_by']:
                            st.write(f"**Group By:** {parsed_result['group_by']}")
                
                # Prepare API request
                chart_request = {
                    "query": query,
                    "chart_type": parsed_result['chart_type'],
                    "columns": parsed_result['columns'][:2],  # Limit to 2 for most charts
                    "data": st.session_state.df.to_dict('records')
                }
                
                # Add grouping if detected
                if parsed_result['group_by']:
                    chart_request['group_by'] = parsed_result['group_by']
                
                # Add aggregation if detected
                if parsed_result['aggregation']:
                    chart_request['aggregation'] = parsed_result['aggregation']
                
                # Special handling for different chart types
                if parsed_result['chart_type'] == 'summary_stats':
                    chart_request['columns'] = parsed_result['columns'][:5]  # Allow more columns for summary
                
                # Make API request
                result = make_api_request("chat", chart_request)
                
                if "error" not in result:
                    # Display AI response
                    if "response" in result:
                        st.write(result["response"])
                    
                    # Display visualization
                    if "visualization" in result:
                        viz_data = result["visualization"]
                        
                        try:
                            if viz_data["type"] == "plotly":
                                fig = go.Figure(viz_data["figure"])
                                st.plotly_chart(fig, use_container_width=True)
                            elif viz_data["type"] == "matplotlib":
                                img_data = base64.b64decode(viz_data["image"])
                                img = Image.open(io.BytesIO(img_data))
                                st.image(img, use_container_width=True)
                            elif viz_data["type"] == "table":
                                df_viz = pd.DataFrame(viz_data["data"])
                                st.dataframe(df_viz, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying visualization: {str(e)}")
                    
                    # Show insights if available
                    if "insights" in result and result["insights"]:
                        st.subheader("💡 Key Insights")
                        for insight in result["insights"]:
                            st.write(f"• {insight}")
                else:
                    st.error(f"Failed to create visualization: {result.get('error', 'Unknown error')}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def direct_visualization():
    """Direct visualization interface with enhanced options"""
    st.markdown('<div class="viz-container">', unsafe_allow_html=True)
    st.subheader("🎯 Direct Visualization")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    df = st.session_state.df
    columns = list(df.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["scatter", "line", "bar", "grouped_bar", "histogram", "pie", "box", "heatmap", "timeseries", "summary_stats"],
            help="Select the type of visualization you want to create"
        )
    
    with col2:
        if chart_type in ["scatter", "line", "grouped_bar", "timeseries"]:
            selected_columns = st.multiselect("Select Columns", columns, default=columns[:2])
        elif chart_type in ["histogram", "pie", "box"]:
            selected_columns = st.multiselect("Select Columns", columns, default=columns[:1])
        elif chart_type == "summary_stats":
            selected_columns = st.multiselect("Select Columns", columns, default=columns[:5])
        else:
            selected_columns = st.multiselect("Select Columns", columns, default=columns[:2])
    
    # Additional options based on chart type
    group_by = None
    aggregation = None
    
    if chart_type in ["bar", "grouped_bar"] and len(selected_columns) >= 1:
        group_by = st.selectbox("Group By (Optional)", [None] + columns, key="group_by_select")
        aggregation = st.selectbox("Aggregation", ["sum", "count", "avg", "max", "min"], index=1)
    
    if st.button("Create Visualization", type="primary"):
        if selected_columns:
            # Prepare request
            viz_request = {
                "chart_type": chart_type,
                "columns": selected_columns,
                "data": df.to_dict('records')
            }
            
            if group_by:
                viz_request["group_by"] = group_by
            if aggregation:
                viz_request["aggregation"] = aggregation
            
            # Make API request
            with st.spinner("Creating visualization..."):
                result = make_api_request("visualize", viz_request)
                
                if "error" not in result and "visualization" in result:
                    viz_data = result["visualization"]
                    
                    try:
                        if viz_data["type"] == "plotly":
                            fig = go.Figure(viz_data["figure"])
                            st.plotly_chart(fig, use_container_width=True)
                        elif viz_data["type"] == "matplotlib":
                            img_data = base64.b64decode(viz_data["image"])
                            img = Image.open(io.BytesIO(img_data))
                            st.image(img, use_container_width=True)
                        elif viz_data["type"] == "table":
                            df_viz = pd.DataFrame(viz_data["data"])
                            st.dataframe(df_viz, use_container_width=True)
                        
                        # Show insights if available
                        if "insights" in result and result["insights"]:
                            st.subheader("💡 Insights")
                            for insight in result["insights"]:
                                st.write(f"• {insight}")
                                
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
                else:
                    st.error(f"Failed to create visualization: {result.get('error', 'Unknown error')}")
        else:
            st.warning("Please select at least one column.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def ai_agent_interface():
    """AI Agent interface for complex analysis"""
    st.markdown('<div class="viz-container">', unsafe_allow_html=True)
    st.subheader("🤖 AI Agent Analysis")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.write("Let the AI Agent perform comprehensive analysis of your data using advanced tools.")
    
    analysis_type = st.selectbox(
        "Analysis Type",
        [
            "Full Data Overview",
            "Statistical Analysis",
            "Correlation Analysis",
            "Outlier Detection",
            "Pattern Discovery",
            "Custom Query"
        ]
    )
    
    if analysis_type == "Custom Query":
        custom_query = st.text_area(
            "Enter your analysis request:",
            placeholder="E.g., 'Find the top 5 customers by revenue and analyze their buying patterns'"
        )
    else:
        custom_query = None
    
    if st.button("🔍 Start Analysis", type="primary"):
        with st.spinner("AI Agent is analyzing your data..."):
            # Prepare agent request
            agent_request = {
                "task": analysis_type.lower().replace(" ", "_"),
                "data": st.session_state.df.to_dict('records'),
                "columns": list(st.session_state.df.columns)
            }
            
            if custom_query:
                agent_request["query"] = custom_query
            
            # Make API request to AI agent
            result = make_api_request("agent/analyze", agent_request)
            
            if "error" not in result:
                # Display agent response
                if "analysis" in result:
                    st.markdown("### 📊 Analysis Results")
                    st.write(result["analysis"])
                
                # Display any visualizations
                if "visualizations" in result:
                    st.markdown("### 📈 Generated Visualizations")
                    for i, viz in enumerate(result["visualizations"]):
                        try:
                            if viz["type"] == "plotly":
                                fig = go.Figure(viz["figure"])
                                st.plotly_chart(fig, use_container_width=True)
                            elif viz["type"] == "matplotlib":
                                img_data = base64.b64decode(viz["image"])
                                img = Image.open(io.BytesIO(img_data))
                                st.image(img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying visualization {i+1}: {str(e)}")
                
                # Display insights
                if "insights" in result and result["insights"]:
                    st.markdown("### 💡 Key Insights")
                    for insight in result["insights"]:
                        st.write(f"• {insight}")
                
                # Display recommendations
                if "recommendations" in result and result["recommendations"]:
                    st.markdown("### 🎯 Recommendations")
                    for rec in result["recommendations"]:
                        st.write(f"• {rec}")
                        
            else:
                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🎯 Omni-Data: GenAI Data Visualization Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Transform your data into insights with AI-powered visualizations")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # File upload section
        df = upload_data()
        
        if df is not None:
            # Quick templates
            render_query_templates()
            
            st.markdown("---")
            
            # Query history
            render_query_history()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("**🚀 Omni-Data Dashboard**")
        st.markdown("Powered by GenAI & LangChain")
    
    # Main content tabs
    if st.session_state.df is not None:
        tab1, tab2, tab3 = st.tabs(["💬 Smart Chat", "🎯 Direct Viz", "🤖 AI Agent"])
        
        with tab1:
            chat_interface()
        
        with tab2:
            direct_visualization()
        
        with tab3:
            ai_agent_interface()
    else:
        st.info("👆 Please upload a CSV file using the sidebar to get started!")
        
        # Show sample data section
        st.markdown("### 📋 Sample Data")
        st.write("Don't have data? Try these sample datasets:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Sales Data"):
                # Create sample sales data
                sample_data = pd.DataFrame({
                    'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
                    'Sales': np.random.normal(1000, 200, 100),
                    'Profit': np.random.normal(200, 50, 100),
                    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                    'Product': np.random.choice(['A', 'B', 'C', 'D'], 100)
                })
                st.session_state.df = sample_data
                st.rerun()
        
        with col2:
            if st.button("👥 Customer Data"):
                # Create sample customer data
                sample_data = pd.DataFrame({
                    'Customer_ID': range(1, 101),
                    'Age': np.random.randint(18, 80, 100),
                    'Income': np.random.normal(50000, 15000, 100),
                    'Spending': np.random.normal(2000, 500, 100),
                    'Category': np.random.choice(['Premium', 'Regular', 'Budget'], 100)
                })
                st.session_state.df = sample_data
                st.rerun()
        
        with col3:
            if st.button("📈 Stock Data"):
                # Create sample stock data
                sample_data = pd.DataFrame({
                    'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
                    'Open': np.random.normal(100, 10, 100),
                    'High': np.random.normal(105, 10, 100),
                    'Low': np.random.normal(95, 10, 100),
                    'Close': np.random.normal(100, 10, 100),
                    'Volume': np.random.randint(1000000, 5000000, 100)
                })
                st.session_state.df = sample_data
                st.rerun()

if __name__ == "__main__":
    main()