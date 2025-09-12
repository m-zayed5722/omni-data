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
.mode-toggle {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px solid #dee2e6;
}
.manual-controls {
    background-color: #fff3cd;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
}
.interpretation-box {
    background-color: #d1ecf1;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #17a2b8;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

class VisualizationEngine:
    """Enhanced visualization engine with manual mode support"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def plot_histogram(self, column: str, bins: int = 30) -> go.Figure:
        """Create histogram"""
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' must be numeric for histogram")
        
        fig = px.histogram(
            self.df, 
            x=column, 
            nbins=bins,
            title=f'Distribution of {column}',
            template="plotly_white"
        )
        fig.update_layout(
            showlegend=False,
            height=500
        )
        return fig
    
    def plot_scatter(self, x: str, y: str, color: str = None, size: str = None) -> go.Figure:
        """Create scatter plot"""
        if x not in self.numeric_columns or y not in self.numeric_columns:
            raise ValueError("Both X and Y columns must be numeric for scatter plot")
        
        fig = px.scatter(
            self.df,
            x=x,
            y=y,
            color=color if color and color in self.df.columns else None,
            size=size if size and size in self.numeric_columns else None,
            title=f'{y} vs {x}',
            template="plotly_white"
        )
        fig.update_layout(height=500)
        return fig
    
    def plot_boxplot(self, column: str, by: str = None) -> go.Figure:
        """Create box plot"""
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' must be numeric for box plot")
        
        if by and by in self.categorical_columns:
            fig = px.box(
                self.df,
                x=by,
                y=column,
                title=f'Box Plot of {column} by {by}',
                template="plotly_white"
            )
        else:
            fig = px.box(
                self.df,
                y=column,
                title=f'Box Plot of {column}',
                template="plotly_white"
            )
        
        fig.update_layout(height=500)
        return fig
    
    def plot_correlation_heatmap(self) -> go.Figure:
        """Create correlation heatmap"""
        if len(self.numeric_columns) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation heatmap")
        
        corr_matrix = self.df[self.numeric_columns].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            template="plotly_white",
            color_continuous_scale="RdBu"
        )
        fig.update_layout(height=600)
        return fig
    
    def plot_bar(self, column: str, group_by: str = None, agg_func: str = 'count') -> go.Figure:
        """Create bar chart"""
        if group_by and group_by not in self.categorical_columns:
            raise ValueError("Group by column must be categorical")
        
        if agg_func == 'count':
            if group_by:
                plot_data = self.df.groupby([column, group_by]).size().reset_index(name='count')
                fig = px.bar(
                    plot_data,
                    x=column,
                    y='count',
                    color=group_by,
                    title=f'Count of {column} by {group_by}',
                    template="plotly_white"
                )
            else:
                value_counts = self.df[column].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f'Count of {column}',
                    template="plotly_white"
                )
        else:
            # For other aggregation functions
            if group_by and column in self.numeric_columns:
                if agg_func == 'mean':
                    plot_data = self.df.groupby(group_by)[column].mean().reset_index()
                elif agg_func == 'sum':
                    plot_data = self.df.groupby(group_by)[column].sum().reset_index()
                else:
                    plot_data = self.df.groupby(group_by)[column].mean().reset_index()
                
                fig = px.bar(
                    plot_data,
                    x=group_by,
                    y=column,
                    title=f'{agg_func.title()} of {column} by {group_by}',
                    template="plotly_white"
                )
        
        fig.update_layout(height=500)
        return fig
    
    def plot_pie(self, column: str, top_n: int = 10) -> go.Figure:
        """Create pie chart"""
        value_counts = self.df[column].value_counts().head(top_n)
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f'Distribution of {column}',
            template="plotly_white"
        )
        fig.update_layout(height=500)
        return fig

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception:
        return False, None

def upload_file(uploaded_file):
    """Upload file to the API"""
    try:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"Upload failed with status {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def get_dataset_summary():
    """Get dataset summary from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/dataset/summary")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def send_natural_language_query(query: str):
    """Send natural language query to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"query": query},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Query failed with status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def parse_natural_language(query: str, columns: List[str]) -> Dict[str, Any]:
    """Simple parser for natural language queries with fallback patterns"""
    query_lower = query.lower()
    
    # Histogram patterns
    histogram_patterns = [
        r"histogram.*of\s+(\w+)",
        r"distribution.*of\s+(\w+)",
        r"show.*(\w+).*distribution"
    ]
    
    # Scatter plot patterns
    scatter_patterns = [
        r"scatter.*(\w+).*vs.*(\w+)",
        r"plot.*(\w+).*against.*(\w+)",
        r"(\w+).*vs.*(\w+).*scatter"
    ]
    
    # Correlation patterns
    correlation_patterns = [
        r"correlation.*heatmap",
        r"heatmap.*correlation",
        r"correlations"
    ]
    
    # Box plot patterns
    boxplot_patterns = [
        r"box.*plot.*of\s+(\w+)",
        r"boxplot.*(\w+)"
    ]
    
    # Bar chart patterns
    bar_patterns = [
        r"bar.*chart.*of\s+(\w+)",
        r"count.*of\s+(\w+)"
    ]
    
    # Try to match patterns
    for pattern in histogram_patterns:
        match = re.search(pattern, query_lower)
        if match:
            column = match.group(1)
            if column in [c.lower() for c in columns]:
                actual_column = next(c for c in columns if c.lower() == column)
                return {
                    "type": "histogram",
                    "column": actual_column,
                    "interpretation": f"→ Histogram of {actual_column}"
                }
    
    for pattern in scatter_patterns:
        match = re.search(pattern, query_lower)
        if match:
            col1, col2 = match.groups()
            actual_col1 = next((c for c in columns if c.lower() == col1), None)
            actual_col2 = next((c for c in columns if c.lower() == col2), None)
            if actual_col1 and actual_col2:
                return {
                    "type": "scatter",
                    "x": actual_col1,
                    "y": actual_col2,
                    "interpretation": f"→ Scatter plot of {actual_col1} vs {actual_col2}"
                }
    
    for pattern in correlation_patterns:
        if re.search(pattern, query_lower):
            return {
                "type": "correlation",
                "interpretation": "→ Correlation heatmap of all numeric columns"
            }
    
    for pattern in boxplot_patterns:
        match = re.search(pattern, query_lower)
        if match:
            column = match.group(1)
            if column in [c.lower() for c in columns]:
                actual_column = next(c for c in columns if c.lower() == column)
                return {
                    "type": "boxplot",
                    "column": actual_column,
                    "interpretation": f"→ Box plot of {actual_column}"
                }
    
    for pattern in bar_patterns:
        match = re.search(pattern, query_lower)
        if match:
            column = match.group(1)
            if column in [c.lower() for c in columns]:
                actual_column = next(c for c in columns if c.lower() == column)
                return {
                    "type": "bar",
                    "column": actual_column,
                    "interpretation": f"→ Bar chart of {actual_column}"
                }
    
    return {"type": "unknown", "interpretation": "❓ Could not interpret query"}

def display_plotly_chart(fig, chart_title: str = "Visualization"):
    """Display plotly chart with download option"""
    st.plotly_chart(fig, use_container_width=True)
    
    # Create download button
    img_bytes = fig.to_image(format="png", width=1200, height=800)
    st.download_button(
        label="📥 Download Chart as PNG",
        data=img_bytes,
        file_name=f"{chart_title.lower().replace(' ', '_')}.png",
        mime="image/png"
    )

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 GenAI Data Visualization Dashboard</h1>', unsafe_allow_html=True)
    
    # Check API health
    is_healthy, health_data = check_api_health()
    
    if not is_healthy:
        st.error("🚨 API is not running! Please start the FastAPI server first.")
        st.code("python start_server.py")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Dashboard Controls")
        
        # Mode Selection
        st.markdown('<div class="mode-toggle">', unsafe_allow_html=True)
        st.markdown("### 🔄 Interaction Mode")
        interaction_mode = st.radio(
            "Choose your preferred interaction mode:",
            ["🗣️ Natural Language", "🎛️ Manual Mode"],
            index=0,
            help="Natural Language: Type what you want to visualize\nManual Mode: Use UI controls to build visualizations"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # File Upload Section
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your dataset to start creating visualizations"
        )
        
        # Dataset Info
        if uploaded_file is not None:
            # Upload file
            with st.spinner("Uploading file..."):
                success, upload_result = upload_file(uploaded_file)
            
            if success:
                st.success(f"✅ File uploaded successfully!")
                st.info(f"📊 **{upload_result['rows']}** rows × **{upload_result['columns']}** columns")
                
                # Get dataset summary
                dataset_summary = get_dataset_summary()
                if dataset_summary:
                    st.markdown("#### 📈 Dataset Overview")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Numeric Columns", len(dataset_summary['numeric_columns']))
                    with col2:
                        st.metric("Categorical Columns", len(dataset_summary['categorical_columns']))
                    
                    with st.expander("📋 Column Details"):
                        if dataset_summary['numeric_columns']:
                            st.markdown("**Numeric:** " + ", ".join(dataset_summary['numeric_columns']))
                        if dataset_summary['categorical_columns']:
                            st.markdown("**Categorical:** " + ", ".join(dataset_summary['categorical_columns']))
            else:
                st.error(f"❌ Upload failed: {upload_result.get('error', 'Unknown error')}")
    
    # Main content area
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to get started!")
        return
    
    # Get current dataset info
    dataset_summary = get_dataset_summary()
    if not dataset_summary:
        st.error("❌ Could not retrieve dataset information")
        return
    
    # Create visualization engine
    try:
        # For manual mode, we need the actual dataframe
        if interaction_mode == "🎛️ Manual Mode":
            # Read the uploaded file directly for manual mode
            df = pd.read_csv(uploaded_file)
            viz_engine = VisualizationEngine(df)
        else:
            viz_engine = None
    except Exception as e:
        st.error(f"❌ Error processing dataset: {e}")
        return
    
    # Mode-specific interface
    if interaction_mode == "🗣️ Natural Language":
        st.markdown("## 💬 Natural Language Interface")
        st.markdown("Ask me to create visualizations using natural language!")
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            - "Show me a histogram of age"
            - "Create a scatter plot of price vs rating"
            - "Plot correlation heatmap"
            - "Box plot of salary by department"
            - "Bar chart of sales by region"
            """)
        
        # Natural language input
        user_query = st.text_area(
            "What would you like to visualize?",
            height=100,
            placeholder="e.g., Create a scatter plot of age vs salary",
            help="Describe the visualization you want in natural language"
        )
        
        if st.button("🚀 Generate Visualization", type="primary"):
            if not user_query.strip():
                st.warning("Please enter a query!")
                return
            
            # Try simple parsing first
            all_columns = dataset_summary['numeric_columns'] + dataset_summary['categorical_columns']
            parsed_result = parse_natural_language(user_query, all_columns)
            
            # Show interpretation
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            st.markdown(f"**Query Interpretation:** {parsed_result['interpretation']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if parsed_result['type'] != 'unknown':
                # Use manual visualization engine for parsed queries
                df = pd.read_csv(uploaded_file)
                viz_engine = VisualizationEngine(df)
                
                try:
                    if parsed_result['type'] == 'histogram':
                        fig = viz_engine.plot_histogram(parsed_result['column'])
                        display_plotly_chart(fig, f"Histogram of {parsed_result['column']}")
                    
                    elif parsed_result['type'] == 'scatter':
                        fig = viz_engine.plot_scatter(parsed_result['x'], parsed_result['y'])
                        display_plotly_chart(fig, f"Scatter plot of {parsed_result['x']} vs {parsed_result['y']}")
                    
                    elif parsed_result['type'] == 'correlation':
                        fig = viz_engine.plot_correlation_heatmap()
                        display_plotly_chart(fig, "Correlation Heatmap")
                    
                    elif parsed_result['type'] == 'boxplot':
                        fig = viz_engine.plot_boxplot(parsed_result['column'])
                        display_plotly_chart(fig, f"Box plot of {parsed_result['column']}")
                    
                    elif parsed_result['type'] == 'bar':
                        fig = viz_engine.plot_bar(parsed_result['column'])
                        display_plotly_chart(fig, f"Bar chart of {parsed_result['column']}")
                    
                except Exception as e:
                    st.error(f"❌ Error creating visualization: {e}")
                    # Fallback to API call
                    st.info("🔄 Falling back to AI agent...")
                    
                    with st.spinner("AI is processing your request..."):
                        result = send_natural_language_query(user_query)
                    
                    if "error" not in result:
                        st.markdown("### 🤖 AI Response")
                        st.markdown(result.get('answer', 'No response'))
                        
                        if result.get('visualization'):
                            viz_data = result['visualization']
                            if viz_data.get('image'):
                                try:
                                    img_data = base64.b64decode(viz_data['image'])
                                    image = Image.open(io.BytesIO(img_data))
                                    st.image(image, caption="Generated Visualization")
                                except Exception as e:
                                    st.error(f"Error displaying image: {e}")
                    else:
                        st.error(f"❌ AI Error: {result['error']}")
            
            else:
                # Fallback to AI agent for unknown queries
                st.info("🤖 Sending to AI agent for processing...")
                
                with st.spinner("AI is processing your request..."):
                    result = send_natural_language_query(user_query)
                
                if "error" not in result:
                    st.markdown("### 🤖 AI Response")
                    st.markdown(result.get('answer', 'No response'))
                    
                    if result.get('visualization'):
                        viz_data = result['visualization']
                        if viz_data.get('image'):
                            try:
                                img_data = base64.b64decode(viz_data['image'])
                                image = Image.open(io.BytesIO(img_data))
                                st.image(image, caption="Generated Visualization")
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")
                else:
                    st.error(f"❌ AI Error: {result['error']}")
    
    else:  # Manual Mode
        st.markdown("## 🎛️ Manual Visualization Builder")
        
        st.markdown('<div class="manual-controls">', unsafe_allow_html=True)
        
        # Visualization type selection
        viz_type = st.selectbox(
            "📊 Choose Visualization Type:",
            ["Histogram", "Scatter Plot", "Box Plot", "Correlation Heatmap", "Bar Chart", "Pie Chart"],
            help="Select the type of visualization you want to create"
        )
        
        # Type-specific controls
        if viz_type == "Histogram":
            col1, col2 = st.columns([3, 1])
            with col1:
                column = st.selectbox("Select Column:", viz_engine.numeric_columns)
            with col2:
                bins = st.slider("Number of Bins:", 10, 100, 30)
            
            if st.button("Create Histogram"):
                fig = viz_engine.plot_histogram(column, bins)
                st.markdown('</div>', unsafe_allow_html=True)
                display_plotly_chart(fig, f"Histogram of {column}")
        
        elif viz_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-axis:", viz_engine.numeric_columns)
                color_column = st.selectbox("Color by:", [None] + viz_engine.categorical_columns)
            with col2:
                y_column = st.selectbox("Y-axis:", viz_engine.numeric_columns)
                size_column = st.selectbox("Size by:", [None] + viz_engine.numeric_columns)
            
            if st.button("Create Scatter Plot"):
                fig = viz_engine.plot_scatter(x_column, y_column, color_column, size_column)
                st.markdown('</div>', unsafe_allow_html=True)
                display_plotly_chart(fig, f"Scatter plot of {x_column} vs {y_column}")
        
        elif viz_type == "Box Plot":
            col1, col2 = st.columns(2)
            with col1:
                column = st.selectbox("Select Column:", viz_engine.numeric_columns)
            with col2:
                by_column = st.selectbox("Group by:", [None] + viz_engine.categorical_columns)
            
            if st.button("Create Box Plot"):
                fig = viz_engine.plot_boxplot(column, by_column)
                st.markdown('</div>', unsafe_allow_html=True)
                display_plotly_chart(fig, f"Box plot of {column}")
        
        elif viz_type == "Correlation Heatmap":
            st.info("This will show correlations between all numeric columns")
            
            if st.button("Create Correlation Heatmap"):
                if len(viz_engine.numeric_columns) < 2:
                    st.error("Need at least 2 numeric columns for correlation heatmap")
                else:
                    fig = viz_engine.plot_correlation_heatmap()
                    st.markdown('</div>', unsafe_allow_html=True)
                    display_plotly_chart(fig, "Correlation Heatmap")
        
        elif viz_type == "Bar Chart":
            col1, col2, col3 = st.columns(3)
            with col1:
                column = st.selectbox("Select Column:", viz_engine.categorical_columns + viz_engine.numeric_columns)
            with col2:
                group_by = st.selectbox("Group by:", [None] + viz_engine.categorical_columns)
            with col3:
                agg_func = st.selectbox("Aggregation:", ["count", "mean", "sum"])
            
            if st.button("Create Bar Chart"):
                fig = viz_engine.plot_bar(column, group_by, agg_func)
                st.markdown('</div>', unsafe_allow_html=True)
                display_plotly_chart(fig, f"Bar chart of {column}")
        
        elif viz_type == "Pie Chart":
            col1, col2 = st.columns(2)
            with col1:
                column = st.selectbox("Select Column:", viz_engine.categorical_columns + viz_engine.numeric_columns)
            with col2:
                top_n = st.slider("Show top N categories:", 5, 20, 10)
            
            if st.button("Create Pie Chart"):
                fig = viz_engine.plot_pie(column, top_n)
                st.markdown('</div>', unsafe_allow_html=True)
                display_plotly_chart(fig, f"Pie chart of {column}")
        
        # Close manual controls div if no button was pressed
        if viz_type:
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()