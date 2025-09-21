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
"""

@st.cache_data(ttl=300)  # Cache for 5 minutes
def check_api_health_cached():
    """Cached version of API health check"""
    return check_api_health()

# Custom CSS
st.markdown(load_css(), unsafe_allow_html=True)

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
    
    def plot_grouped_bar(self, value_column: str, group_column: str, agg_func: str = 'mean') -> go.Figure:
        """Create grouped bar chart with aggregation"""
        if agg_func == 'sum':
            plot_data = self.df.groupby(group_column)[value_column].sum().reset_index()
        elif agg_func == 'count':
            plot_data = self.df.groupby(group_column)[value_column].count().reset_index()
        elif agg_func == 'max':
            plot_data = self.df.groupby(group_column)[value_column].max().reset_index()
        elif agg_func == 'min':
            plot_data = self.df.groupby(group_column)[value_column].min().reset_index()
        else:  # mean (default)
            plot_data = self.df.groupby(group_column)[value_column].mean().reset_index()
        
        fig = px.bar(
            plot_data,
            x=group_column,
            y=value_column,
            title=f'{agg_func.title()} of {value_column} by {group_column}',
            template="plotly_white",
            text=value_column
        )
        
        # Format text on bars
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=500)
        return fig
    
    def plot_timeseries(self, y_column: str, time_column: str) -> go.Figure:
        """Create time series line plot"""
        # Try to convert time column to datetime if it's not already
        df_copy = self.df.copy()
        
        try:
            if not pd.api.types.is_datetime64_any_dtype(df_copy[time_column]):
                df_copy[time_column] = pd.to_datetime(df_copy[time_column])
        except Exception:
            # If conversion fails, use as is
            pass
        
        # Sort by time column
        df_copy = df_copy.sort_values(time_column)
        
        fig = px.line(
            df_copy,
            x=time_column,
            y=y_column,
            title=f'{y_column} over time',
            template="plotly_white",
            markers=True
        )
        
        fig.update_layout(
            height=500,
            xaxis_title=time_column,
            yaxis_title=y_column
        )
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
            timeout=600  # 10 minutes timeout
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

def enhanced_text_parser(query: str, columns: List[str], numeric_columns: List[str], categorical_columns: List[str]) -> Dict[str, Any]:
    """Enhanced rule-based parser for visualization queries without LLM"""
    
    if not query or len(query.strip()) < 3:
        return {
            "type": "invalid",
            "error": "Please enter a meaningful query (at least 3 characters)."
        }
    
    query_lower = query.lower().strip()
    
    # Filter out meaningless queries
    meaningless_patterns = [
        r'^hi+$', r'^hello+$', r'^hey+$', r'^ok+$', r'^yes+$', r'^no+$',
        r'^thanks?$', r'^good+$', r'^bad+$', r'^nice+$', r'^cool+$',
        r'^wow+$', r'^awesome+$', r'^great+$', r'^amazing+$', r'^test+$',
        r'^[a-z]{1,2}$',  # Single or double letters
        r'^[0-9]+$',      # Only numbers
        r'^[!@#$%^&*()_+=\-\[\]{}|;:\'",.<>?/`~]+$'  # Only special characters
    ]
    
    for pattern in meaningless_patterns:
        if re.match(pattern, query_lower):
            return {
                "type": "invalid",
                "error": "Please enter a visualization request related to your data. For example: 'histogram of age', 'scatter plot of price vs rating', etc."
            }
    
    # Check if query contains any column names (fuzzy matching)
    mentioned_columns = []
    for col in columns:
        if col.lower() in query_lower or any(word in col.lower() for word in query_lower.split() if len(word) > 2):
            mentioned_columns.append(col)
    
    # If no columns mentioned and no visualization keywords, it's probably irrelevant
    viz_keywords = [
        'plot', 'chart', 'graph', 'histogram', 'scatter', 'bar', 'pie', 'box',
        'correlation', 'heatmap', 'distribution', 'visualize', 'show', 'display',
        'create', 'make', 'generate', 'draw'
    ]
    
    has_viz_keywords = any(keyword in query_lower for keyword in viz_keywords)
    
    if not mentioned_columns and not has_viz_keywords:
        return {
            "type": "invalid",
            "error": f"I don't see any column names or visualization keywords in your query. Available columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}"
        }
    
    # Enhanced pattern matching with multiple alternatives and better column detection
    patterns = {
        'histogram': [
            r'(?:histogram|distribution|freq(?:uency)?)\s+(?:of\s+)?([a-zA-Z_]\w*)',
            r'(?:show|display|create|make|generate)\s+.*?(?:histogram|distribution)\s+(?:of\s+)?([a-zA-Z_]\w*)',
            r'([a-zA-Z_]\w*)\s+(?:histogram|distribution)',
            r'(?:plot|chart|graph)\s+(?:the\s+)?(?:distribution|histogram)\s+(?:of\s+)?([a-zA-Z_]\w*)'
        ],
        
        'scatter': [
            r'(?:scatter|scatterplot)\s+(?:plot\s+)?(?:of\s+)?([a-zA-Z_]\w*)\s+(?:vs?|against|and)\s+([a-zA-Z_]\w*)',
            r'(?:plot|chart|graph)\s+([a-zA-Z_]\w*)\s+(?:vs?|against|and)\s+([a-zA-Z_]\w*)',
            r'([a-zA-Z_]\w*)\s+(?:vs?|against|versus)\s+([a-zA-Z_]\w*)\s*(?:scatter)?',
            r'(?:relationship|correlation)\s+(?:between\s+)?([a-zA-Z_]\w*)\s+(?:and|vs?)\s+([a-zA-Z_]\w*)',
            r'(?:show|display|create|make)\s+.*?([a-zA-Z_]\w*)\s+(?:vs?|against)\s+([a-zA-Z_]\w*)'
        ],
        
        'correlation': [
            r'correlation\s+(?:matrix|heatmap|table)',
            r'heatmap\s+(?:of\s+)?correlation',
            r'(?:show|display|create|make)\s+.*?correlation.*?(?:heatmap|matrix)',
            r'correlations?\s+(?:between|of|among)\s+(?:all\s+)?(?:variables|columns|features)',
            r'(?:correlation|corr)\s*(?:heatmap|matrix)?'
        ],
        
        'boxplot': [
            r'(?:box\s*plot|boxplot)\s+(?:of\s+)?([a-zA-Z_]\w*)',
            r'(?:show|display|create|make)\s+.*?(?:box\s*plot|boxplot)\s+(?:of\s+)?([a-zA-Z_]\w*)',
            r'([a-zA-Z_]\w*)\s+(?:box\s*plot|boxplot)',
            r'(?:box\s*plot|boxplot)\s+(?:for|of)\s+([a-zA-Z_]\w*)\s+(?:by|grouped?\s+by)\s+([a-zA-Z_]\w*)'
        ],
        
        'bar': [
            r'(?:bar\s*chart?|bar\s*graph)\s+(?:of\s+)?([a-zA-Z_]\w*)',
            r'(?:count|frequency)\s+(?:of\s+)?([a-zA-Z_]\w*)',
            r'(?:show|display|create|make)\s+.*?(?:bar\s*chart?|bar\s*graph)\s+(?:of\s+)?([a-zA-Z_]\w*)',
            r'([a-zA-Z_]\w*)\s+(?:bar\s*chart?|counts?)',
            r'(?:distribution|breakdown)\s+(?:of\s+)?([a-zA-Z_]\w*)'
        ],
        
        'pie': [
            r'(?:pie\s*chart?|pie\s*graph)\s+(?:of\s+)?([a-zA-Z_]\w*)',
            r'(?:show|display|create|make)\s+.*?(?:pie\s*chart?|pie\s*graph)\s+(?:of\s+)?([a-zA-Z_]\w*)',
            r'([a-zA-Z_]\w*)\s+(?:pie\s*chart?)',
            r'(?:proportion|percentage)\s+(?:of\s+)?([a-zA-Z_]\w*)'
        ],
        
        # NEW: Grouped/Aggregated visualizations
        'grouped_bar': [
            r'(?:average|mean|avg)\s+([a-zA-Z_]\w*)\s+(?:by|per|grouped?\s+by)\s+([a-zA-Z_]\w*)',
            r'([a-zA-Z_]\w*)\s+(?:by|per|grouped?\s+by)\s+([a-zA-Z_]\w*)\s+(?:bar|chart)',
            r'(?:sum|total)\s+(?:of\s+)?([a-zA-Z_]\w*)\s+(?:by|per|grouped?\s+by)\s+([a-zA-Z_]\w*)',
            r'(?:show|display|create)\s+([a-zA-Z_]\w*)\s+(?:by|per|grouped?\s+by)\s+([a-zA-Z_]\w*)'
        ],
        
        # NEW: Time series patterns
        'timeseries': [
            r'(?:trend|time\s+series|over\s+time)\s+(?:of\s+)?([a-zA-Z_]\w*)',
            r'([a-zA-Z_]\w*)\s+(?:trend|over\s+time|time\s+series)',
            r'(?:line\s+chart|line\s+plot)\s+(?:of\s+)?([a-zA-Z_]\w*)\s+(?:by|vs)\s+([a-zA-Z_]\w*)',
            r'(?:monthly|daily|yearly|weekly)\s+([a-zA-Z_]\w*)'
        ],
        
        # NEW: Statistical summaries
        'summary_stats': [
            r'(?:summary|statistics|stats)\s+(?:of\s+)?([a-zA-Z_]\w*)',
            r'(?:describe|summarize)\s+([a-zA-Z_]\w*)',
            r'([a-zA-Z_]\w*)\s+(?:summary|statistics|stats)'
        ]
    }
    
    # Try to match patterns
    for viz_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                groups = match.groups()
                
                if viz_type == 'correlation':
                    if len(numeric_columns) < 2:
                        return {
                            "type": "error",
                            "error": f"Correlation requires at least 2 numeric columns. Found: {len(numeric_columns)}"
                        }
                    return {
                        "type": "correlation",
                        "interpretation": "→ Correlation heatmap of all numeric columns",
                        "confidence": "high"
                    }
                
                elif viz_type == 'scatter' and len(groups) >= 2:
                    col1, col2 = groups[0], groups[1]
                    # Find best column matches
                    actual_col1 = find_best_column_match(col1, columns)
                    actual_col2 = find_best_column_match(col2, columns)
                    
                    if not actual_col1 or not actual_col2:
                        return {
                            "type": "error",
                            "error": f"Could not find columns '{col1}' and/or '{col2}'. Available: {', '.join(columns)}"
                        }
                    
                    if actual_col1 not in numeric_columns or actual_col2 not in numeric_columns:
                        return {
                            "type": "error",
                            "error": f"Scatter plots require numeric columns. '{actual_col1}' and '{actual_col2}' must both be numeric."
                        }
                    
                    return {
                        "type": "scatter",
                        "x": actual_col1,
                        "y": actual_col2,
                        "interpretation": f"→ Scatter plot of {actual_col1} vs {actual_col2}",
                        "confidence": "high"
                    }
                
                elif viz_type in ['histogram', 'boxplot', 'bar', 'pie'] and len(groups) >= 1:
                    col = groups[0]
                    actual_col = find_best_column_match(col, columns)
                    
                    if not actual_col:
                        return {
                            "type": "error",
                            "error": f"Could not find column '{col}'. Available: {', '.join(columns)}"
                        }
                    
                    # Validate column type for visualization
                    if viz_type == 'histogram' and actual_col not in numeric_columns:
                        return {
                            "type": "error",
                            "error": f"Histograms require numeric columns. '{actual_col}' is not numeric."
                        }
                    
                    if viz_type == 'boxplot' and actual_col not in numeric_columns:
                        return {
                            "type": "error",
                            "error": f"Box plots require numeric columns. '{actual_col}' is not numeric."
                        }
                    
                    return {
                        "type": viz_type,
                        "column": actual_col,
                        "interpretation": f"→ {viz_type.title()} of {actual_col}",
                        "confidence": "high"
                    }
                
                # NEW: Handle grouped bar charts and aggregations
                elif viz_type == 'grouped_bar' and len(groups) >= 2:
                    value_col, group_col = groups[0], groups[1]
                    actual_value_col = find_best_column_match(value_col, columns)
                    actual_group_col = find_best_column_match(group_col, columns)
                    
                    if not actual_value_col or not actual_group_col:
                        return {
                            "type": "error",
                            "error": f"Could not find columns '{value_col}' and/or '{group_col}'. Available: {', '.join(columns)}"
                        }
                    
                    # Determine aggregation type from query
                    agg_type = 'mean'  # default
                    if 'sum' in query_lower or 'total' in query_lower:
                        agg_type = 'sum'
                    elif 'count' in query_lower:
                        agg_type = 'count'
                    elif 'max' in query_lower or 'maximum' in query_lower:
                        agg_type = 'max'
                    elif 'min' in query_lower or 'minimum' in query_lower:
                        agg_type = 'min'
                    
                    return {
                        "type": "grouped_bar",
                        "value_column": actual_value_col,
                        "group_column": actual_group_col,
                        "aggregation": agg_type,
                        "interpretation": f"→ {agg_type.title()} of {actual_value_col} by {actual_group_col}",
                        "confidence": "high"
                    }
                
                # NEW: Handle time series
                elif viz_type == 'timeseries' and len(groups) >= 1:
                    if len(groups) == 2:
                        y_col, time_col = groups[0], groups[1]
                    else:
                        y_col, time_col = groups[0], None
                    
                    actual_y_col = find_best_column_match(y_col, columns)
                    if not actual_y_col:
                        return {
                            "type": "error",
                            "error": f"Could not find column '{y_col}'. Available: {', '.join(columns)}"
                        }
                    
                    # Try to find a date/time column if not specified
                    if not time_col:
                        date_candidates = [col for col in columns if any(keyword in col.lower() 
                                         for keyword in ['date', 'time', 'created', 'year', 'month', 'day'])]
                        if date_candidates:
                            actual_time_col = date_candidates[0]
                        else:
                            return {
                                "type": "error",
                                "error": f"No time/date column found for time series. Available: {', '.join(columns)}"
                            }
                    else:
                        actual_time_col = find_best_column_match(time_col, columns)
                        if not actual_time_col:
                            return {
                                "type": "error",
                                "error": f"Could not find time column '{time_col}'. Available: {', '.join(columns)}"
                            }
                    
                    return {
                        "type": "timeseries",
                        "y_column": actual_y_col,
                        "time_column": actual_time_col,
                        "interpretation": f"→ Time series of {actual_y_col} over {actual_time_col}",
                        "confidence": "high"
                    }
                
                # NEW: Handle summary statistics
                elif viz_type == 'summary_stats' and len(groups) >= 1:
                    col = groups[0]
                    actual_col = find_best_column_match(col, columns)
                    
                    if not actual_col:
                        return {
                            "type": "error",
                            "error": f"Could not find column '{col}'. Available: {', '.join(columns)}"
                        }
                    
                    return {
                        "type": "summary_stats",
                        "column": actual_col,
                        "interpretation": f"→ Summary statistics for {actual_col}",
                        "confidence": "high"
                    }
    
    # If we reach here, we have column names or viz keywords but no clear pattern
    if mentioned_columns:
        suggestions = []
        for col in mentioned_columns:
            if col in numeric_columns:
                suggestions.extend([f"histogram of {col}", f"box plot of {col}"])
            suggestions.extend([f"bar chart of {col}", f"pie chart of {col}"])
        
        if len(mentioned_columns) >= 2:
            numeric_mentioned = [col for col in mentioned_columns if col in numeric_columns]
            if len(numeric_mentioned) >= 2:
                suggestions.append(f"scatter plot of {numeric_mentioned[0]} vs {numeric_mentioned[1]}")
        
        return {
            "type": "suggestion",
            "error": f"I found column(s): {', '.join(mentioned_columns)} but couldn't determine the visualization type.",
            "suggestions": suggestions[:3]  # Limit to 3 suggestions
        }
    
    return {
        "type": "invalid",
        "error": "I couldn't understand your request. Try using patterns like 'histogram of column_name', 'scatter plot of x vs y', etc."
    }

def find_best_column_match(query_col: str, available_columns: List[str]) -> Optional[str]:
    """Find the best matching column name using fuzzy matching with advanced techniques"""
    query_col = query_col.lower().strip()
    
    # Direct exact match first
    for col in available_columns:
        if col.lower() == query_col:
            return col
    
    # Advanced matching techniques
    normalized_query = normalize_column_name(query_col)
    best_match = None
    best_score = 0
    
    for col in available_columns:
        normalized_col = normalize_column_name(col.lower())
        
        # Calculate multiple similarity scores
        scores = []
        
        # 1. Exact substring match (highest priority)
        if normalized_query in normalized_col or normalized_col in normalized_query:
            scores.append(0.9)
        
        # 2. Word-level matching
        query_words = set(normalized_query.split('_'))
        col_words = set(normalized_col.split('_'))
        if query_words & col_words:  # Intersection
            word_match_ratio = len(query_words & col_words) / len(query_words | col_words)
            scores.append(0.8 * word_match_ratio)
        
        # 3. Levenshtein distance similarity
        max_len = max(len(normalized_query), len(normalized_col))
        if max_len > 0:
            distance = levenshtein_distance(normalized_query, normalized_col)
            similarity = 1 - (distance / max_len)
            if similarity > 0.6:  # Only consider if reasonably similar
                scores.append(0.7 * similarity)
        
        # 4. Synonym matching
        if has_synonym_match(normalized_query, normalized_col):
            scores.append(0.85)
        
        # 5. Partial acronym matching (e.g., 'cust' for 'customer')
        if is_acronym_match(normalized_query, normalized_col):
            scores.append(0.75)
        
        # Take the best score for this column
        if scores:
            column_score = max(scores)
            if column_score > best_score:
                best_score = column_score
                best_match = col
    
    # Only return if confidence is high enough
    return best_match if best_score > 0.6 else None

def normalize_column_name(col_name: str) -> str:
    """Normalize column names for better matching"""
    # Convert camelCase and PascalCase to snake_case
    import re
    col_name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', col_name)
    
    # Replace common separators with underscores
    col_name = re.sub(r'[-\s.]+', '_', col_name)
    
    # Remove special characters
    col_name = re.sub(r'[^\w]', '_', col_name)
    
    # Clean up multiple underscores
    col_name = re.sub(r'_+', '_', col_name)
    
    return col_name.lower().strip('_')

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

def has_synonym_match(query: str, column: str) -> bool:
    """Check if query and column have synonym relationships"""
    synonyms = {
        'age': ['years', 'old', 'birth', 'born'],
        'price': ['cost', 'amount', 'value', 'fee', 'charge', 'rate'],
        'name': ['title', 'label', 'id', 'identifier'],
        'date': ['time', 'when', 'timestamp', 'created', 'modified'],
        'count': ['number', 'num', 'quantity', 'total'],
        'category': ['type', 'class', 'group', 'kind'],
        'revenue': ['income', 'earnings', 'sales', 'money'],
        'customer': ['client', 'user', 'buyer', 'consumer'],
        'product': ['item', 'good', 'service'],
        'rating': ['score', 'review', 'feedback', 'stars'],
        'location': ['place', 'region', 'area', 'city', 'state'],
    }
    
    for main_word, synonym_list in synonyms.items():
        if main_word in query and any(syn in column for syn in synonym_list):
            return True
        if main_word in column and any(syn in query for syn in synonym_list):
            return True
    
    return False

def is_acronym_match(query: str, column: str) -> bool:
    """Check if query could be an acronym of the column"""
    if len(query) < 2 or len(query) >= len(column):
        return False
    
    # Check if query characters appear in order in column
    col_index = 0
    for char in query:
        found = False
        while col_index < len(column):
            if column[col_index] == char:
                found = True
                col_index += 1
                break
            col_index += 1
        if not found:
            return False
    
    return True
    
    # Exact match first
    for col in available_columns:
        if col.lower() == query_col:
            return col
    
    # Partial match
    for col in available_columns:
        if query_col in col.lower() or col.lower() in query_col:
            return col
    
    # Check if query_col is a word within any column name
    for col in available_columns:
        col_words = re.split(r'[_\s]+', col.lower())
        if query_col in col_words:
            return col
    
    # Fuzzy matching - check if most characters match
    for col in available_columns:
        if len(query_col) >= 3 and len(col) >= 3:
            common_chars = sum(1 for a, b in zip(query_col, col.lower()) if a == b)
            if common_chars / min(len(query_col), len(col)) > 0.6:  # 60% similarity
                return col
    
    return None
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
    """Display plotly chart with optional download"""
    st.plotly_chart(fig, use_container_width=True)
    
    # Try to create download button (optional - requires kaleido)
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=800)
        st.download_button(
            label="📥 Download Chart as PNG",
            data=img_bytes,
            file_name=f"{chart_title.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
    except Exception as e:
        # If kaleido is not installed, show a note instead of failing
        st.info("💡 Install `kaleido` package to enable PNG download: `pip install kaleido`")

def main():
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'dataset_summary' not in st.session_state:
        st.session_state.dataset_summary = None
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'uploaded_file_id' not in st.session_state:
        st.session_state.uploaded_file_id = None
    if 'last_interaction_mode' not in st.session_state:
        st.session_state.last_interaction_mode = None
    if 'api_health_checked' not in st.session_state:
        st.session_state.api_health_checked = False

    # Header
    st.markdown('<h1 class="main-header">🤖 GenAI Data Visualization Dashboard</h1>', unsafe_allow_html=True)
    
    # Check API health only once or when needed
    if not st.session_state.api_health_checked:
        is_healthy, health_data = check_api_health_cached()
        st.session_state.api_health_checked = True
        st.session_state.api_healthy = is_healthy
    
    if not st.session_state.get('api_healthy', False):
        st.error("🚨 API is not running! Please start the FastAPI server first.")
        st.code("python start_server.py")
        if st.button("🔄 Retry Connection"):
            st.session_state.api_health_checked = False
            st.rerun()
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Dashboard Controls")
        
        # Mode Selection
        st.markdown('<div class="mode-toggle">', unsafe_allow_html=True)
        st.markdown("### 🔄 Interaction Mode")
        interaction_mode = st.radio(
            "Choose your preferred interaction mode:",
            ["🗣️ Natural Language", "🎛️ Manual Mode", "🧠 Smart Parser"],
            index=0,
            key="interaction_mode_radio",
            help="Natural Language: AI-powered query processing\nManual Mode: Use UI controls to build visualizations\nSmart Parser: Fast rule-based text parsing (no AI)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Store mode change to avoid unnecessary processing
        if st.session_state.last_interaction_mode != interaction_mode:
            st.session_state.last_interaction_mode = interaction_mode
        
        # File Upload Section
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            key="csv_uploader",
            help="Upload your dataset to start creating visualizations"
        )
        
        # Check if this is a new file
        current_file_id = None
        if uploaded_file is not None:
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Process file only if it's new or changed
        if uploaded_file is not None and current_file_id != st.session_state.uploaded_file_id:
            # Upload file and cache the dataframe
            with st.spinner("Uploading and processing file..."):
                success, upload_result = upload_file(uploaded_file)
            
            if success:
                # Cache the dataframe in session state
                uploaded_file.seek(0)  # Reset file pointer
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_file_id = current_file_id
                st.session_state.file_uploaded = True
                
                # Get and cache dataset summary
                st.session_state.dataset_summary = get_dataset_summary()
                
                st.success(f"✅ File uploaded successfully!")
                st.info(f"📊 **{upload_result['rows']}** rows × **{upload_result['columns']}** columns")
            else:
                st.error(f"❌ Upload failed: {upload_result.get('error', 'Unknown error')}")
                st.session_state.file_uploaded = False
        
        # Display dataset info if file is loaded
        if st.session_state.file_uploaded and st.session_state.dataset_summary:
            st.success("✅ File ready!")
            dataset_summary = st.session_state.dataset_summary
            
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
    
    # Main content area
    if not st.session_state.file_uploaded:
        st.info("👆 Please upload a CSV file to get started!")
        return
    
    # Use cached dataset summary
    dataset_summary = st.session_state.dataset_summary
    if not dataset_summary:
        st.error("❌ Could not retrieve dataset information")
        return
    
    # Create visualization engine from cached dataframe (with caching)
    @st.cache_data
    def create_viz_engine(df_id):
        """Create visualization engine with caching based on dataframe ID"""
        if st.session_state.df is not None:
            return VisualizationEngine(st.session_state.df)
        return None
    
    try:
        if interaction_mode in ["🎛️ Manual Mode", "🧠 Smart Parser"] and st.session_state.file_uploaded:
            # Use file ID to cache the viz engine
            viz_engine = create_viz_engine(st.session_state.uploaded_file_id)
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
        
        # Use form to prevent reruns on every keystroke
        with st.form(key="natural_language_form", clear_on_submit=False):
            user_query = st.text_area(
                "What would you like to visualize?",
                height=100,
                placeholder="e.g., Create a scatter plot of age vs salary",
                help="Describe the visualization you want in natural language",
                key="nl_query_input"
            )
            
            submitted = st.form_submit_button("🚀 Generate Visualization", type="primary")
        
        if submitted and user_query.strip():
            
            # Try simple parsing first
            all_columns = dataset_summary['numeric_columns'] + dataset_summary['categorical_columns']
            parsed_result = parse_natural_language(user_query, all_columns)
            
            # Show interpretation
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            st.markdown(f"**Query Interpretation:** {parsed_result['interpretation']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if parsed_result['type'] != 'unknown':
                # Use manual visualization engine for parsed queries with cached dataframe
                viz_engine = VisualizationEngine(st.session_state.df)
                
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
        
        elif submitted and not user_query.strip():
            st.warning("Please enter a query!")
    
    elif interaction_mode == "🧠 Smart Parser":
        st.markdown("## 🧠 Smart Parser Mode")
        st.markdown("Fast rule-based parsing without AI - just type what you want!")
        
        # Info about Smart Parser
        with st.expander("ℹ️ How Smart Parser Works"):
            st.markdown("""
            **Smart Parser** uses advanced rule-based pattern matching with fuzzy column matching:
            
            **✅ Supported Patterns:**
            
            **Basic Visualizations:**
            - `histogram of [column]` or `distribution of [column]`
            - `scatter plot of [column1] vs [column2]`  
            - `correlation heatmap` or `correlations`
            - `box plot of [column]`
            - `bar chart of [column]` or `count of [column]`
            - `pie chart of [column]`
            
            **📊 NEW: Advanced Patterns:**
            - `average [column] by [category]` → Grouped bar chart with mean
            - `sum of [column] by [category]` → Grouped bar chart with sum
            - `[column] trend over time` → Time series line plot
            - `summary of [column]` → Detailed statistics table
            - `[column] by [category] bar chart` → Grouped visualization
            
            **🧠 Smart Features:**
            - **Fuzzy column matching**: 'age' finds 'customer_age', 'user_age', etc.
            - **Synonym support**: 'price' matches 'cost', 'amount', 'value'
            - **Multi-word handling**: Handles spaces, underscores, camelCase
            - **Intelligent suggestions**: Shows relevant alternatives when patterns aren't clear
            
            **⚡ Advantages:**
            - Instant processing (no AI waiting time)
            - Predictable, consistent behavior
            - Advanced error messages with suggestions
            - Supports complex aggregations and grouping
            
            **🎯 Pro Tips:**
            - Use partial column names (e.g., 'cust' for 'customer_id')
            - Try synonyms (e.g., 'revenue' for 'sales_amount')
            - Include keywords like 'average', 'sum', 'trend', 'by'
            - For time series, mention 'trend', 'over time', or 'time series'
            """)
        
        # NEW: Quick Examples Section
        with st.expander("💡 Quick Examples"):
            st.markdown("""
            **Copy these patterns and modify with your column names:**
            
            ```
            histogram of price
            scatter plot of age vs income  
            average sales by region
            sum of revenue by month
            price trend over time
            summary of customer_satisfaction
            correlation heatmap
            box plot of scores
            ```
            """)
        
        # NEW: Column Helper
        if 'dataset_summary' in locals() and dataset_summary:
            with st.expander("📋 Available Columns"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📊 Numeric Columns:**")
                    for col in dataset_summary['numeric_columns'][:10]:  # Show first 10
                        st.code(col)
                    if len(dataset_summary['numeric_columns']) > 10:
                        st.markdown(f"... and {len(dataset_summary['numeric_columns']) - 10} more")
                
                with col2:
                    st.markdown("**🏷️ Categorical Columns:**")
                    for col in dataset_summary['categorical_columns'][:10]:  # Show first 10  
                        st.code(col)
                    if len(dataset_summary['categorical_columns']) > 10:
                        st.markdown(f"... and {len(dataset_summary['categorical_columns']) - 10} more")
        
        # Smart Parser input with form to prevent reruns
        with st.form(key="smart_parser_form", clear_on_submit=False):
            # Check if we have a template query to pre-populate
            default_query = ""
            if 'sp_template_query' in st.session_state:
                default_query = st.session_state['sp_template_query']
                # Clear it so it doesn't persist
                del st.session_state['sp_template_query']
            
            user_query = st.text_area(
                "What visualization do you want?",
                height=100,
                placeholder="e.g., histogram of price, average sales by region, price trend over time, summary of age",
                help="Enter your visualization request - supports fuzzy matching and advanced patterns!",
                key="sp_query_input",
                value=default_query
            )
            
            # NEW: Quick Template Buttons (inside form)
            st.markdown("**🚀 Quick Templates:** Click any button to use a template:")
            
            # Get column names for templates
            if 'dataset_summary' in locals() and dataset_summary:
                numeric_cols = dataset_summary['numeric_columns']
                categorical_cols = dataset_summary['categorical_columns']
                all_cols = numeric_cols + categorical_cols
                
                # Create template buttons in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    template1 = st.form_submit_button("📊 Data Overview", help="Generate correlation heatmap")
                    if numeric_cols:
                        template2 = st.form_submit_button(f"📈 Histogram", help=f"Histogram of {numeric_cols[0]}")
                
                with col2:
                    if len(numeric_cols) >= 2:
                        template3 = st.form_submit_button(f"🔵 Scatter Plot", help=f"Scatter plot analysis")
                    if categorical_cols and numeric_cols:
                        template4 = st.form_submit_button("📊 Grouped Chart", help="Average by category")
                
                with col3:
                    # Look for date/time columns
                    date_cols = [col for col in all_cols if any(keyword in col.lower() 
                               for keyword in ['date', 'time', 'created', 'year', 'month', 'day'])]
                    if date_cols and numeric_cols:
                        template5 = st.form_submit_button("📅 Time Trends", help="Time series analysis")
                    if numeric_cols:
                        template6 = st.form_submit_button(f"📦 Box Plot", help=f"Box plot analysis")
                
                with col4:
                    if categorical_cols:
                        template7 = st.form_submit_button(f"🥧 Pie Chart", help=f"Category breakdown")
                    if numeric_cols:
                        template8 = st.form_submit_button(f"📋 Statistics", help=f"Summary statistics")
            
            submitted = st.form_submit_button("⚡ Parse & Generate", type="primary")
            
            # Handle template button clicks
            if 'dataset_summary' in locals() and dataset_summary:
                if template1:
                    user_query = "correlation heatmap"
                    submitted = True
                elif 'template2' in locals() and template2 and numeric_cols:
                    user_query = f"histogram of {numeric_cols[0]}"
                    submitted = True
                elif 'template3' in locals() and template3 and len(numeric_cols) >= 2:
                    user_query = f"scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}"
                    submitted = True
                elif 'template4' in locals() and template4 and categorical_cols and numeric_cols:
                    user_query = f"average {numeric_cols[0]} by {categorical_cols[0]}"
                    submitted = True
                elif 'template5' in locals() and template5 and date_cols and numeric_cols:
                    user_query = f"{numeric_cols[0]} trend over time"
                    submitted = True
                elif 'template6' in locals() and template6 and numeric_cols:
                    user_query = f"box plot of {numeric_cols[0]}"
                    submitted = True
                elif 'template7' in locals() and template7 and categorical_cols:
                    user_query = f"pie chart of {categorical_cols[0]}"
                    submitted = True
                elif 'template8' in locals() and template8 and numeric_cols:
                    user_query = f"summary of {numeric_cols[0]}"
                    submitted = True
        
        if submitted and user_query and user_query.strip():
            
            # Add to query history
            if 'sp_query_history' not in st.session_state:
                st.session_state.sp_query_history = []
            
            # Add current query to history (avoid duplicates)
            if user_query.strip() not in st.session_state.sp_query_history:
                st.session_state.sp_query_history.insert(0, user_query.strip())
                # Keep only last 10 queries
                st.session_state.sp_query_history = st.session_state.sp_query_history[:10]
            
            # Use enhanced parser
            all_columns = dataset_summary['numeric_columns'] + dataset_summary['categorical_columns']
            parsed_result = enhanced_text_parser(
                user_query, 
                all_columns, 
                dataset_summary['numeric_columns'], 
                dataset_summary['categorical_columns']
            )
            
            # Show parsing result
            if parsed_result['type'] == 'invalid':
                st.error(f"❌ {parsed_result['error']}")
                
            elif parsed_result['type'] == 'error':
                st.error(f"❌ {parsed_result['error']}")
                
            elif parsed_result['type'] == 'suggestion':
                st.warning(f"⚠️ {parsed_result['error']}")
                if parsed_result.get('suggestions'):
                    st.info("💡 **Try these instead:**")
                    for suggestion in parsed_result['suggestions']:
                        st.markdown(f"- `{suggestion}`")
                        
            else:
                # Valid parsing result - show interpretation
                confidence_icon = "🎯" if parsed_result.get('confidence') == 'high' else "🤔"
                st.success(f"{confidence_icon} **Parsed Successfully:** {parsed_result['interpretation']}")
                
                # Generate visualization
                try:
                    if parsed_result['type'] == 'histogram':
                        fig = viz_engine.plot_histogram(parsed_result['column'])
                        display_plotly_chart(fig, f"Histogram of {parsed_result['column']}")
                        st.info(f"📊 Generated histogram for **{parsed_result['column']}** with {len(st.session_state.df)} data points")
                    
                    elif parsed_result['type'] == 'scatter':
                        fig = viz_engine.plot_scatter(parsed_result['x'], parsed_result['y'])
                        display_plotly_chart(fig, f"Scatter plot of {parsed_result['x']} vs {parsed_result['y']}")
                        correlation = st.session_state.df[parsed_result['x']].corr(st.session_state.df[parsed_result['y']])
                        st.info(f"📈 Scatter plot shows correlation of **{correlation:.3f}** between {parsed_result['x']} and {parsed_result['y']}")
                    
                    elif parsed_result['type'] == 'correlation':
                        fig = viz_engine.plot_correlation_heatmap()
                        display_plotly_chart(fig, "Correlation Heatmap")
                        st.info(f"🔥 Correlation heatmap for **{len(dataset_summary['numeric_columns'])}** numeric columns")
                    
                    elif parsed_result['type'] == 'boxplot':
                        fig = viz_engine.plot_boxplot(parsed_result['column'])
                        display_plotly_chart(fig, f"Box plot of {parsed_result['column']}")
                        stats = st.session_state.df[parsed_result['column']].describe()
                        st.info(f"📦 Box plot shows median: **{stats['50%']:.2f}**, IQR: **{stats['75%'] - stats['25%']:.2f}**")
                    
                    elif parsed_result['type'] == 'bar':
                        fig = viz_engine.plot_bar(parsed_result['column'])
                        display_plotly_chart(fig, f"Bar chart of {parsed_result['column']}")
                        unique_vals = st.session_state.df[parsed_result['column']].nunique()
                        st.info(f"📊 Bar chart shows **{unique_vals}** unique values in {parsed_result['column']}")
                    
                    elif parsed_result['type'] == 'pie':
                        fig = viz_engine.plot_pie(parsed_result['column'])
                        display_plotly_chart(fig, f"Pie chart of {parsed_result['column']}")
                        top_category = st.session_state.df[parsed_result['column']].mode().iloc[0]
                        st.info(f"🥧 Most common value: **{top_category}**")
                    
                    # NEW: Handle grouped bar charts
                    elif parsed_result['type'] == 'grouped_bar':
                        fig = viz_engine.plot_grouped_bar(
                            parsed_result['value_column'], 
                            parsed_result['group_column'],
                            parsed_result['aggregation']
                        )
                        display_plotly_chart(fig, f"{parsed_result['aggregation'].title()} of {parsed_result['value_column']} by {parsed_result['group_column']}")
                        groups = st.session_state.df[parsed_result['group_column']].nunique()
                        st.info(f"📊 Grouped by **{groups}** categories in {parsed_result['group_column']}")
                    
                    # NEW: Handle time series
                    elif parsed_result['type'] == 'timeseries':
                        fig = viz_engine.plot_timeseries(parsed_result['y_column'], parsed_result['time_column'])
                        display_plotly_chart(fig, f"Time series of {parsed_result['y_column']}")
                        time_range = f"{st.session_state.df[parsed_result['time_column']].min()} to {st.session_state.df[parsed_result['time_column']].max()}"
                        st.info(f"📅 Time range: **{time_range}**")
                    
                    # NEW: Handle summary statistics
                    elif parsed_result['type'] == 'summary_stats':
                        col_data = st.session_state.df[parsed_result['column']]
                        
                        # Display statistics table
                        st.markdown(f"### 📊 Summary Statistics: {parsed_result['column']}")
                        
                        if parsed_result['column'] in dataset_summary['numeric_columns']:
                            stats = col_data.describe()
                            stats_df = pd.DataFrame({
                                'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50% (Median)', '75%', 'Max'],
                                'Value': [
                                    f"{stats['count']:.0f}",
                                    f"{stats['mean']:.3f}",
                                    f"{stats['std']:.3f}",
                                    f"{stats['min']:.3f}",
                                    f"{stats['25%']:.3f}",
                                    f"{stats['50%']:.3f}",
                                    f"{stats['75%']:.3f}",
                                    f"{stats['max']:.3f}"
                                ]
                            })
                            st.dataframe(stats_df, use_container_width=True)
                            
                            # Create a distribution plot alongside stats
                            fig = viz_engine.plot_histogram(parsed_result['column'])
                            display_plotly_chart(fig, f"Distribution of {parsed_result['column']}")
                            
                        else:  # Categorical column
                            value_counts = col_data.value_counts()
                            stats_df = pd.DataFrame({
                                'Category': value_counts.index,
                                'Count': value_counts.values,
                                'Percentage': (value_counts.values / len(col_data) * 100).round(2)
                            })
                            st.dataframe(stats_df, use_container_width=True)
                            
                            # Create a bar chart alongside stats
                            fig = viz_engine.plot_bar(parsed_result['column'])
                            display_plotly_chart(fig, f"Distribution of {parsed_result['column']}")
                        
                        st.info(f"📈 Statistics generated for **{len(col_data)}** data points")
                
                except Exception as e:
                    st.error(f"❌ Error creating visualization: {e}")
                    st.info("💡 Try checking your column names or data types")
        
        elif submitted and (not user_query or not user_query.strip()):
            st.warning("Please enter a query!")
        
        # NEW: Query History Section (outside the form)
        if 'sp_query_history' in st.session_state and st.session_state.sp_query_history:
            with st.expander("🕒 Recent Queries", expanded=False):
                st.markdown("**Click any query to reuse it:**")
                for i, old_query in enumerate(st.session_state.sp_query_history):
                    if st.button(f"🔄 {old_query}", key=f"history_{i}", help=f"Click to reuse: {old_query}"):
                        st.session_state['sp_reuse_query'] = old_query
                        st.rerun()
                
                # Clear history button
                if st.button("🗑️ Clear History", help="Remove all saved queries"):
                    st.session_state.sp_query_history = []
                    st.rerun()
    
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