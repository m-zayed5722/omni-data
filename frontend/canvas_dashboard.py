#!/usr/bin/env python3

"""
Advanced Multi-Visualization Canvas Dashboard
Interactive grid-based layout with drag-and-drop, customization, and persistence
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
import zipfile
import json
import uuid

# Configure for better performance
st.set_page_config(
    page_title="Multi-Viz Canvas Dashboard",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ==========================================
# DASHBOARD CONFIGURATION CLASSES
# ==========================================

class ChartConfig:
    """Configuration for individual charts"""
    def __init__(self, chart_id: str = None, chart_type: str = "scatter", 
                 title: str = "", x_column: str = "", y_column: str = "", 
                 category_column: str = "", size: str = "medium", 
                 position: int = 0, custom_params: dict = None):
        self.chart_id = chart_id or str(uuid.uuid4())
        self.chart_type = chart_type
        self.title = title
        self.x_column = x_column
        self.y_column = y_column
        self.category_column = category_column
        self.size = size  # small, medium, large, full
        self.position = position
        self.custom_params = custom_params or {}
    
    def to_dict(self):
        return {
            'chart_id': self.chart_id,
            'chart_type': self.chart_type,
            'title': self.title,
            'x_column': self.x_column,
            'y_column': self.y_column,
            'category_column': self.category_column,
            'size': self.size,
            'position': self.position,
            'custom_params': self.custom_params
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

class DashboardState:
    """Manages the entire dashboard state"""
    def __init__(self):
        self.charts: List[ChartConfig] = []
        self.dataset: pd.DataFrame = None
        self.dataset_analysis: dict = {}
        self.dashboard_name: str = "My Dashboard"
        self.created_at: str = datetime.now().isoformat()
    
    def add_chart(self, chart_config: ChartConfig):
        self.charts.append(chart_config)
        self.sort_charts()
    
    def remove_chart(self, chart_id: str):
        self.charts = [c for c in self.charts if c.chart_id != chart_id]
        self.sort_charts()
    
    def sort_charts(self):
        self.charts.sort(key=lambda x: x.position)
    
    def to_dict(self):
        return {
            'dashboard_name': self.dashboard_name,
            'created_at': self.created_at,
            'charts': [chart.to_dict() for chart in self.charts]
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        dashboard = cls()
        dashboard.dashboard_name = data.get('dashboard_name', 'My Dashboard')
        dashboard.created_at = data.get('created_at', datetime.now().isoformat())
        dashboard.charts = [ChartConfig.from_dict(c) for c in data.get('charts', [])]
        return dashboard

# ==========================================
# CHART CREATION FUNCTIONS
# ==========================================

def create_plotly_chart(chart_config: ChartConfig, data: pd.DataFrame) -> go.Figure:
    """Create interactive Plotly charts"""
    
    if chart_config.chart_type == "scatter":
        fig = px.scatter(
            data, 
            x=chart_config.x_column, 
            y=chart_config.y_column,
            color=chart_config.category_column if chart_config.category_column else None,
            title=chart_config.title,
            template="plotly_white"
        )
        
    elif chart_config.chart_type == "histogram":
        fig = px.histogram(
            data, 
            x=chart_config.x_column,
            color=chart_config.category_column if chart_config.category_column else None,
            title=chart_config.title,
            template="plotly_white"
        )
        
    elif chart_config.chart_type == "box":
        fig = px.box(
            data, 
            x=chart_config.category_column,
            y=chart_config.x_column,
            title=chart_config.title,
            template="plotly_white"
        )
        
    elif chart_config.chart_type == "bar":
        value_counts = data[chart_config.x_column].value_counts().head(20)
        fig = px.bar(
            x=value_counts.index, 
            y=value_counts.values,
            title=chart_config.title,
            template="plotly_white"
        )
        fig.update_xaxes(title=chart_config.x_column)
        fig.update_yaxes(title="Count")
        
    elif chart_config.chart_type == "pie":
        value_counts = data[chart_config.x_column].value_counts().head(10)
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=chart_config.title
        )
        
    elif chart_config.chart_type == "line":
        fig = px.line(
            data, 
            x=chart_config.x_column, 
            y=chart_config.y_column,
            color=chart_config.category_column if chart_config.category_column else None,
            title=chart_config.title,
            template="plotly_white"
        )
        
    elif chart_config.chart_type == "heatmap":
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title=chart_config.title, template="plotly_white")
        
    elif chart_config.chart_type == "violin":
        fig = px.violin(
            data, 
            x=chart_config.category_column,
            y=chart_config.x_column,
            title=chart_config.title,
            template="plotly_white"
        )
        
    else:
        # Default to scatter
        fig = px.scatter(data, x=data.columns[0], y=data.columns[1], title=chart_config.title)
    
    # Update layout based on size
    height_map = {"small": 300, "medium": 400, "large": 500, "full": 600}
    fig.update_layout(height=height_map.get(chart_config.size, 400))
    
    return fig

def auto_generate_dashboard(data: pd.DataFrame, analysis: dict) -> List[ChartConfig]:
    """Automatically generate initial dashboard configuration"""
    charts = []
    numeric_cols = analysis.get('numeric_columns', [])
    categorical_cols = analysis.get('categorical_columns', [])
    
    position = 0
    
    # 1. Scatter plot for first two numeric columns
    if len(numeric_cols) >= 2:
        charts.append(ChartConfig(
            chart_type="scatter",
            title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
            x_column=numeric_cols[0],
            y_column=numeric_cols[1],
            category_column=categorical_cols[0] if categorical_cols else "",
            size="large",
            position=position
        ))
        position += 1
    
    # 2. Histogram for first numeric column
    if len(numeric_cols) >= 1:
        charts.append(ChartConfig(
            chart_type="histogram",
            title=f"Distribution of {numeric_cols[0]}",
            x_column=numeric_cols[0],
            category_column=categorical_cols[0] if categorical_cols else "",
            size="medium",
            position=position
        ))
        position += 1
    
    # 3. Bar chart for first categorical column
    if len(categorical_cols) >= 1:
        charts.append(ChartConfig(
            chart_type="bar",
            title=f"Count by {categorical_cols[0]}",
            x_column=categorical_cols[0],
            size="medium",
            position=position
        ))
        position += 1
    
    # 4. Correlation heatmap if we have multiple numeric columns
    if len(numeric_cols) >= 3:
        charts.append(ChartConfig(
            chart_type="heatmap",
            title="Correlation Matrix",
            size="large",
            position=position
        ))
        position += 1
    
    # 5. Box plot for numeric vs categorical
    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        charts.append(ChartConfig(
            chart_type="box",
            title=f"{numeric_cols[0]} by {categorical_cols[0]}",
            x_column=numeric_cols[0],
            category_column=categorical_cols[0],
            size="medium",
            position=position
        ))
        position += 1
    
    # 6. Pie chart for categorical data
    if len(categorical_cols) >= 1:
        unique_count = data[categorical_cols[0]].nunique()
        if 2 <= unique_count <= 10:
            charts.append(ChartConfig(
                chart_type="pie",
                title=f"{categorical_cols[0]} Distribution",
                x_column=categorical_cols[0],
                size="medium",
                position=position
            ))
    
    return charts

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Enhanced dataset analysis"""
    analysis = {
        'shape': df.shape,
        'columns': list(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Enhanced categorical analysis
    analysis['categorical_summary'] = {}
    for col in analysis['categorical_columns']:
        analysis['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    
    # Enhanced numeric analysis
    if analysis['numeric_columns']:
        analysis['numeric_summary'] = df[analysis['numeric_columns']].describe().to_dict()
        analysis['correlation_matrix'] = df[analysis['numeric_columns']].corr()
    
    return analysis

# ==========================================
# UI COMPONENTS
# ==========================================

def render_chart_card(chart_config: ChartConfig, data: pd.DataFrame, dashboard: DashboardState):
    """Render individual chart card with controls"""
    
    # Chart type icons
    chart_icons = {
        'scatter': '🔵', 'histogram': '📊', 'bar': '📈', 'box': '📦',
        'pie': '🥧', 'line': '📉', 'heatmap': '🔥', 'violin': '🎻'
    }
    
    icon = chart_icons.get(chart_config.chart_type, '📊')
    
    with st.container():
        # Card header with controls
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.markdown(f"### {icon} {chart_config.title}")
        
        with col2:
            # Size control
            new_size = st.selectbox(
                "Size", 
                ["small", "medium", "large", "full"], 
                index=["small", "medium", "large", "full"].index(chart_config.size),
                key=f"size_{chart_config.chart_id}"
            )
            if new_size != chart_config.size:
                chart_config.size = new_size
                st.rerun()
        
        with col3:
            # Edit button
            if st.button("✏️ Edit", key=f"edit_{chart_config.chart_id}"):
                st.session_state[f"editing_{chart_config.chart_id}"] = True
        
        with col4:
            # Remove button
            if st.button("🗑️ Remove", key=f"remove_{chart_config.chart_id}"):
                dashboard.remove_chart(chart_config.chart_id)
                st.rerun()
        
        # Edit form (if editing)
        if st.session_state.get(f"editing_{chart_config.chart_id}", False):
            with st.expander("📝 Edit Chart", expanded=True):
                edit_chart_form(chart_config, data)
        
        # Render the actual chart
        try:
            fig = create_plotly_chart(chart_config, data)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_config.chart_id}")
            
            # Download button
            chart_html = fig.to_html()
            st.download_button(
                "📥 Download HTML",
                data=chart_html,
                file_name=f"{chart_config.title.replace(' ', '_')}.html",
                mime="text/html",
                key=f"download_{chart_config.chart_id}"
            )
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")

def edit_chart_form(chart_config: ChartConfig, data: pd.DataFrame):
    """Form for editing chart configuration"""
    
    # Chart type selection
    chart_types = ["scatter", "histogram", "bar", "box", "pie", "line", "heatmap", "violin"]
    new_type = st.selectbox(
        "Chart Type", 
        chart_types, 
        index=chart_types.index(chart_config.chart_type),
        key=f"type_edit_{chart_config.chart_id}"
    )
    
    # Title
    new_title = st.text_input(
        "Title", 
        value=chart_config.title,
        key=f"title_edit_{chart_config.chart_id}"
    )
    
    # Column selections based on chart type
    numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
    all_cols = list(data.columns)
    
    if new_type in ["scatter", "line"]:
        new_x = st.selectbox("X Column", all_cols, 
                           index=all_cols.index(chart_config.x_column) if chart_config.x_column in all_cols else 0,
                           key=f"x_edit_{chart_config.chart_id}")
        new_y = st.selectbox("Y Column", all_cols,
                           index=all_cols.index(chart_config.y_column) if chart_config.y_column in all_cols else 0,
                           key=f"y_edit_{chart_config.chart_id}")
        new_category = st.selectbox("Color By (Optional)", [""] + categorical_cols,
                                  index=([""] + categorical_cols).index(chart_config.category_column) if chart_config.category_column in categorical_cols else 0,
                                  key=f"cat_edit_{chart_config.chart_id}")
    
    elif new_type in ["histogram", "bar", "pie"]:
        new_x = st.selectbox("Column", all_cols,
                           index=all_cols.index(chart_config.x_column) if chart_config.x_column in all_cols else 0,
                           key=f"x_edit_{chart_config.chart_id}")
        new_y = ""
        new_category = st.selectbox("Group By (Optional)", [""] + categorical_cols,
                                  index=([""] + categorical_cols).index(chart_config.category_column) if chart_config.category_column in categorical_cols else 0,
                                  key=f"cat_edit_{chart_config.chart_id}")
    
    elif new_type in ["box", "violin"]:
        new_x = st.selectbox("Numeric Column", numeric_cols,
                           index=numeric_cols.index(chart_config.x_column) if chart_config.x_column in numeric_cols else 0,
                           key=f"x_edit_{chart_config.chart_id}")
        new_y = ""
        new_category = st.selectbox("Category Column", categorical_cols,
                                  index=categorical_cols.index(chart_config.category_column) if chart_config.category_column in categorical_cols else 0,
                                  key=f"cat_edit_{chart_config.chart_id}")
    
    else:  # heatmap
        new_x = ""
        new_y = ""
        new_category = ""
    
    # Update and save buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Changes", key=f"save_{chart_config.chart_id}"):
            chart_config.chart_type = new_type
            chart_config.title = new_title
            chart_config.x_column = new_x
            chart_config.y_column = new_y
            chart_config.category_column = new_category
            st.session_state[f"editing_{chart_config.chart_id}"] = False
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel", key=f"cancel_{chart_config.chart_id}"):
            st.session_state[f"editing_{chart_config.chart_id}"] = False
            st.rerun()

def add_visualization_form(dashboard: DashboardState, data: pd.DataFrame):
    """Form for adding new visualizations"""
    
    st.subheader("➕ Add New Visualization")
    
    with st.form("add_chart_form"):
        # Chart configuration
        chart_type = st.selectbox("Chart Type", 
                                 ["scatter", "histogram", "bar", "box", "pie", "line", "heatmap", "violin"])
        
        title = st.text_input("Chart Title", value=f"New {chart_type.title()} Chart")
        
        # Column selections
        numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
        all_cols = list(data.columns)
        
        if chart_type in ["scatter", "line"]:
            x_col = st.selectbox("X Column", all_cols)
            y_col = st.selectbox("Y Column", all_cols)
            cat_col = st.selectbox("Color By (Optional)", [""] + categorical_cols)
        elif chart_type in ["histogram", "bar", "pie"]:
            x_col = st.selectbox("Column", all_cols)
            y_col = ""
            cat_col = st.selectbox("Group By (Optional)", [""] + categorical_cols)
        elif chart_type in ["box", "violin"]:
            x_col = st.selectbox("Numeric Column", numeric_cols) if numeric_cols else ""
            y_col = ""
            cat_col = st.selectbox("Category Column", categorical_cols) if categorical_cols else ""
        else:  # heatmap
            x_col = ""
            y_col = ""
            cat_col = ""
        
        size = st.selectbox("Size", ["small", "medium", "large", "full"], index=1)
        
        if st.form_submit_button("🎨 Add Visualization"):
            new_chart = ChartConfig(
                chart_type=chart_type,
                title=title,
                x_column=x_col,
                y_column=y_col,
                category_column=cat_col,
                size=size,
                position=len(dashboard.charts)
            )
            dashboard.add_chart(new_chart)
            st.success(f"Added {chart_type} chart: {title}")
            st.rerun()

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    st.title("🎨 Multi-Visualization Canvas Dashboard")
    st.markdown("---")
    
    # Initialize session state
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = DashboardState()
    
    dashboard = st.session_state.dashboard
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # File upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                dashboard.dataset = df
                dashboard.dataset_analysis = analyze_dataset(df)
                st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Auto-generate dashboard button
                if st.button("🚀 Auto-Generate Dashboard"):
                    dashboard.charts = auto_generate_dashboard(df, dashboard.dataset_analysis)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        # Dashboard management
        if dashboard.dataset is not None:
            st.subheader("💾 Dashboard Management")
            
            # Dashboard name
            dashboard.dashboard_name = st.text_input("Dashboard Name", dashboard.dashboard_name)
            
            # Export dashboard
            if st.button("📤 Export Dashboard"):
                dashboard_json = json.dumps(dashboard.to_dict(), indent=2)
                st.download_button(
                    "📥 Download Config",
                    data=dashboard_json,
                    file_name=f"{dashboard.dashboard_name.replace(' ', '_')}_config.json",
                    mime="application/json"
                )
            
            # Import dashboard
            config_file = st.file_uploader("📥 Import Dashboard", type=['json'])
            if config_file is not None:
                try:
                    config_data = json.loads(config_file.read())
                    imported_dashboard = DashboardState.from_dict(config_data)
                    dashboard.charts = imported_dashboard.charts
                    dashboard.dashboard_name = imported_dashboard.dashboard_name
                    st.success("✅ Dashboard imported!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error importing: {str(e)}")
            
            # Clear dashboard
            if st.button("🗑️ Clear Dashboard"):
                dashboard.charts = []
                st.rerun()
    
    # Main content area
    if dashboard.dataset is None:
        st.info("👈 Please upload a CSV file to get started")
        return
    
    # Dataset overview
    st.header(f"📊 {dashboard.dashboard_name}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", dashboard.dataset.shape[0])
    with col2:
        st.metric("Columns", dashboard.dataset.shape[1])
    with col3:
        st.metric("Charts", len(dashboard.charts))
    with col4:
        st.metric("Numeric Cols", len(dashboard.dataset_analysis.get('numeric_columns', [])))
    
    # Add new visualization section
    with st.expander("➕ Add New Visualization"):
        add_visualization_form(dashboard, dashboard.dataset)
    
    # Main dashboard grid
    if dashboard.charts:
        st.header("🎨 Dashboard Canvas")
        
        # Render charts in grid layout
        for i in range(0, len(dashboard.charts), 2):
            cols = st.columns(2)
            
            # First chart in row
            with cols[0]:
                render_chart_card(dashboard.charts[i], dashboard.dataset, dashboard)
            
            # Second chart in row (if exists)
            if i + 1 < len(dashboard.charts):
                with cols[1]:
                    render_chart_card(dashboard.charts[i + 1], dashboard.dataset, dashboard)
        
        # Bulk export section
        st.header("📤 Export Dashboard")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export All Charts (HTML)"):
                # Create ZIP with all charts
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for chart in dashboard.charts:
                        try:
                            fig = create_plotly_chart(chart, dashboard.dataset)
                            chart_html = fig.to_html()
                            filename = f"{chart.title.replace(' ', '_')}.html"
                            zip_file.writestr(filename, chart_html)
                        except Exception as e:
                            st.warning(f"Skipped {chart.title}: {str(e)}")
                
                zip_buffer.seek(0)
                st.download_button(
                    "📥 Download All Charts (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"{dashboard.dashboard_name.replace(' ', '_')}_charts.zip",
                    mime="application/zip"
                )
        
        with col2:
            if st.button("📊 Export Data Summary"):
                # Create comprehensive data summary
                summary_html = f"""
                <html>
                <head><title>{dashboard.dashboard_name} - Data Summary</title></head>
                <body>
                <h1>{dashboard.dashboard_name}</h1>
                <h2>Dataset Overview</h2>
                <p>Rows: {dashboard.dataset.shape[0]}</p>
                <p>Columns: {dashboard.dataset.shape[1]}</p>
                <h2>Data Sample</h2>
                {dashboard.dataset.head(10).to_html()}
                <h2>Statistical Summary</h2>
                {dashboard.dataset.describe().to_html()}
                </body>
                </html>
                """
                st.download_button(
                    "📥 Download Summary",
                    data=summary_html,
                    file_name=f"{dashboard.dashboard_name.replace(' ', '_')}_summary.html",
                    mime="text/html"
                )
    
    else:
        st.info("🎨 No visualizations yet. Upload data and click 'Auto-Generate Dashboard' or add charts manually!")

if __name__ == "__main__":
    main()