#!/usr/bin/env python3

"""
Enhanced Canvas Dashboard with Advanced Features
- Drag and Drop Chart Reordering
- Cross-filtering Between Charts  
- Advanced Chart Customization
- Real-time Updates
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import zipfile
import json
import uuid

# ==========================================
# ENHANCED CHART CONFIGURATIONS
# ==========================================

class AdvancedChartConfig:
    """Enhanced chart configuration with filtering and interactions"""
    def __init__(self, chart_id: str = None, chart_type: str = "scatter", 
                 title: str = "", x_column: str = "", y_column: str = "", 
                 category_column: str = "", size: str = "medium", 
                 position: int = 0, filters: dict = None, 
                 color_scheme: str = "plotly", custom_params: dict = None):
        self.chart_id = chart_id or str(uuid.uuid4())
        self.chart_type = chart_type
        self.title = title
        self.x_column = x_column
        self.y_column = y_column
        self.category_column = category_column
        self.size = size
        self.position = position
        self.filters = filters or {}
        self.color_scheme = color_scheme
        self.custom_params = custom_params or {}
        self.last_updated = datetime.now()
    
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
            'filters': self.filters,
            'color_scheme': self.color_scheme,
            'custom_params': self.custom_params
        }

class DashboardManager:
    """Advanced dashboard management with cross-filtering"""
    def __init__(self):
        self.charts: List[AdvancedChartConfig] = []
        self.dataset: pd.DataFrame = None
        self.filtered_dataset: pd.DataFrame = None
        self.global_filters: Dict[str, Any] = {}
        self.dataset_analysis: dict = {}
        self.dashboard_name: str = "Advanced Dashboard"
        self.theme: str = "plotly_white"
        self.created_at: str = datetime.now().isoformat()
        self.auto_update: bool = True
    
    def apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply global filters to dataset"""
        filtered_data = data.copy()
        
        for column, filter_value in self.global_filters.items():
            if column in filtered_data.columns:
                if isinstance(filter_value, list) and filter_value:
                    filtered_data = filtered_data[filtered_data[column].isin(filter_value)]
                elif isinstance(filter_value, dict):
                    if 'min' in filter_value and filter_value['min'] is not None:
                        filtered_data = filtered_data[filtered_data[column] >= filter_value['min']]
                    if 'max' in filter_value and filter_value['max'] is not None:
                        filtered_data = filtered_data[filtered_data[column] <= filter_value['max']]
        
        return filtered_data
    
    def add_chart(self, chart_config: AdvancedChartConfig):
        self.charts.append(chart_config)
        self.sort_charts()
    
    def remove_chart(self, chart_id: str):
        self.charts = [c for c in self.charts if c.chart_id != chart_id]
        self.sort_charts()
    
    def sort_charts(self):
        self.charts.sort(key=lambda x: x.position)
    
    def duplicate_chart(self, chart_id: str):
        original = next((c for c in self.charts if c.chart_id == chart_id), None)
        if original:
            duplicate = AdvancedChartConfig(
                chart_type=original.chart_type,
                title=f"{original.title} (Copy)",
                x_column=original.x_column,
                y_column=original.y_column,
                category_column=original.category_column,
                size=original.size,
                position=len(self.charts),
                filters=original.filters.copy(),
                color_scheme=original.color_scheme,
                custom_params=original.custom_params.copy()
            )
            self.add_chart(duplicate)
            return duplicate
        return None

# ==========================================
# ENHANCED CHART CREATION
# ==========================================

def create_advanced_plotly_chart(chart_config: AdvancedChartConfig, data: pd.DataFrame, theme: str = "plotly_white") -> go.Figure:
    """Create advanced interactive Plotly charts with enhanced features"""
    
    # Apply chart-specific filters
    filtered_data = data.copy()
    for column, filter_value in chart_config.filters.items():
        if column in filtered_data.columns and filter_value:
            if isinstance(filter_value, list):
                filtered_data = filtered_data[filtered_data[column].isin(filter_value)]
    
    color_schemes = {
        "plotly": px.colors.qualitative.Plotly,
        "vivid": px.colors.qualitative.Vivid,
        "pastel": px.colors.qualitative.Pastel,
        "dark24": px.colors.qualitative.Dark24
    }
    
    color_palette = color_schemes.get(chart_config.color_scheme, px.colors.qualitative.Plotly)
    
    if chart_config.chart_type == "scatter":
        fig = px.scatter(
            filtered_data, 
            x=chart_config.x_column, 
            y=chart_config.y_column,
            color=chart_config.category_column if chart_config.category_column else None,
            title=chart_config.title,
            template=theme,
            color_discrete_sequence=color_palette,
            hover_data=filtered_data.columns[:5].tolist()  # Show first 5 columns on hover
        )
        
        # Add trend line if enabled
        if chart_config.custom_params.get('trendline', False):
            fig.add_trace(go.Scatter(
                x=filtered_data[chart_config.x_column],
                y=filtered_data[chart_config.x_column].rolling(window=10).mean(),
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='red')
            ))
            
    elif chart_config.chart_type == "histogram":
        bins = chart_config.custom_params.get('bins', 30)
        fig = px.histogram(
            filtered_data, 
            x=chart_config.x_column,
            color=chart_config.category_column if chart_config.category_column else None,
            title=chart_config.title,
            template=theme,
            nbins=bins,
            color_discrete_sequence=color_palette
        )
        
        # Add statistical lines
        if chart_config.custom_params.get('show_stats', True):
            mean_val = filtered_data[chart_config.x_column].mean()
            median_val = filtered_data[chart_config.x_column].median()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.2f}")
            fig.add_vline(x=median_val, line_dash="dash", line_color="blue", 
                         annotation_text=f"Median: {median_val:.2f}")
    
    elif chart_config.chart_type == "box":
        fig = px.box(
            filtered_data, 
            x=chart_config.category_column,
            y=chart_config.x_column,
            title=chart_config.title,
            template=theme,
            color=chart_config.category_column,
            color_discrete_sequence=color_palette
        )
        
    elif chart_config.chart_type == "violin":
        fig = px.violin(
            filtered_data, 
            x=chart_config.category_column,
            y=chart_config.x_column,
            title=chart_config.title,
            template=theme,
            color=chart_config.category_column,
            color_discrete_sequence=color_palette
        )
        
    elif chart_config.chart_type == "bar":
        if chart_config.custom_params.get('aggregation') == 'mean' and chart_config.y_column:
            # Grouped bar chart with aggregation
            agg_data = filtered_data.groupby(chart_config.x_column)[chart_config.y_column].mean().reset_index()
            fig = px.bar(
                agg_data,
                x=chart_config.x_column, 
                y=chart_config.y_column,
                title=chart_config.title,
                template=theme,
                color=chart_config.x_column,
                color_discrete_sequence=color_palette
            )
        else:
            # Standard count bar chart
            value_counts = filtered_data[chart_config.x_column].value_counts().head(20)
            fig = px.bar(
                x=value_counts.index, 
                y=value_counts.values,
                title=chart_config.title,
                template=theme,
                color=value_counts.index,
                color_discrete_sequence=color_palette
            )
            fig.update_xaxes(title=chart_config.x_column)
            fig.update_yaxes(title="Count")
        
    elif chart_config.chart_type == "pie":
        value_counts = filtered_data[chart_config.x_column].value_counts().head(10)
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=chart_config.title,
            template=theme,
            color_discrete_sequence=color_palette
        )
        
    elif chart_config.chart_type == "line":
        fig = px.line(
            filtered_data, 
            x=chart_config.x_column, 
            y=chart_config.y_column,
            color=chart_config.category_column if chart_config.category_column else None,
            title=chart_config.title,
            template=theme,
            color_discrete_sequence=color_palette
        )
        
    elif chart_config.chart_type == "heatmap":
        if chart_config.custom_params.get('correlation', True):
            # Correlation heatmap
            numeric_data = filtered_data.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.around(corr_matrix.values, decimals=2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
        else:
            # Pivot table heatmap
            if chart_config.x_column and chart_config.y_column and chart_config.category_column:
                pivot_data = filtered_data.pivot_table(
                    values=chart_config.category_column,
                    index=chart_config.x_column,
                    columns=chart_config.y_column,
                    aggfunc='mean'
                )
                fig = px.imshow(pivot_data, title=chart_config.title, template=theme)
            else:
                # Fallback to correlation
                numeric_data = filtered_data.select_dtypes(include=[np.number])
                corr_matrix = numeric_data.corr()
                fig = px.imshow(corr_matrix, title=chart_config.title, template=theme)
        
        fig.update_layout(title=chart_config.title)
        
    elif chart_config.chart_type == "sunburst":
        # New chart type: Sunburst for hierarchical data
        if len([c for c in [chart_config.x_column, chart_config.y_column, chart_config.category_column] if c]) >= 2:
            path_columns = [c for c in [chart_config.x_column, chart_config.y_column, chart_config.category_column] if c][:3]
            fig = px.sunburst(
                filtered_data,
                path=path_columns,
                title=chart_config.title,
                color_discrete_sequence=color_palette
            )
        else:
            # Fallback to pie chart
            value_counts = filtered_data[chart_config.x_column].value_counts().head(10)
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=chart_config.title)
    
    elif chart_config.chart_type == "treemap":
        # New chart type: Treemap
        if chart_config.x_column and chart_config.y_column:
            fig = px.treemap(
                filtered_data,
                path=[chart_config.x_column],
                values=chart_config.y_column,
                title=chart_config.title,
                color_discrete_sequence=color_palette
            )
        else:
            # Fallback using value counts
            value_counts = filtered_data[chart_config.x_column].value_counts().head(15)
            fig = px.treemap(
                names=value_counts.index,
                values=value_counts.values,
                title=chart_config.title
            )
    
    else:
        # Default to scatter
        fig = px.scatter(filtered_data, x=filtered_data.columns[0], y=filtered_data.columns[1], title=chart_config.title)
    
    # Update layout based on size
    height_map = {"small": 350, "medium": 450, "large": 550, "full": 700}
    fig.update_layout(
        height=height_map.get(chart_config.size, 450),
        margin=dict(t=50, l=50, r=50, b=50),
        showlegend=True
    )
    
    # Add interactivity
    fig.update_traces(
        selector=dict(mode="markers"),
        hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>%{y}<extra></extra>"
    )
    
    return fig

# ==========================================
# ENHANCED UI COMPONENTS
# ==========================================

def render_advanced_chart_card(chart_config: AdvancedChartConfig, data: pd.DataFrame, dashboard: DashboardManager):
    """Render advanced chart card with enhanced controls"""
    
    # Chart type icons with new types
    chart_icons = {
        'scatter': '🔵', 'histogram': '📊', 'bar': '📈', 'box': '📦',
        'pie': '🥧', 'line': '📉', 'heatmap': '🔥', 'violin': '🎻',
        'sunburst': '☀️', 'treemap': '🌳'
    }
    
    icon = chart_icons.get(chart_config.chart_type, '📊')
    
    with st.container():
        # Enhanced header with more controls
        col1, col2, col3, col4, col5, col6 = st.columns([2, 0.8, 0.8, 0.8, 0.8, 0.8])
        
        with col1:
            st.markdown(f"### {icon} {chart_config.title}")
        
        with col2:
            # Size control
            new_size = st.selectbox(
                "📏", 
                ["small", "medium", "large", "full"], 
                index=["small", "medium", "large", "full"].index(chart_config.size),
                key=f"size_{chart_config.chart_id}",
                help="Chart Size"
            )
            if new_size != chart_config.size:
                chart_config.size = new_size
                st.rerun()
        
        with col3:
            # Color scheme
            new_scheme = st.selectbox(
                "🎨",
                ["plotly", "vivid", "pastel", "dark24"],
                index=["plotly", "vivid", "pastel", "dark24"].index(chart_config.color_scheme),
                key=f"color_{chart_config.chart_id}",
                help="Color Scheme"
            )
            if new_scheme != chart_config.color_scheme:
                chart_config.color_scheme = new_scheme
                st.rerun()
        
        with col4:
            # Duplicate button
            if st.button("📋", key=f"duplicate_{chart_config.chart_id}", help="Duplicate Chart"):
                dashboard.duplicate_chart(chart_config.chart_id)
                st.rerun()
        
        with col5:
            # Edit button
            if st.button("✏️", key=f"edit_{chart_config.chart_id}", help="Edit Chart"):
                st.session_state[f"editing_{chart_config.chart_id}"] = True
        
        with col6:
            # Remove button
            if st.button("🗑️", key=f"remove_{chart_config.chart_id}", help="Remove Chart"):
                dashboard.remove_chart(chart_config.chart_id)
                st.rerun()
        
        # Quick filters for this chart
        if chart_config.category_column and chart_config.category_column in data.columns:
            unique_values = data[chart_config.category_column].unique()[:20]  # Limit to 20 for performance
            if len(unique_values) > 1:
                selected_values = st.multiselect(
                    f"Filter {chart_config.category_column}:",
                    options=unique_values,
                    default=chart_config.filters.get(chart_config.category_column, unique_values),
                    key=f"filter_{chart_config.chart_id}"
                )
                chart_config.filters[chart_config.category_column] = selected_values
        
        # Edit form (if editing)
        if st.session_state.get(f"editing_{chart_config.chart_id}", False):
            with st.expander("📝 Advanced Chart Editor", expanded=True):
                edit_advanced_chart_form(chart_config, data, dashboard)
        
        # Render the actual chart
        try:
            # Apply global filters
            filtered_data = dashboard.apply_filters(data)
            
            fig = create_advanced_plotly_chart(chart_config, filtered_data, dashboard.theme)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_config.chart_id}")
            
            # Chart statistics
            with st.expander("📊 Chart Stats"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Points", len(filtered_data))
                with col2:
                    if chart_config.x_column in filtered_data.columns:
                        if filtered_data[chart_config.x_column].dtype in ['int64', 'float64']:
                            st.metric("X Mean", f"{filtered_data[chart_config.x_column].mean():.2f}")
                        else:
                            st.metric("X Unique", filtered_data[chart_config.x_column].nunique())
                with col3:
                    if chart_config.y_column and chart_config.y_column in filtered_data.columns:
                        if filtered_data[chart_config.y_column].dtype in ['int64', 'float64']:
                            st.metric("Y Mean", f"{filtered_data[chart_config.y_column].mean():.2f}")
                        else:
                            st.metric("Y Unique", filtered_data[chart_config.y_column].nunique())
            
            # Download options
            col1, col2, col3 = st.columns(3)
            with col1:
                chart_html = fig.to_html()
                st.download_button(
                    "📥 HTML",
                    data=chart_html,
                    file_name=f"{chart_config.title.replace(' ', '_')}.html",
                    mime="text/html",
                    key=f"download_html_{chart_config.chart_id}"
                )
            with col2:
                chart_json = fig.to_json()
                st.download_button(
                    "📥 JSON",
                    data=chart_json,
                    file_name=f"{chart_config.title.replace(' ', '_')}.json",
                    mime="application/json",
                    key=f"download_json_{chart_config.chart_id}"
                )
            with col3:
                # Export filtered data
                csv_data = filtered_data.to_csv(index=False)
                st.download_button(
                    "📥 Data",
                    data=csv_data,
                    file_name=f"{chart_config.title.replace(' ', '_')}_data.csv",
                    mime="text/csv",
                    key=f"download_data_{chart_config.chart_id}"
                )
            
        except Exception as e:
            st.error(f"❌ Error creating {chart_config.chart_type}: {str(e)}")
            with st.expander("🐛 Debug Info"):
                st.code(f"Config: {chart_config.to_dict()}")
                st.code(f"Error: {str(e)}")

def edit_advanced_chart_form(chart_config: AdvancedChartConfig, data: pd.DataFrame, dashboard: DashboardManager):
    """Advanced form for editing chart configuration"""
    
    # Chart type selection with new types
    chart_types = ["scatter", "histogram", "bar", "box", "pie", "line", "heatmap", "violin", "sunburst", "treemap"]
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
    
    # Column selections
    numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
    all_cols = list(data.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic configuration
        if new_type in ["scatter", "line"]:
            new_x = st.selectbox("X Column", all_cols, 
                               index=all_cols.index(chart_config.x_column) if chart_config.x_column in all_cols else 0,
                               key=f"x_edit_{chart_config.chart_id}")
            new_y = st.selectbox("Y Column", all_cols,
                               index=all_cols.index(chart_config.y_column) if chart_config.y_column in all_cols else 0,
                               key=f"y_edit_{chart_config.chart_id}")
            new_category = st.selectbox("Color By (Optional)", [""] + categorical_cols,
                                      index=([""] + categorical_cols).index(chart_config.category_column) if chart_config.category_column in ([""] + categorical_cols) else 0,
                                      key=f"cat_edit_{chart_config.chart_id}")
        
        elif new_type in ["histogram", "bar", "pie"]:
            new_x = st.selectbox("Column", all_cols,
                               index=all_cols.index(chart_config.x_column) if chart_config.x_column in all_cols else 0,
                               key=f"x_edit_{chart_config.chart_id}")
            new_y = st.selectbox("Y Column (Optional)", [""] + numeric_cols,
                               key=f"y_edit_{chart_config.chart_id}")
            new_category = st.selectbox("Group By (Optional)", [""] + categorical_cols,
                                      key=f"cat_edit_{chart_config.chart_id}")
        
        elif new_type in ["box", "violin"]:
            new_x = st.selectbox("Numeric Column", numeric_cols,
                               index=numeric_cols.index(chart_config.x_column) if chart_config.x_column in numeric_cols else 0,
                               key=f"x_edit_{chart_config.chart_id}")
            new_y = ""
            new_category = st.selectbox("Category Column", categorical_cols,
                                      index=categorical_cols.index(chart_config.category_column) if chart_config.category_column in categorical_cols else 0,
                                      key=f"cat_edit_{chart_config.chart_id}")
        
        elif new_type in ["sunburst", "treemap"]:
            new_x = st.selectbox("Primary Column", all_cols,
                               key=f"x_edit_{chart_config.chart_id}")
            new_y = st.selectbox("Value Column (Optional)", [""] + numeric_cols,
                               key=f"y_edit_{chart_config.chart_id}")
            new_category = st.selectbox("Secondary Column (Optional)", [""] + all_cols,
                                      key=f"cat_edit_{chart_config.chart_id}")
        
        else:  # heatmap
            new_x = ""
            new_y = ""
            new_category = ""
    
    with col2:
        # Advanced configuration
        st.subheader("⚙️ Advanced Settings")
        
        if new_type == "histogram":
            bins = st.slider("Number of Bins", 10, 100, 
                           chart_config.custom_params.get('bins', 30),
                           key=f"bins_{chart_config.chart_id}")
            show_stats = st.checkbox("Show Statistics Lines", 
                                   chart_config.custom_params.get('show_stats', True),
                                   key=f"stats_{chart_config.chart_id}")
            chart_config.custom_params['bins'] = bins
            chart_config.custom_params['show_stats'] = show_stats
        
        elif new_type == "scatter":
            trendline = st.checkbox("Show Trend Line", 
                                  chart_config.custom_params.get('trendline', False),
                                  key=f"trend_{chart_config.chart_id}")
            chart_config.custom_params['trendline'] = trendline
        
        elif new_type == "bar":
            if new_y:
                aggregation = st.selectbox("Aggregation", ["count", "mean", "sum"],
                                         index=["count", "mean", "sum"].index(chart_config.custom_params.get('aggregation', 'count')),
                                         key=f"agg_{chart_config.chart_id}")
                chart_config.custom_params['aggregation'] = aggregation
        
        elif new_type == "heatmap":
            correlation = st.checkbox("Show Correlation Matrix", 
                                    chart_config.custom_params.get('correlation', True),
                                    key=f"corr_{chart_config.chart_id}")
            chart_config.custom_params['correlation'] = correlation
    
    # Update and save buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Changes", key=f"save_{chart_config.chart_id}"):
            chart_config.chart_type = new_type
            chart_config.title = new_title
            chart_config.x_column = new_x
            chart_config.y_column = new_y
            chart_config.category_column = new_category
            chart_config.last_updated = datetime.now()
            st.session_state[f"editing_{chart_config.chart_id}"] = False
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel", key=f"cancel_{chart_config.chart_id}"):
            st.session_state[f"editing_{chart_config.chart_id}"] = False
            st.rerun()

def render_global_filters(dashboard: DashboardManager):
    """Render global filters sidebar"""
    if dashboard.dataset is None:
        return
    
    st.subheader("🔍 Global Filters")
    
    # Numeric filters
    numeric_cols = dashboard.dataset.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
        col_min = float(dashboard.dataset[col].min())
        col_max = float(dashboard.dataset[col].max())
        
        current_range = dashboard.global_filters.get(col, {'min': col_min, 'max': col_max})
        
        new_range = st.slider(
            f"{col} Range",
            col_min, col_max,
            (current_range.get('min', col_min), current_range.get('max', col_max)),
            key=f"global_filter_{col}"
        )
        
        if new_range != (current_range.get('min', col_min), current_range.get('max', col_max)):
            dashboard.global_filters[col] = {'min': new_range[0], 'max': new_range[1]}
    
    # Categorical filters
    categorical_cols = dashboard.dataset.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        unique_values = dashboard.dataset[col].unique()[:20]  # Limit options
        
        current_selection = dashboard.global_filters.get(col, unique_values)
        if not isinstance(current_selection, list):
            current_selection = list(unique_values)
        
        new_selection = st.multiselect(
            f"Filter {col}",
            options=unique_values,
            default=current_selection,
            key=f"global_cat_filter_{col}"
        )
        
        dashboard.global_filters[col] = new_selection
    
    # Clear all filters
    if st.button("🧹 Clear All Filters"):
        dashboard.global_filters = {}
        st.rerun()

# ==========================================
# MAIN ENHANCED APPLICATION
# ==========================================

def main():
    st.title("🎨 Advanced Multi-Visualization Canvas")
    st.markdown("*Interactive Dashboard with Cross-Filtering & Advanced Customization*")
    st.markdown("---")
    
    # Initialize session state
    if 'advanced_dashboard' not in st.session_state:
        st.session_state.advanced_dashboard = DashboardManager()
    
    dashboard = st.session_state.advanced_dashboard
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Control Center")
        
        # File upload section
        st.subheader("📁 Data Management")
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                dashboard.dataset = df
                dashboard.dataset_analysis = analyze_dataset(df)
                st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Auto-generate enhanced dashboard
                if st.button("🚀 Auto-Generate Advanced Dashboard"):
                    dashboard.charts = []
                    numeric_cols = dashboard.dataset_analysis.get('numeric_columns', [])
                    categorical_cols = dashboard.dataset_analysis.get('categorical_columns', [])
                    
                    auto_charts = []
                    position = 0
                    
                    # Enhanced auto-generation with new chart types
                    if len(numeric_cols) >= 2:
                        auto_charts.append(AdvancedChartConfig(
                            chart_type="scatter", title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                            x_column=numeric_cols[0], y_column=numeric_cols[1],
                            category_column=categorical_cols[0] if categorical_cols else "",
                            size="large", position=position
                        ))
                        position += 1
                    
                    if len(numeric_cols) >= 1:
                        auto_charts.append(AdvancedChartConfig(
                            chart_type="histogram", title=f"Distribution of {numeric_cols[0]}",
                            x_column=numeric_cols[0], size="medium", position=position
                        ))
                        position += 1
                    
                    if len(categorical_cols) >= 1:
                        auto_charts.append(AdvancedChartConfig(
                            chart_type="sunburst", title=f"Hierarchical View of {categorical_cols[0]}",
                            x_column=categorical_cols[0], size="large", position=position
                        ))
                        position += 1
                    
                    if len(numeric_cols) >= 3:
                        auto_charts.append(AdvancedChartConfig(
                            chart_type="heatmap", title="Correlation Analysis",
                            size="large", position=position
                        ))
                        position += 1
                    
                    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                        auto_charts.append(AdvancedChartConfig(
                            chart_type="treemap", title=f"{categorical_cols[0]} Treemap",
                            x_column=categorical_cols[0], y_column=numeric_cols[0],
                            size="medium", position=position
                        ))
                        position += 1
                    
                    for chart in auto_charts:
                        dashboard.add_chart(chart)
                    
                    st.success(f"Generated {len(auto_charts)} advanced visualizations!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        # Global filters
        if dashboard.dataset is not None:
            render_global_filters(dashboard)
            
            st.markdown("---")
            
            # Dashboard settings
            st.subheader("⚙️ Dashboard Settings")
            
            dashboard.dashboard_name = st.text_input("Dashboard Name", dashboard.dashboard_name)
            
            dashboard.theme = st.selectbox("Theme", 
                                         ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
                                         index=["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"].index(dashboard.theme))
            
            dashboard.auto_update = st.checkbox("Auto-Update Charts", dashboard.auto_update)
            
            st.markdown("---")
            
            # Export/Import
            st.subheader("💾 Dashboard Persistence")
            
            if st.button("📤 Export Dashboard Config"):
                config_data = {
                    'dashboard_name': dashboard.dashboard_name,
                    'theme': dashboard.theme,
                    'charts': [chart.to_dict() for chart in dashboard.charts],
                    'global_filters': dashboard.global_filters,
                    'created_at': dashboard.created_at
                }
                config_json = json.dumps(config_data, indent=2)
                st.download_button(
                    "📥 Download Config",
                    data=config_json,
                    file_name=f"{dashboard.dashboard_name.replace(' ', '_')}_advanced_config.json",
                    mime="application/json"
                )
            
            config_file = st.file_uploader("📥 Import Dashboard Config", type=['json'])
            if config_file is not None:
                try:
                    config_data = json.loads(config_file.read())
                    dashboard.dashboard_name = config_data.get('dashboard_name', 'Imported Dashboard')
                    dashboard.theme = config_data.get('theme', 'plotly_white')
                    dashboard.global_filters = config_data.get('global_filters', {})
                    dashboard.charts = [AdvancedChartConfig(**chart_data) for chart_data in config_data.get('charts', [])]
                    st.success("✅ Dashboard imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error importing: {str(e)}")
    
    # Main content area
    if dashboard.dataset is None:
        st.info("👈 Please upload a CSV file to get started with the advanced dashboard")
        return
    
    # Dashboard header with enhanced metrics
    st.header(f"🎨 {dashboard.dashboard_name}")
    
    # Apply global filters for metrics
    filtered_data = dashboard.apply_filters(dashboard.dataset)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Rows", dashboard.dataset.shape[0])
    with col2:
        st.metric("Filtered Rows", len(filtered_data), 
                 delta=len(filtered_data) - dashboard.dataset.shape[0])
    with col3:
        st.metric("Total Charts", len(dashboard.charts))
    with col4:
        st.metric("Active Filters", len([f for f in dashboard.global_filters.values() if f]))
    with col5:
        data_quality = ((dashboard.dataset.shape[0] * dashboard.dataset.shape[1] - dashboard.dataset.isnull().sum().sum()) / 
                       (dashboard.dataset.shape[0] * dashboard.dataset.shape[1]) * 100)
        st.metric("Data Quality", f"{data_quality:.1f}%")
    
    # Add new visualization section (enhanced)
    with st.expander("➕ Add Advanced Visualization", expanded=False):
        st.subheader("🎯 Create New Chart")
        
        tab1, tab2 = st.tabs(["🎨 Quick Add", "⚙️ Advanced Add"])
        
        with tab1:
            # Quick add form
            col1, col2 = st.columns(2)
            with col1:
                chart_type = st.selectbox("Chart Type", 
                                        ["scatter", "histogram", "bar", "box", "pie", "line", "heatmap", "violin", "sunburst", "treemap"])
                title = st.text_input("Chart Title", value=f"New {chart_type.title()} Chart")
            
            with col2:
                size = st.selectbox("Size", ["small", "medium", "large", "full"], index=1)
                color_scheme = st.selectbox("Color Scheme", ["plotly", "vivid", "pastel", "dark24"])
            
            if st.button("🎨 Add Quick Chart"):
                # Auto-configure based on data types
                numeric_cols = list(dashboard.dataset.select_dtypes(include=[np.number]).columns)
                categorical_cols = list(dashboard.dataset.select_dtypes(include=['object', 'category']).columns)
                
                x_col = numeric_cols[0] if numeric_cols else (categorical_cols[0] if categorical_cols else dashboard.dataset.columns[0])
                y_col = numeric_cols[1] if len(numeric_cols) > 1 else ""
                cat_col = categorical_cols[0] if categorical_cols else ""
                
                new_chart = AdvancedChartConfig(
                    chart_type=chart_type, title=title,
                    x_column=x_col, y_column=y_col, category_column=cat_col,
                    size=size, color_scheme=color_scheme,
                    position=len(dashboard.charts)
                )
                dashboard.add_chart(new_chart)
                st.success(f"Added {chart_type} chart!")
                st.rerun()
        
        with tab2:
            # Advanced add form (detailed configuration)
            with st.form("advanced_add_chart"):
                col1, col2 = st.columns(2)
                
                with col1:
                    chart_type = st.selectbox("Chart Type", 
                                            ["scatter", "histogram", "bar", "box", "pie", "line", "heatmap", "violin", "sunburst", "treemap"])
                    title = st.text_input("Chart Title")
                    size = st.selectbox("Size", ["small", "medium", "large", "full"])
                
                with col2:
                    color_scheme = st.selectbox("Color Scheme", ["plotly", "vivid", "pastel", "dark24"])
                    
                    # Column selections
                    numeric_cols = list(dashboard.dataset.select_dtypes(include=[np.number]).columns)
                    categorical_cols = list(dashboard.dataset.select_dtypes(include=['object', 'category']).columns)
                    all_cols = list(dashboard.dataset.columns)
                
                # Dynamic column selection based on chart type
                if chart_type in ["scatter", "line"]:
                    x_col = st.selectbox("X Column", all_cols)
                    y_col = st.selectbox("Y Column", all_cols)
                    cat_col = st.selectbox("Color By (Optional)", [""] + categorical_cols)
                elif chart_type in ["histogram", "bar", "pie"]:
                    x_col = st.selectbox("Column", all_cols)
                    y_col = st.selectbox("Value Column (Optional)", [""] + numeric_cols)
                    cat_col = st.selectbox("Group By (Optional)", [""] + categorical_cols)
                elif chart_type in ["box", "violin"]:
                    x_col = st.selectbox("Numeric Column", numeric_cols) if numeric_cols else ""
                    y_col = ""
                    cat_col = st.selectbox("Category Column", categorical_cols) if categorical_cols else ""
                elif chart_type in ["sunburst", "treemap"]:
                    x_col = st.selectbox("Primary Column", all_cols)
                    y_col = st.selectbox("Value Column (Optional)", [""] + numeric_cols)
                    cat_col = st.selectbox("Secondary Column (Optional)", [""] + all_cols)
                else:  # heatmap
                    x_col = ""
                    y_col = ""
                    cat_col = ""
                
                if st.form_submit_button("🎨 Create Advanced Chart"):
                    new_chart = AdvancedChartConfig(
                        chart_type=chart_type, title=title,
                        x_column=x_col, y_column=y_col, category_column=cat_col,
                        size=size, color_scheme=color_scheme,
                        position=len(dashboard.charts)
                    )
                    dashboard.add_chart(new_chart)
                    st.success(f"Created advanced {chart_type} chart!")
                    st.rerun()
    
    # Main dashboard grid (enhanced)
    if dashboard.charts:
        st.header("🎨 Interactive Dashboard Canvas")
        
        # Chart reordering controls
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{len(dashboard.charts)} visualizations** • Last updated: {datetime.now().strftime('%H:%M:%S')}")
        with col2:
            if st.button("🔄 Refresh All Charts"):
                st.rerun()
        
        # Render charts in responsive grid
        for i in range(0, len(dashboard.charts), 2):
            # Create responsive columns based on chart sizes
            chart1 = dashboard.charts[i]
            chart2 = dashboard.charts[i + 1] if i + 1 < len(dashboard.charts) else None
            
            if chart1.size == "full" or (chart2 and chart2.size == "full"):
                # Full width layout
                render_advanced_chart_card(chart1, dashboard.dataset, dashboard)
                if chart2:
                    render_advanced_chart_card(chart2, dashboard.dataset, dashboard)
            else:
                # Two column layout
                cols = st.columns(2)
                with cols[0]:
                    render_advanced_chart_card(chart1, dashboard.dataset, dashboard)
                if chart2:
                    with cols[1]:
                        render_advanced_chart_card(chart2, dashboard.dataset, dashboard)
        
        # Enhanced export section
        st.header("📤 Export & Share Dashboard")
        
        tab1, tab2, tab3 = st.tabs(["📊 Charts Export", "📋 Data Export", "🔗 Share"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export All Charts (HTML)"):
                    # Enhanced bulk export
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # Export individual charts
                        for chart in dashboard.charts:
                            try:
                                filtered_data = dashboard.apply_filters(dashboard.dataset)
                                fig = create_advanced_plotly_chart(chart, filtered_data, dashboard.theme)
                                chart_html = fig.to_html()
                                filename = f"{chart.title.replace(' ', '_')}.html"
                                zip_file.writestr(filename, chart_html)
                            except Exception as e:
                                st.warning(f"Skipped {chart.title}: {str(e)}")
                        
                        # Export dashboard config
                        config_data = {
                            'dashboard_name': dashboard.dashboard_name,
                            'theme': dashboard.theme,
                            'charts': [chart.to_dict() for chart in dashboard.charts],
                            'export_date': datetime.now().isoformat()
                        }
                        zip_file.writestr("dashboard_config.json", json.dumps(config_data, indent=2))
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        "📥 Download Charts Package",
                        data=zip_buffer.getvalue(),
                        file_name=f"{dashboard.dashboard_name.replace(' ', '_')}_complete_export.zip",
                        mime="application/zip"
                    )
            
            with col2:
                if st.button("🖼️ Export as Images"):
                    st.info("📸 Image export would require additional setup with kaleido/orca")
            
            with col3:
                if st.button("📱 Create Mobile Dashboard"):
                    st.info("📱 Mobile optimization feature - coming soon!")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Export filtered data
                filtered_data = dashboard.apply_filters(dashboard.dataset)
                csv_data = filtered_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Filtered Data (CSV)",
                    data=csv_data,
                    file_name=f"{dashboard.dashboard_name.replace(' ', '_')}_filtered_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export data summary
                summary_html = f"""
                <html>
                <head><title>{dashboard.dashboard_name} - Enhanced Summary</title></head>
                <body style="font-family: Arial, sans-serif;">
                <h1>🎨 {dashboard.dashboard_name}</h1>
                <h2>📊 Dataset Overview</h2>
                <ul>
                    <li><strong>Total Rows:</strong> {dashboard.dataset.shape[0]:,}</li>
                    <li><strong>Filtered Rows:</strong> {len(filtered_data):,}</li>
                    <li><strong>Columns:</strong> {dashboard.dataset.shape[1]}</li>
                    <li><strong>Charts:</strong> {len(dashboard.charts)}</li>
                    <li><strong>Active Filters:</strong> {len([f for f in dashboard.global_filters.values() if f])}</li>
                </ul>
                <h2>📈 Charts Summary</h2>
                <ul>
                """
                
                for chart in dashboard.charts:
                    summary_html += f"<li><strong>{chart.title}</strong> ({chart.chart_type}) - {chart.size} size</li>"
                
                summary_html += f"""
                </ul>
                <h2>📋 Data Sample</h2>
                {filtered_data.head(10).to_html(classes='table table-striped')}
                <h2>📊 Statistical Summary</h2>
                {filtered_data.describe().to_html(classes='table table-striped')}
                <footer>
                <p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Advanced Canvas Dashboard</em></p>
                </footer>
                </body>
                </html>
                """
                
                st.download_button(
                    "📥 Download Enhanced Summary",
                    data=summary_html,
                    file_name=f"{dashboard.dashboard_name.replace(' ', '_')}_enhanced_summary.html",
                    mime="text/html"
                )
        
        with tab3:
            st.info("🔗 **Sharing Features Coming Soon:**")
            st.write("• Generate shareable dashboard links")
            st.write("• Embed dashboard in websites")
            st.write("• Real-time collaboration")
            st.write("• Dashboard templates library")
    
    else:
        st.info("🎨 No visualizations yet. Upload data and click 'Auto-Generate Advanced Dashboard' or add charts manually!")
        
        # Showcase section for empty dashboard
        st.subheader("✨ What You Can Create:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Interactive Charts**")
            st.write("• Scatter plots with trend lines")
            st.write("• Dynamic histograms")
            st.write("• Hierarchical sunbursts")
        
        with col2:
            st.write("**🎨 Advanced Features**")
            st.write("• Cross-chart filtering")
            st.write("• Custom color schemes")
            st.write("• Responsive sizing")
        
        with col3:
            st.write("**💾 Smart Management**")
            st.write("• Save/load configurations")
            st.write("• Bulk data exports")
            st.write("• Dashboard sharing")

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Enhanced dataset analysis for advanced dashboard"""
    analysis = {
        'shape': df.shape,
        'columns': list(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_columns': [],
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    # Detect datetime columns
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                pd.to_datetime(df[col])
                analysis['datetime_columns'].append(col)
            except:
                pass
    
    # Enhanced categorical analysis
    analysis['categorical_summary'] = {}
    for col in analysis['categorical_columns']:
        analysis['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'top_values': df[col].value_counts().head(5).to_dict(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100
        }
    
    # Enhanced numeric analysis
    if analysis['numeric_columns']:
        analysis['numeric_summary'] = df[analysis['numeric_columns']].describe().to_dict()
        analysis['correlation_matrix'] = df[analysis['numeric_columns']].corr()
        analysis['outliers'] = {}
        
        # Detect outliers using IQR method
        for col in analysis['numeric_columns']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            analysis['outliers'][col] = len(outliers)
    
    return analysis

if __name__ == "__main__":
    main()