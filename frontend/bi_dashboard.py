import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import io
import base64
from zipfile import ZipFile
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================

st.set_page_config(
    page_title="GenAI BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for BI-style layout
st.markdown("""
<style>
/* Main styling */
.main-header {
    font-size: 2.2rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 600;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.viz-card {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s;
}

.viz-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}

.viz-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 0.5rem;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
}

.viz-description {
    color: #7f8c8d;
    font-size: 0.9rem;
    margin-bottom: 1rem;
    font-style: italic;
}

.settings-panel {
    background-color: #f8f9fa;
    border-left: 4px solid #3498db;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}

.data-preview {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 1rem;
    border-left: 4px solid #28a745;
}

.summary-stats {
    background-color: #fff3cd;
    border-radius: 8px;
    padding: 1rem;
    border-left: 4px solid #ffc107;
}

.upload-area {
    border: 2px dashed #3498db;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background-color: #f8f9fa;
    margin: 1rem 0;
}

/* Button styling */
.stButton > button {
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: all 0.3s;
}

.add-viz-btn {
    background-color: #28a745;
    color: white;
}

.remove-viz-btn {
    background-color: #dc3545;
    color: white;
}

.export-btn {
    background-color: #17a2b8;
    color: white;
}

/* Grid layout */
.viz-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

/* Responsive design */
@media (max-width: 768px) {
    .viz-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA PROCESSING & ANALYSIS FUNCTIONS
# ==========================================

@st.cache_data
def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive dataset analysis"""
    analysis = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
    }
    
    # Statistical summaries
    if analysis['numeric_columns']:
        analysis['numeric_summary'] = df[analysis['numeric_columns']].describe()
        analysis['correlation_matrix'] = df[analysis['numeric_columns']].corr()
    
    if analysis['categorical_columns']:
        analysis['categorical_summary'] = {}
        for col in analysis['categorical_columns']:
            analysis['categorical_summary'][col] = {
                'unique_count': df[col].nunique(),
                'top_values': df[col].value_counts().head(10).to_dict(),
                'mode': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
            }
    
    return analysis

def get_visualization_recommendations(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate visualization recommendations based on dataset analysis - WORKING VISUALIZATIONS ONLY"""
    recommendations = []
    
    numeric_cols = analysis['numeric_columns']
    categorical_cols = analysis['categorical_columns']
    
    # ONLY INCLUDE WORKING VISUALIZATIONS: bar_chart, box_plot, pie_chart
    
    # Bar charts for categorical columns (Priority 1 - always works)
    for col in categorical_cols[:4]:  # Show more since we have fewer types
        if analysis['categorical_summary'][col]['unique_count'] <= 20:  # Avoid too many categories
            recommendations.append({
                'type': 'bar_chart',
                'title': f'Distribution of {col}',
                'description': f'Shows frequency of different {col} categories',
                'column': col,
                'priority': 1
            })
    
    # Box plots for numeric vs categorical (Priority 2)
    if numeric_cols and categorical_cols:
        for num_col in numeric_cols[:3]:  # Show more numeric comparisons
            for cat_col in categorical_cols[:2]:  # Top 2 categorical
                if analysis['categorical_summary'][cat_col]['unique_count'] <= 10:
                    recommendations.append({
                        'type': 'box_plot',
                        'title': f'{num_col} by {cat_col}',
                        'description': f'Distribution of {num_col} across {cat_col} categories',
                        'numeric_column': num_col,
                        'categorical_column': cat_col,
                        'priority': 2
                    })
    
    # Single numeric box plots (Priority 3)
    for col in numeric_cols[:3]:
        recommendations.append({
            'type': 'box_plot',
            'title': f'Distribution of {col}',
            'description': f'Shows quartiles and outliers for {col}',
            'numeric_column': col,
            'categorical_column': None,
            'priority': 3
        })
    
    # Pie charts for categorical columns (Priority 4 - fewer categories work better)
    for col in categorical_cols[:3]:
        if 3 <= analysis['categorical_summary'][col]['unique_count'] <= 8:  # Good for pie charts
            recommendations.append({
                'type': 'pie_chart',
                'title': f'Composition of {col}',
                'description': f'Shows proportional breakdown of {col}',
                'column': col,
                'priority': 4
            })
    
    return sorted(recommendations, key=lambda x: x['priority'])

def get_all_visualizations_for_export(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate ALL visualization types for export functionality"""
    all_visualizations = []
    
    numeric_cols = analysis['numeric_columns']
    categorical_cols = analysis['categorical_columns']
    
    # Correlation heatmap for numeric data
    if len(numeric_cols) >= 2:
        all_visualizations.append({
            'type': 'correlation_heatmap',
            'title': 'Feature Correlation Matrix',
            'description': 'Shows relationships between numeric variables',
            'columns': numeric_cols,
        })
    
    # Histograms for numeric columns
    for col in numeric_cols:
        all_visualizations.append({
            'type': 'histogram',
            'title': f'Distribution of {col}',
            'description': f'Shows the frequency distribution of {col}',
            'column': col,
        })
    
    # Scatter plots for numeric pairs
    if len(numeric_cols) >= 2:
        corr_matrix = analysis['correlation_matrix']
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:i+3]:  # Limit pairs to avoid too many
                all_visualizations.append({
                    'type': 'scatter_plot',
                    'title': f'{col1} vs {col2}',
                    'description': f'Scatter plot showing relationship',
                    'x_column': col1,
                    'y_column': col2,
                })
    
    # Bar charts for categorical columns
    for col in categorical_cols:
        if analysis['categorical_summary'][col]['unique_count'] <= 50:
            all_visualizations.append({
                'type': 'bar_chart',
                'title': f'Distribution of {col}',
                'description': f'Shows frequency of different {col} categories',
                'column': col,
            })
    
    # Box plots for numeric vs categorical
    if numeric_cols and categorical_cols:
        for num_col in numeric_cols:
            for cat_col in categorical_cols[:2]:  # Limit categorical
                if analysis['categorical_summary'][cat_col]['unique_count'] <= 20:
                    all_visualizations.append({
                        'type': 'box_plot',
                        'title': f'{num_col} by {cat_col}',
                        'description': f'Distribution of {num_col} across {cat_col} categories',
                        'numeric_column': num_col,
                        'categorical_column': cat_col,
                    })
    
    # Single numeric box plots
    for col in numeric_cols:
        all_visualizations.append({
            'type': 'box_plot',
            'title': f'Distribution of {col}',
            'description': f'Shows quartiles and outliers for {col}',
            'numeric_column': col,
            'categorical_column': None,
        })
    
    # Pie charts for categorical columns
    for col in categorical_cols:
        if analysis['categorical_summary'][col]['unique_count'] <= 10:
            all_visualizations.append({
                'type': 'pie_chart',
                'title': f'Composition of {col}',
                'description': f'Shows proportional breakdown of {col}',
                'column': col,
            })
    
    return all_visualizations

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

class VisualizationEngine:
    """Unified visualization engine for all chart types"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analysis = analyze_dataset(df)
        
    def create_correlation_heatmap(self, columns: Optional[List[str]] = None, **kwargs) -> go.Figure:
        """Create correlation heatmap"""
        if columns is None:
            columns = self.analysis['numeric_columns']
        
        if len(columns) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation heatmap")
        
        # Clean data - use only numeric columns and remove NaN
        numeric_df = self.df[columns].select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 valid numeric columns for correlation heatmap")
        
        if len(numeric_df) == 0:
            raise ValueError("No valid data for correlation heatmap")
        
        corr_matrix = numeric_df.corr()
        
        # Use go.Heatmap instead of px.imshow for better reliability
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            height=500,
            template="plotly_white",
            xaxis_title="",
            yaxis_title="",
            title_x=0.5
        )
        
        return fig
        
        return fig
    
    def create_histogram(self, column: str, bins: int = 30, color: str = "#3498db", **kwargs) -> go.Figure:
        """Create histogram"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        
        # Clean data - remove NaN values
        clean_data = self.df[column].dropna()
        
        if len(clean_data) == 0:
            raise ValueError(f"No valid data for column '{column}'")
        
        # Use explicit x parameter instead of data_frame
        fig = px.histogram(
            x=clean_data,  # Pass series directly
            nbins=bins,
            title=f'Distribution of {column}',
            color_discrete_sequence=[color],
            labels={'x': column, 'y': 'Frequency'}
        )
        
        # Add statistics
        mean_val = clean_data.mean()
        median_val = clean_data.median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                     annotation_text=f"Median: {median_val:.2f}")
        
        fig.update_layout(
            height=400,
            template="plotly_white",
            showlegend=False,
            xaxis_title=column,
            yaxis_title="Frequency"
        )
        
        return fig
    
    def create_bar_chart(self, column: str, limit: int = 20, sort_by: str = "count", **kwargs) -> go.Figure:
        """Create bar chart for categorical data"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        
        # Clean data
        clean_data = self.df[column].dropna()
        if len(clean_data) == 0:
            raise ValueError("No valid data for bar chart")
        
        value_counts = clean_data.value_counts().head(limit)
        
        if len(value_counts) == 0:
            raise ValueError("No data to display in bar chart")
        
        if sort_by == "alphabetical":
            value_counts = value_counts.sort_index()
        
        fig = px.bar(
            x=value_counts.index.astype(str),  # Ensure x-axis labels are strings
            y=value_counts.values,
            title=f'Distribution of {column}',
            labels={'x': column, 'y': 'Count'},
            template="plotly_white"
        )
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            xaxis_title=column,
            yaxis_title="Count"
        )
        
        return fig
    
    def create_scatter_plot(self, x_column: str, y_column: str, color_column: Optional[str] = None, 
                           size_column: Optional[str] = None, **kwargs) -> go.Figure:
        """Create scatter plot"""
        if x_column not in self.df.columns or y_column not in self.df.columns:
            raise ValueError("Specified columns not found in dataset")
        
        # Clean data - remove rows with NaN in required columns
        required_cols = [x_column, y_column]
        if color_column and color_column in self.df.columns:
            required_cols.append(color_column)
        if size_column and size_column in self.df.columns:
            required_cols.append(size_column)
            
        clean_df = self.df[required_cols].dropna()
        
        if len(clean_df) == 0:
            raise ValueError("No valid data for scatter plot")
        
        # Use explicit x, y parameters instead of data_frame
        scatter_kwargs = {
            'x': clean_df[x_column],
            'y': clean_df[y_column],
            'title': f'{y_column} vs {x_column}',
            'template': "plotly_white",
            'labels': {'x': x_column, 'y': y_column}
        }
        
        if color_column and color_column in clean_df.columns:
            scatter_kwargs['color'] = clean_df[color_column]
        if size_column and size_column in clean_df.columns:
            scatter_kwargs['size'] = clean_df[size_column]
            
        fig = px.scatter(**scatter_kwargs)
        
        # Add correlation line for numeric data
        if (clean_df[x_column].dtype in ['int64', 'float64'] and 
            clean_df[y_column].dtype in ['int64', 'float64']):
            
            correlation = clean_df[x_column].corr(clean_df[y_column])
            
            # Add trendline
            try:
                z = np.polyfit(clean_df[x_column], clean_df[y_column], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(clean_df[x_column].min(), clean_df[x_column].max(), 100)
                
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name=f'Trend (r={correlation:.2f})',
                    line=dict(color='red', dash='dash')
                ))
            except:
                pass  # Skip trendline if calculation fails
        
        fig.update_layout(
            height=400,
            xaxis_title=x_column,
            yaxis_title=y_column
        )
        
        return fig
    
    def create_box_plot(self, numeric_column: str, categorical_column: Optional[str] = None, **kwargs) -> go.Figure:
        """Create box plot"""
        if numeric_column not in self.df.columns:
            raise ValueError(f"Column '{numeric_column}' not found in dataset")
        
        # Clean data
        if categorical_column and categorical_column in self.df.columns:
            clean_df = self.df[[numeric_column, categorical_column]].dropna()
            if len(clean_df) == 0:
                raise ValueError("No valid data for box plot")
                
            fig = px.box(
                clean_df,
                x=categorical_column,
                y=numeric_column,
                title=f'{numeric_column} by {categorical_column}',
                template="plotly_white"
            )
            fig.update_layout(xaxis_title=categorical_column, yaxis_title=numeric_column)
        else:
            clean_data = self.df[numeric_column].dropna()
            if len(clean_data) == 0:
                raise ValueError("No valid data for box plot")
                
            fig = px.box(
                y=clean_data,
                title=f'Distribution of {numeric_column}',
                template="plotly_white"
            )
            fig.update_layout(yaxis_title=numeric_column)
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_pie_chart(self, column: str, limit: int = 10, **kwargs) -> go.Figure:
        """Create pie chart"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        
        # Clean data and get value counts
        clean_data = self.df[column].dropna()
        if len(clean_data) == 0:
            raise ValueError("No valid data for pie chart")
            
        value_counts = clean_data.value_counts().head(limit)
        
        if len(value_counts) == 0:
            raise ValueError("No data to display in pie chart")
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f'Distribution of {column}',
            template="plotly_white"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400, showlegend=True)
        
        return fig

# ==========================================
# EXPORT FUNCTIONALITY
# ==========================================

def export_figure_as_png(fig: go.Figure, filename: str) -> bytes:
    """Export plotly figure as PNG bytes"""
    img_bytes = fig.to_image(format="png", width=1200, height=800, engine="kaleido")
    return img_bytes

def create_charts_zip(figures: Dict[str, go.Figure]) -> bytes:
    """Create ZIP file containing all charts as PNG files"""
    zip_buffer = io.BytesIO()
    
    with ZipFile(zip_buffer, 'w') as zip_file:
        for title, fig in figures.items():
            png_bytes = export_figure_as_png(fig, f"{title}.png")
            zip_file.writestr(f"{title.replace(' ', '_')}.png", png_bytes)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# ==========================================
# UI COMPONENTS
# ==========================================

def display_dataset_preview(df: pd.DataFrame, analysis: Dict[str, Any]):
    """Display dataset preview and summary statistics"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="data-preview">', unsafe_allow_html=True)
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="summary-stats">', unsafe_allow_html=True)
        st.markdown("### 📊 Summary Statistics")
        
        # Key metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Rows", f"{analysis['shape'][0]:,}")
            st.metric("Numeric Cols", len(analysis['numeric_columns']))
        with col_b:
            st.metric("Columns", analysis['shape'][1])
            st.metric("Categorical Cols", len(analysis['categorical_columns']))
        
        # Missing values
        missing_total = sum(analysis['missing_values'].values())
        if missing_total > 0:
            st.warning(f"⚠️ {missing_total} missing values found")
            with st.expander("Missing Values Detail"):
                for col, missing in analysis['missing_values'].items():
                    if missing > 0:
                        pct = (missing / analysis['shape'][0]) * 100
                        st.write(f"**{col}**: {missing} ({pct:.1f}%)")
        else:
            st.success("✅ No missing values")
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_visualization_card(viz_id: str, viz_config: Dict[str, Any], viz_engine: VisualizationEngine):
    """Create a visualization card with settings panel"""
    
    with st.container():
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        
        # Header with title and controls
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f'<div class="viz-header">{viz_config["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="viz-description">{viz_config["description"]}</div>', unsafe_allow_html=True)
        
        with col2:
            show_settings = st.button("⚙️ Settings", key=f"settings_{viz_id}")
        
        with col3:
            if st.button("🗑️ Remove", key=f"remove_{viz_id}"):
                if viz_id in st.session_state.visualizations:
                    del st.session_state.visualizations[viz_id]
                st.experimental_rerun()
        
        # Settings panel
        if show_settings:
            st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
            
            # Chart type selection - WORKING TYPES ONLY
            chart_types = {
                'bar_chart': 'Bar Chart',
                'box_plot': 'Box Plot', 
                'pie_chart': 'Pie Chart'
            }
            
            new_chart_type = st.selectbox(
                "Chart Type (Working Only)",
                options=list(chart_types.keys()),
                format_func=lambda x: chart_types[x],
                index=list(chart_types.keys()).index(viz_config.get('type', 'bar_chart')),
                key=f"chart_type_{viz_id}"
            )
            
            # Dynamic settings based on chart type (WORKING TYPES ONLY)
            if new_chart_type == 'bar_chart':
                column = st.selectbox(
                    "Column",
                    options=viz_engine.analysis['categorical_columns'] + viz_engine.analysis['numeric_columns'],
                    key=f"bar_column_{viz_id}"
                )
                limit = st.slider("Max Categories", 5, 50, viz_config.get('limit', 20), key=f"bar_limit_{viz_id}")
                sort_by = st.radio("Sort by", ["count", "alphabetical"], key=f"bar_sort_{viz_id}")
                
                viz_config.update({'type': new_chart_type, 'column': column, 'limit': limit, 'sort_by': sort_by})
                
            elif new_chart_type == 'box_plot':
                numeric_col = st.selectbox("Numeric Column", viz_engine.analysis['numeric_columns'], key=f"box_numeric_{viz_id}")
                categorical_col = st.selectbox("Group by (optional)", [None] + viz_engine.analysis['categorical_columns'], key=f"box_categorical_{viz_id}")
                
                viz_config.update({
                    'type': new_chart_type,
                    'numeric_column': numeric_col,
                    'categorical_column': categorical_col
                })
                
            elif new_chart_type == 'pie_chart':
                column = st.selectbox("Column", viz_engine.analysis['categorical_columns'], key=f"pie_column_{viz_id}")
                limit = st.slider("Max Slices", 3, 20, viz_config.get('limit', 10), key=f"pie_limit_{viz_id}")
                
                viz_config.update({'type': new_chart_type, 'column': column, 'limit': limit})
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate and display visualization
        try:
            # Add direct test before generating figure
            st.write("🔍 **Debug: Testing basic chart creation...**")
            
            # Test 1: Simple bar chart with test data
            if viz_config['type'] == 'bar_chart':
                test_data = viz_engine.df[viz_config['column']].value_counts().head(10)
                st.write(f"Bar chart data preview: {test_data}")
                
                import plotly.express as px
                test_fig = px.bar(x=test_data.index.astype(str), y=test_data.values, 
                                 title=f"Test: {viz_config['title']}")
                st.plotly_chart(test_fig, use_container_width=True)
                st.success("✅ Direct bar chart test successful!")
            
            elif viz_config['type'] == 'box_plot':
                if viz_config.get('categorical_column'):
                    test_data = viz_engine.df[[viz_config['numeric_column'], viz_config['categorical_column']]].dropna()
                    st.write(f"Box plot data shape: {test_data.shape}")
                    
                    import plotly.express as px
                    test_fig = px.box(test_data, x=viz_config['categorical_column'], 
                                     y=viz_config['numeric_column'], title=f"Test: {viz_config['title']}")
                    st.plotly_chart(test_fig, use_container_width=True)
                    st.success("✅ Direct grouped box plot test successful!")
                else:
                    test_data = viz_engine.df[viz_config['numeric_column']].dropna()
                    st.write(f"Box plot data length: {len(test_data)}")
                    
                    import plotly.express as px
                    test_fig = px.box(y=test_data, title=f"Test: {viz_config['title']}")
                    st.plotly_chart(test_fig, use_container_width=True)
                    st.success("✅ Direct single box plot test successful!")
            
            elif viz_config['type'] == 'pie_chart':
                test_data = viz_engine.df[viz_config['column']].value_counts().head(viz_config.get('limit', 10))
                st.write(f"Pie chart data preview: {test_data}")
                
                import plotly.express as px
                test_fig = px.pie(values=test_data.values, names=test_data.index, 
                                 title=f"Test: {viz_config['title']}")
                st.plotly_chart(test_fig, use_container_width=True)
                st.success("✅ Direct pie chart test successful!")
            
            st.write("---")
            st.write("🔍 **Now testing with VisualizationEngine method...**")
            
            fig = generate_figure(viz_config, viz_engine)
            
            # Verify figure has data
            if not fig.data:
                st.error("⚠️ Figure generated but contains no data traces")
                return
            
            st.write(f"📊 Generated {len(fig.data)} data trace(s)")
            
            # Ensure proper figure configuration for Streamlit display
            fig.update_layout(
                autosize=True,
                margin=dict(l=50, r=50, t=60, b=50),
                height=450,
                width=None  # Let Streamlit handle width
            )
            
            # Try multiple display methods to ensure visibility
            st.write("**Rendering with VisualizationEngine method...**")
            
            # Method 1: Standard plotly_chart with explicit config
            try:
                st.plotly_chart(
                    fig, 
                    use_container_width=True,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
                        'responsive': True
                    },
                    theme="streamlit"
                )
                st.write("✅ VisualizationEngine method successful")
            except Exception as e1:
                st.warning(f"VisualizationEngine method failed: {str(e1)}")
            
            # Debug info
            st.write(f"✅ Chart type: {viz_config['type']}")
            st.write(f"✅ Data points: {len(viz_engine.df)} rows")
            
            # Export button
            png_bytes = export_figure_as_png(fig, viz_config['title'])
            st.download_button(
                "📥 Download PNG",
                data=png_bytes,
                file_name=f"{viz_config['title'].replace(' ', '_')}.png",
                mime="image/png",
                key=f"download_{viz_id}"
            )
            
        except Exception as e:
            st.error(f"❌ Error generating visualization: {str(e)}")
            st.write("Debug info:", str(viz_config))
            import traceback
            st.code(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)

def generate_figure(viz_config: Dict[str, Any], viz_engine: VisualizationEngine) -> go.Figure:
    """Generate plotly figure based on configuration"""
    
    chart_type = viz_config['type']
    
    # Add basic validation
    if not hasattr(viz_engine, 'df') or viz_engine.df is None or len(viz_engine.df) == 0:
        raise ValueError("No data available for visualization")
    
    if chart_type == 'correlation_heatmap':
        return viz_engine.create_correlation_heatmap()
    elif chart_type == 'histogram':
        return viz_engine.create_histogram(
            column=viz_config['column'],
            bins=viz_config.get('bins', 30)
        )
    elif chart_type == 'scatter_plot':
        return viz_engine.create_scatter_plot(
            x_column=viz_config['x_column'],
            y_column=viz_config['y_column'],
            color_column=viz_config.get('color_column'),
            size_column=viz_config.get('size_column')
        )
    elif chart_type == 'bar_chart':
        return viz_engine.create_bar_chart(
            column=viz_config['column'],
            limit=viz_config.get('limit', 20),
            sort_by=viz_config.get('sort_by', 'count')
        )
    elif chart_type == 'box_plot':
        return viz_engine.create_box_plot(
            numeric_column=viz_config['numeric_column'],
            categorical_column=viz_config.get('categorical_column')
        )
    elif chart_type == 'pie_chart':
        return viz_engine.create_pie_chart(
            column=viz_config['column'],
            limit=viz_config.get('limit', 10)
        )
    else:
        # Fallback: create a simple test chart
        import plotly.express as px
        fig = px.scatter(x=[1, 2, 3], y=[1, 2, 3], title=f"Test Chart ({chart_type})")
        return fig

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = {}
    if 'viz_counter' not in st.session_state:
        st.session_state.viz_counter = 0
    
    # Header
    st.markdown('<h1 class="main-header">🎯 GenAI Business Intelligence Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Dashboard Controls")
        
        # File upload
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your dataset to start creating visualizations"
        )
        
        if uploaded_file is not None:
            try:
                # Load and cache dataset
                df = pd.read_csv(uploaded_file)
                st.session_state.dataset = df
                st.session_state.analysis = analyze_dataset(df)
                
                st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
                
                # Auto-generate visualizations button
                if st.button("🚀 Generate Auto Visualizations", type="primary"):
                    recommendations = get_visualization_recommendations(st.session_state.analysis)
                    
                    # Clear existing visualizations
                    st.session_state.visualizations = {}
                    st.session_state.viz_counter = 0
                    
                    # Add recommended visualizations
                    for rec in recommendations[:6]:  # Limit to 6 auto visualizations
                        viz_id = f"auto_{st.session_state.viz_counter}"
                        st.session_state.visualizations[viz_id] = rec
                        st.session_state.viz_counter += 1
                    
                    st.experimental_rerun()
                
                # Manual visualization controls
                st.markdown("### ➕ Add Custom Visualization")
                
                if st.button("📊 Add New Chart"):
                    viz_id = f"custom_{st.session_state.viz_counter}"
                    st.session_state.visualizations[viz_id] = {
                        'type': 'histogram',
                        'title': f'Custom Chart {st.session_state.viz_counter + 1}',
                        'description': 'User-created visualization',
                        'column': st.session_state.analysis['numeric_columns'][0] if st.session_state.analysis['numeric_columns'] else st.session_state.analysis['columns'][0]
                    }
                    st.session_state.viz_counter += 1
                    st.experimental_rerun()
                
                # Export all charts
                if st.session_state.visualizations:
                    st.markdown("### 💾 Export Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📦 Download Displayed Charts (ZIP)"):
                            try:
                                viz_engine = VisualizationEngine(st.session_state.dataset)
                                figures = {}
                                
                                for viz_id, viz_config in st.session_state.visualizations.items():
                                    fig = generate_figure(viz_config, viz_engine)
                                    figures[viz_config['title']] = fig
                                
                                zip_bytes = create_charts_zip(figures)
                                
                                st.download_button(
                                    "📥 Download Displayed Charts ZIP",
                                    data=zip_bytes,
                                    file_name=f"displayed_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip"
                                )
                            except Exception as e:
                                st.error(f"Error creating displayed charts ZIP: {str(e)}")
                    
                    with col2:
                        if st.button("📦 Download ALL Visualizations (ZIP)"):
                            try:
                                viz_engine = VisualizationEngine(st.session_state.dataset)
                                figures = {}
                                
                                # Generate ALL visualization types for export
                                all_viz_configs = get_all_visualizations_for_export(st.session_state.analysis)
                                
                                for viz_config in all_viz_configs:
                                    try:
                                        fig = generate_figure(viz_config, viz_engine)
                                        figures[viz_config['title']] = fig
                                    except Exception as e:
                                        st.warning(f"Skipped {viz_config['title']}: {str(e)}")
                                        continue
                                
                                zip_bytes = create_charts_zip(figures)
                                
                                st.download_button(
                                    "📥 Download ALL Charts ZIP",
                                    data=zip_bytes,
                                    file_name=f"all_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip"
                                )
                                
                                st.info(f"✅ Generated {len(figures)} total visualizations (including histograms, scatter plots, correlation heatmaps)")
                                
                            except Exception as e:
                                st.error(f"Error creating complete charts ZIP: {str(e)}")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Main content
    if st.session_state.dataset is not None:
        # Display dataset preview and summary
        st.markdown("## 📊 Dataset Overview")
        display_dataset_preview(st.session_state.dataset, st.session_state.analysis)
        
        st.markdown("---")
        
        # Display visualizations
        if st.session_state.visualizations:
            st.markdown("## 📈 Visualizations Dashboard")
            
            viz_engine = VisualizationEngine(st.session_state.dataset)
            
            # Create grid layout
            viz_items = list(st.session_state.visualizations.items())
            
            # Display visualizations in a 2-column grid
            for i in range(0, len(viz_items), 2):
                cols = st.columns(2)
                
                # First visualization in left column
                with cols[0]:
                    viz_id, viz_config = viz_items[i]
                    create_visualization_card(viz_id, viz_config, viz_engine)
                
                # Second visualization in right column (if exists)
                if i + 1 < len(viz_items):
                    with cols[1]:
                        viz_id, viz_config = viz_items[i + 1]
                        create_visualization_card(viz_id, viz_config, viz_engine)
        
        else:
            st.info("👆 Click 'Generate Auto Visualizations' to create standard charts, or 'Add New Chart' for custom visualizations.")
    
    else:
        # Welcome screen
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        st.markdown("""
        ## 🎯 Welcome to GenAI BI Dashboard
        
        This intelligent dashboard automatically generates comprehensive visualizations for your data:
        
        ### ✨ **Auto-Generated Insights**
        - 📊 **Correlation heatmaps** for numeric relationships
        - 📈 **Distribution histograms** for data patterns
        - 📋 **Category breakdowns** with smart bar charts
        - 🔍 **Scatter plots** for variable relationships
        - 📦 **Box plots** for statistical summaries
        
        ### 🎛️ **Interactive Features**
        - **Multi-chart layout** - See all insights at once
        - **Customizable settings** - Modify any visualization
        - **Export capabilities** - Download individual charts or full reports
        - **Add/remove charts** - Build your perfect dashboard
        
        ### 🚀 **Get Started**
        Upload a CSV file in the sidebar to begin your data exploration journey!
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()