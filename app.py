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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error, r2_score, silhouette_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
from datetime import datetime
import uuid
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import os
try:
    from sqlalchemy import create_engine, text
    import psycopg2
    import pymongo
    import redis
    DATABASE_SUPPORT = True
except ImportError:
    DATABASE_SUPPORT = False
    st.warning("⚠️ Database connectors not available. Install psycopg2-binary, pymongo, redis, sqlalchemy for database support.")
warnings.filterwarnings('ignore')

# Configuration - For Streamlit Cloud (frontend-only mode)
API_BASE_URL = None  # Set to None for standalone mode without backend
STANDALONE_MODE = True  # Enable standalone mode for Streamlit Cloud

# Page config
st.set_page_config(
    page_title="GenAI Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache expensive functions
@st.cache_data
def load_css(theme="light"):
    """Enhanced CSS with dark/light theme support"""
    if theme == "dark":
        return """
<style>
/* Dark Theme */
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}
.main-header {
    font-size: 2.8rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 700;
}
.chat-message {
    padding: 1.2rem;
    margin: 0.7rem 0;
    border-radius: 12px;
    border-left: 4px solid #667eea;
    background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.viz-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    border: 1px solid #333;
    margin: 1.5rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 0.7rem;
    box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    transform: translateY(0px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
}
.error-message {
    background: linear-gradient(135deg, #2d1b69 0%, #11998e 100%);
    color: #ff6b6b;
    border-left: 4px solid #ff4757;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.success-message {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: #2d3436;
    border-left: 4px solid #00b894;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
.stSelectbox > div > div {
    background-color: #2d2d44;
    color: #fafafa;
    border: 1px solid #667eea;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.6rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
}
</style>
"""
    else:  # Light theme
        return """
<style>
/* Light Theme */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
}
.main-header {
    font-size: 2.8rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.chat-message {
    padding: 1.2rem;
    margin: 0.7rem 0;
    border-radius: 12px;
    border-left: 4px solid #667eea;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.viz-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
    margin: 1.5rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 0.7rem;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    transform: translateY(0px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
}
.error-message {
    background: rgba(255, 230, 230, 0.9);
    color: #d63031;
    border-left: 4px solid #ff4757;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}
.success-message {
    background: rgba(230, 255, 230, 0.9);
    color: #00b894;
    border-left: 4px solid #00b894;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.6rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.1);
    padding: 0.5rem;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.8);
    color: #2d3436;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
}
</style>
"""

def display_enhanced_chart(fig, title="Visualization", description="", chart_type="unknown"):
    """Display chart with enhanced styling and container"""
    st.markdown(f'<div class="viz-container">', unsafe_allow_html=True)
    
    if title:
        st.markdown(f"### 📊 {title}")
    
    if description:
        st.markdown(f"*{description}*")
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Track figure for export
    figure_data = {
        'figure': fig,
        'title': title,
        'description': description,
        'type': chart_type,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add to current figures (avoid duplicates)
    if 'current_figures' not in st.session_state:
        st.session_state.current_figures = []
    
    # Replace if same title exists, otherwise append
    existing_idx = next((i for i, f in enumerate(st.session_state.current_figures) if f['title'] == title), None)
    if existing_idx is not None:
        st.session_state.current_figures[existing_idx] = figure_data
    else:
        st.session_state.current_figures.append(figure_data)
    
    # Add some analytics info if available
    if hasattr(fig, 'data') and fig.data:
        data_points = sum(len(trace.x) if hasattr(trace, 'x') and trace.x is not None else 0 for trace in fig.data)
        if data_points > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Data Points", f"{data_points:,}")
            with col2:
                st.metric("📋 Chart Type", fig.data[0].type.title() if fig.data else "Unknown")
            with col3:
                st.metric("🎨 Theme", st.session_state.theme.title())
    
    # Quick export options for this chart
    col1, col2 = st.columns(2)
    with col1:
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            st.download_button(
                label="📥 Download PNG",
                data=img_bytes,
                file_name=f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                key=f"download_{title}_{len(st.session_state.current_figures)}"
            )
        except:
            st.info("💡 Install `kaleido` for PNG downloads")
    
    with col2:
        if st.button("🔗 Share Chart", key=f"share_{title}_{len(st.session_state.current_figures)}"):
            try:
                share_info = ExportManager.generate_share_link({'chart': title})
                st.success(f"🔗 Share link: `{share_info['id']}`")
            except Exception as e:
                st.error(f"Share failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

class ExportManager:
    """Advanced export and sharing functionality"""
    
    @staticmethod
    def create_pdf_report(figures_data, dataset_info, analysis_summary):
        """Create a comprehensive PDF report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#667eea')
        )
        
        # Title
        story.append(Paragraph("Omni-Data Analytics Report", title_style))
        story.append(Spacer(1, 20))
        
        # Report metadata
        metadata_style = ParagraphStyle('Metadata', parent=styles['Normal'], fontSize=10)
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", metadata_style))
        story.append(Paragraph(f"Report ID: {str(uuid.uuid4())[:8]}", metadata_style))
        story.append(Spacer(1, 30))
        
        # Dataset Summary
        story.append(Paragraph("Dataset Overview", styles['Heading2']))
        if dataset_info:
            data = [
                ['Metric', 'Value'],
                ['Total Rows', f"{dataset_info.get('rows', 'N/A'):,}"],
                ['Total Columns', f"{dataset_info.get('columns', 'N/A'):,}"],
                ['Numeric Columns', f"{len(dataset_info.get('numeric_columns', [])):,}"],
                ['Categorical Columns', f"{len(dataset_info.get('categorical_columns', [])):,}"]
            ]
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
        
        # Analysis Summary
        if analysis_summary:
            story.append(Paragraph("Key Insights", styles['Heading2']))
            for insight in analysis_summary:
                story.append(Paragraph(f"• {insight}", styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Charts section
        story.append(Paragraph("Visualizations", styles['Heading2']))
        
        for i, fig_data in enumerate(figures_data):
            try:
                # Convert plotly figure to image
                img_bytes = fig_data['figure'].to_image(format="png", width=600, height=400)
                
                # Create temporary file for image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(img_bytes)
                    tmp_file_path = tmp_file.name
                
                # Add image to PDF
                img = RLImage(tmp_file_path, width=5*inch, height=3.33*inch)
                story.append(img)
                story.append(Spacer(1, 10))
                
                # Add caption
                caption = fig_data.get('title', f'Visualization {i+1}')
                story.append(Paragraph(f"Figure {i+1}: {caption}", styles['Caption']))
                story.append(Spacer(1, 20))
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                story.append(Paragraph(f"[Chart {i+1} could not be rendered: {str(e)}]", styles['Normal']))
                story.append(Spacer(1, 20))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("Generated by Omni-Data Analytics Platform", metadata_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def create_excel_export(df, figures_data, analysis_summary):
        """Create comprehensive Excel export with multiple sheets"""
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns', 'Missing Values'],
                'Value': [
                    len(df),
                    len(df.columns),
                    len(df.select_dtypes(include=[np.number]).columns),
                    len(df.select_dtypes(include=['object']).columns),
                    df.isnull().sum().sum()
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Dataset Summary', index=False)
            
            # Statistical summary for numeric columns
            if not df.select_dtypes(include=[np.number]).empty:
                numeric_summary = df.describe()
                numeric_summary.to_excel(writer, sheet_name='Statistical Summary')
            
            # Analysis insights
            if analysis_summary:
                insights_df = pd.DataFrame({'Insights': analysis_summary})
                insights_df.to_excel(writer, sheet_name='Key Insights', index=False)
            
            # Chart metadata
            if figures_data:
                chart_info = []
                for i, fig_data in enumerate(figures_data):
                    chart_info.append({
                        'Chart Number': i + 1,
                        'Title': fig_data.get('title', f'Chart {i+1}'),
                        'Type': fig_data.get('type', 'Unknown'),
                        'Created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                charts_df = pd.DataFrame(chart_info)
                charts_df.to_excel(writer, sheet_name='Charts Info', index=False)
        
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def generate_share_link(visualization_config):
        """Generate shareable link configuration"""
        link_id = str(uuid.uuid4())[:8]
        
        # In a real application, this would be stored in a database
        share_config = {
            'id': link_id,
            'config': visualization_config,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + pd.Timedelta(days=30)).isoformat()
        }
        
        # For demo purposes, return a mock URL
        share_url = f"https://omni-data.streamlit.app/shared/{link_id}"
        
        return {
            'url': share_url,
            'id': link_id,
            'expires': share_config['expires_at'][:10]  # Date only
        }

def display_export_options(current_figures, dataset_info=None, insights=None):
    """Display comprehensive export options"""
    st.markdown("### 📊 Export & Share Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Export PDF Report", help="Generate comprehensive PDF report"):
            if current_figures:
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_buffer = ExportManager.create_pdf_report(
                            current_figures, dataset_info, insights or []
                        )
                        
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"omni_data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("✅ PDF report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ PDF generation failed: {str(e)}")
            else:
                st.warning("No visualizations to export!")
    
    with col2:
        if st.button("📊 Export Excel", help="Export data and analysis to Excel"):
            if st.session_state.df is not None:
                with st.spinner("Generating Excel export..."):
                    try:
                        excel_buffer = ExportManager.create_excel_export(
                            st.session_state.df, current_figures, insights or []
                        )
                        
                        st.download_button(
                            label="📥 Download Excel File",
                            data=excel_buffer,
                            file_name=f"omni_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        st.success("✅ Excel export generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Excel generation failed: {str(e)}")
            else:
                st.warning("No data to export!")
    
    with col3:
        if st.button("🖼️ Export Charts", help="Export individual charts as PNG"):
            if current_figures:
                with st.spinner("Exporting charts..."):
                    try:
                        for i, fig_data in enumerate(current_figures):
                            img_bytes = fig_data['figure'].to_image(format="png", width=1200, height=800)
                            
                            st.download_button(
                                label=f"📥 Chart {i+1}: {fig_data.get('title', 'Visualization')}",
                                data=img_bytes,
                                file_name=f"chart_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                key=f"chart_download_{i}"
                            )
                        
                        st.success(f"✅ {len(current_figures)} chart(s) ready for download!")
                        
                    except Exception as e:
                        st.error(f"❌ Chart export failed: {str(e)}")
            else:
                st.warning("No charts to export!")
    
    with col4:
        if st.button("🔗 Generate Share Link", help="Create shareable dashboard link"):
            if current_figures:
                try:
                    share_info = ExportManager.generate_share_link({
                        'figures': len(current_figures),
                        'dataset_rows': len(st.session_state.df) if st.session_state.df is not None else 0
                    })
                    
                    st.success("✅ Share link generated!")
                    st.code(share_info['url'], language='text')
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.info(f"🔗 Link ID: `{share_info['id']}`")
                    with col_b:
                        st.info(f"📅 Expires: {share_info['expires']}")
                        
                except Exception as e:
                    st.error(f"❌ Share link generation failed: {str(e)}")
            else:
                st.warning("No visualizations to share!")

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
if 'current_visualization' not in st.session_state:
    st.session_state.current_visualization = None
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'current_figures' not in st.session_state:
    st.session_state.current_figures = []
if 'dataset_insights' not in st.session_state:
    st.session_state.dataset_insights = []
if 'refinement_options' not in st.session_state:
    st.session_state.refinement_options = {
        'trendline': False,
        'log_scale_x': False,
        'log_scale_y': False,
        'color_by': None,
        'size_by': None,
        'filter_by': None,
        'theme': 'plotly',
        'show_grid': True
    }

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

def parse_refinement_command(command: str, columns: List[str]) -> Dict[str, Any]:
    """Parse natural language refinement commands"""
    command_lower = command.lower()
    refinements = {}
    
    # Trendline detection
    if any(word in command_lower for word in ['trendline', 'trend line', 'regression', 'fit line']):
        refinements['trendline'] = True
    
    # Log scale detection
    if 'log scale' in command_lower or 'logarithmic' in command_lower:
        if 'x' in command_lower or 'x-axis' in command_lower:
            refinements['log_scale_x'] = True
        elif 'y' in command_lower or 'y-axis' in command_lower:
            refinements['log_scale_y'] = True
        else:
            refinements['log_scale_y'] = True  # Default to y-axis
    
    # Color by detection
    color_patterns = [
        r'color by (\w+)',
        r'colour by (\w+)', 
        r'group by color (\w+)',
        r'categorize by (\w+)',
        r'split by (\w+)'
    ]
    
    for pattern in color_patterns:
        match = re.search(pattern, command_lower)
        if match:
            potential_column = match.group(1)
            # Find best matching column
            matched_cols = find_best_column_match(potential_column, columns, threshold=0.4)
            if matched_cols:
                refinements['color_by'] = matched_cols[0]
            break
    
    # Size by detection
    size_patterns = [
        r'size by (\w+)',
        r'scale by (\w+)',
        r'bubble size (\w+)'
    ]
    
    for pattern in size_patterns:
        match = re.search(pattern, command_lower)
        if match:
            potential_column = match.group(1)
            matched_cols = find_best_column_match(potential_column, columns, threshold=0.4)
            if matched_cols:
                refinements['size_by'] = matched_cols[0]
            break
    
    # Theme detection
    if any(word in command_lower for word in ['dark theme', 'dark mode']):
        refinements['theme'] = 'plotly_dark'
    elif any(word in command_lower for word in ['white theme', 'light mode']):
        refinements['theme'] = 'plotly_white'
    
    # Grid detection
    if any(word in command_lower for word in ['remove grid', 'hide grid', 'no grid']):
        refinements['show_grid'] = False
    elif any(word in command_lower for word in ['show grid', 'add grid', 'with grid']):
        refinements['show_grid'] = True
    
    return refinements

def apply_visualization_refinements(fig, refinements: Dict[str, Any], df: pd.DataFrame, 
                                  chart_type: str, x_col: str = None, y_col: str = None):
    """Apply refinements to a plotly figure"""
    try:
        # Apply log scales
        if refinements.get('log_scale_x', False) and x_col:
            fig.update_xaxes(type="log", title=f"{x_col} (log scale)")
        
        if refinements.get('log_scale_y', False) and y_col:
            fig.update_yaxes(type="log", title=f"{y_col} (log scale)")
        
        # Apply trendline (for scatter plots)
        if refinements.get('trendline', False) and chart_type == 'scatter' and x_col and y_col:
            # Add trendline using numpy polyfit
            x_data = df[x_col].dropna()
            y_data = df[y_col].dropna()
            
            if len(x_data) > 1 and len(y_data) > 1:
                # Ensure same length
                min_len = min(len(x_data), len(y_data))
                x_data = x_data.iloc[:min_len]
                y_data = y_data.iloc[:min_len]
                
                # Calculate trendline
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                
                # Add trendline to figure
                fig.add_scatter(x=x_data, y=p(x_data), 
                              mode='lines', name='Trendline',
                              line=dict(color='red', dash='dash'))
        
        # Apply theme
        if refinements.get('theme'):
            fig.update_layout(template=refinements['theme'])
        
        # Apply grid settings
        if not refinements.get('show_grid', True):
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
        
        return fig
        
    except Exception as e:
        st.error(f"Error applying refinements: {str(e)}")
        return fig

def render_refinement_controls(chart_type: str, columns: List[str]):
    """Render interactive refinement controls"""
    if st.session_state.current_visualization is None:
        return
    
    # Show current active refinements
    active_refinements = [k.replace('_', ' ').title() for k, v in st.session_state.refinement_options.items() if v and v != 'plotly']
    if active_refinements:
        st.info(f"🎨 **Active refinements:** {', '.join(active_refinements)}")
    
    st.markdown("### 🎨 Refine Your Visualization")
    
    # Create tabs for different types of refinements
    tab1, tab2, tab3 = st.tabs(["📊 Chart Options", "🎭 Style & Theme", "💬 Natural Language"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Chart Enhancements**")
            
            # Trendline (for scatter plots)
            if chart_type in ['scatter']:
                if st.button("➕ Add Trendline", key="add_trendline"):
                    st.session_state.refinement_options['trendline'] = True
                    st.rerun()
                
                if st.button("➖ Remove Trendline", key="remove_trendline"):
                    st.session_state.refinement_options['trendline'] = False
                    st.rerun()
            
            # Log scale options
            st.markdown("**Scale Options**")
            if st.button("📈 Y-axis Log Scale", key="log_y"):
                st.session_state.refinement_options['log_scale_y'] = not st.session_state.refinement_options['log_scale_y']
                st.rerun()
                
            if st.button("📊 X-axis Log Scale", key="log_x"):
                st.session_state.refinement_options['log_scale_x'] = not st.session_state.refinement_options['log_scale_x']
                st.rerun()
        
        with col2:
            st.markdown("**Grouping & Colors**")
            
            # Color by options
            categorical_cols = [col for col in columns if st.session_state.df[col].dtype == 'object']
            
            color_by = st.selectbox(
                "Color by:",
                [None] + categorical_cols,
                index=0 if st.session_state.refinement_options['color_by'] is None 
                else categorical_cols.index(st.session_state.refinement_options['color_by']) + 1 
                if st.session_state.refinement_options['color_by'] in categorical_cols else 0,
                key="color_by_select"
            )
            
            if st.button("🎨 Apply Color Grouping", key="apply_color"):
                st.session_state.refinement_options['color_by'] = color_by
                st.rerun()
            
            # Size by options (for scatter plots)
            if chart_type == 'scatter':
                numeric_cols = [col for col in columns if st.session_state.df[col].dtype in ['int64', 'float64']]
                
                size_by = st.selectbox(
                    "Size by:",
                    [None] + numeric_cols,
                    key="size_by_select"
                )
                
                if st.button("📏 Apply Size Mapping", key="apply_size"):
                    st.session_state.refinement_options['size_by'] = size_by
                    st.rerun()
    
    with tab2:
        st.markdown("**Theme & Style**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            theme_options = ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn']
            
            current_theme_idx = 0
            if st.session_state.refinement_options['theme'] in theme_options:
                current_theme_idx = theme_options.index(st.session_state.refinement_options['theme'])
            
            theme = st.selectbox(
                "Theme:",
                theme_options,
                index=current_theme_idx,
                key="theme_select"
            )
            
            if st.button("🎭 Apply Theme", key="apply_theme"):
                st.session_state.refinement_options['theme'] = theme
                st.rerun()
        
        with col2:
            # Grid toggle
            if st.button("🔲 Toggle Grid", key="toggle_grid"):
                st.session_state.refinement_options['show_grid'] = not st.session_state.refinement_options['show_grid']
                st.rerun()
            
            # Reset all refinements
            if st.button("🔄 Reset All Refinements", key="reset_refinements"):
                st.session_state.refinement_options = {
                    'trendline': False,
                    'log_scale_x': False,
                    'log_scale_y': False,
                    'color_by': None,
                    'size_by': None,
                    'filter_by': None,
                    'theme': 'plotly',
                    'show_grid': True
                }
                st.rerun()
            
            # Refresh visualization with current refinements
            if st.button("🔄 Update Visualization", key="update_viz", type="primary"):
                # Recreate visualization with current refinements
                if st.session_state.current_visualization:
                    result = create_standalone_visualization(st.session_state.current_visualization)
                    if "visualization" in result and result["visualization"]["type"] == "plotly":
                        fig = go.Figure(result["visualization"]["figure"])
                        st.plotly_chart(fig, use_container_width=True)
                        st.success("✅ Visualization updated with refinements!")
                    else:
                        st.error("Could not update visualization")
    
    with tab3:
        st.markdown("**Natural Language Refinements**")
        st.write("Tell me how you'd like to enhance your visualization:")
        
        with st.form("refinement_form"):
            refinement_query = st.text_area(
                "Refinement request:",
                placeholder="e.g., 'Add a trendline and color by region' or 'Switch y-axis to log scale'",
                height=60
            )
            
            if st.form_submit_button("🪄 Apply Refinements"):
                if refinement_query.strip():
                    # Parse the refinement command
                    refinements = parse_refinement_command(refinement_query, columns)
                    
                    # Apply refinements to session state
                    for key, value in refinements.items():
                        st.session_state.refinement_options[key] = value
                    
                    st.success(f"Applied refinements: {', '.join(refinements.keys())}")
                    st.rerun()

def create_standalone_visualization(data: dict) -> dict:
    """Create visualizations in standalone mode without backend API"""
    try:
        chart_type = data.get('chart_type', 'bar')
        columns = data.get('columns', [])
        df_data = data.get('data', [])
        
        if not df_data or not columns:
            return {"error": "No data or columns provided"}
        
        df = pd.DataFrame(df_data)
        
        # Store current visualization context
        st.session_state.current_visualization = {
            'chart_type': chart_type,
            'columns': columns,
            'data': df_data
        }
        
        # Get refinement options
        refinements = st.session_state.refinement_options
        
        # Create base visualization
        fig = None
        x_col = columns[0] if len(columns) > 0 else None
        y_col = columns[1] if len(columns) > 1 else None
        
        # Create visualization based on chart type with refinements
        if chart_type == 'scatter' and len(columns) >= 2:
            color_col = refinements.get('color_by')
            size_col = refinements.get('size_by')
            
            fig = px.scatter(df, x=x_col, y=y_col,
                           color=color_col if color_col and color_col in df.columns else None,
                           size=size_col if size_col and size_col in df.columns else None,
                           title=f"Scatter Plot: {x_col} vs {y_col}")
            
        elif chart_type == 'line' and len(columns) >= 2:
            color_col = refinements.get('color_by')
            fig = px.line(df, x=x_col, y=y_col,
                         color=color_col if color_col and color_col in df.columns else None,
                         title=f"Line Chart: {y_col} over {x_col}")
                         
        elif chart_type == 'bar' and len(columns) >= 1:
            if len(columns) >= 2:
                color_col = refinements.get('color_by')
                fig = px.bar(df, x=x_col, y=y_col,
                           color=color_col if color_col and color_col in df.columns else None,
                           title=f"Bar Chart: {y_col} by {x_col}")
            else:
                value_counts = df[x_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Bar Chart: {x_col} Distribution")
                           
        elif chart_type == 'grouped_bar' and len(columns) >= 2:
            group_by = data.get('group_by')
            if group_by and group_by in df.columns:
                fig = px.bar(df, x=x_col, y=y_col, color=group_by,
                           title=f"Grouped Bar Chart: {y_col} by {x_col}, grouped by {group_by}")
            else:
                # Fallback to regular bar chart
                fig = px.bar(df, x=x_col, y=y_col,
                           title=f"Bar Chart: {y_col} by {x_col}")
                           
        elif chart_type == 'timeseries' and len(columns) >= 2:
            # Try to detect date columns
            date_cols = [col for col in df.columns if df[col].dtype == 'object']
            for col in date_cols:
                try:
                    df[col] = pd.to_datetime(df[col])
                    break
                except:
                    continue
            
            color_col = refinements.get('color_by')
            fig = px.line(df, x=x_col, y=y_col,
                         color=color_col if color_col and color_col in df.columns else None,
                         title=f"Time Series: {y_col} over {x_col}")
                           
        elif chart_type == 'pie' and len(columns) >= 1:
            value_counts = df[x_col].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index,
                        title=f"Pie Chart: {x_col} Distribution")
                        
        elif chart_type == 'histogram' and len(columns) >= 1:
            color_col = refinements.get('color_by')
            fig = px.histogram(df, x=x_col,
                             color=color_col if color_col and color_col in df.columns else None,
                             title=f"Histogram: {x_col} Distribution")
                             
        elif chart_type == 'heatmap':
            # Create correlation heatmap for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix,
                              labels=dict(x="Variables", y="Variables", color="Correlation"),
                              x=corr_matrix.columns,
                              y=corr_matrix.columns,
                              color_continuous_scale="RdBu_r",
                              aspect="auto",
                              title="Correlation Heatmap")
                
                # Add correlation values as text annotations
                fig.update_traces(text=np.around(corr_matrix.values, decimals=2),
                                texttemplate="%{text}", textfont_size=10)
            else:
                return {"error": "Heatmap requires at least 2 numeric columns for correlation analysis"}
                
        elif chart_type == 'box' and len(columns) >= 1:
            if len(columns) >= 2:
                # Box plot with grouping
                color_col = refinements.get('color_by')
                fig = px.box(df, x=x_col, y=y_col,
                           color=color_col if color_col and color_col in df.columns else None,
                           title=f"Box Plot: {y_col} by {x_col}")
            else:
                # Single variable box plot
                fig = px.box(df, y=x_col, title=f"Box Plot: {x_col} Distribution")
                             
        elif chart_type == 'summary_stats':
            # Return summary statistics as a table
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary = df[numeric_cols].describe()
                return {
                    "response": "Here's a statistical summary of your numeric data:",
                    "visualization": {
                        "type": "table",
                        "data": summary.reset_index().to_dict('records')
                    },
                    "insights": [
                        f"Dataset contains {len(df)} rows and {len(df.columns)} columns",
                        f"Numeric columns analyzed: {', '.join(numeric_cols)}",
                        f"Data types: {dict(df.dtypes)}"
                    ]
                }
        else:
            # Default to basic bar chart
            if len(columns) >= 1:
                value_counts = df[x_col].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Top 10 {x_col} Values")
            else:
                return {"error": "No suitable columns for visualization"}
        
        # Apply refinements to the figure
        if fig is not None:
            fig = apply_visualization_refinements(fig, refinements, df, chart_type, x_col, y_col)
            
            return {
                "response": f"Created {chart_type} visualization with refinements applied.",
                "visualization": {
                    "type": "plotly",
                    "figure": fig.to_dict()
                },
                "insights": [
                    f"Visualization type: {chart_type}",
                    f"Columns used: {', '.join(columns)}",
                    f"Data points: {len(df)} rows",
                    f"Active refinements: {[k for k, v in refinements.items() if v]}"
                ]
            }
        else:
            return {"error": "Could not create visualization"}
            
    except Exception as e:
        return {"error": f"Error creating visualization: {str(e)}"}

def make_api_request(endpoint: str, data: dict) -> dict:
    """Make API request with error handling - works in standalone mode for Streamlit Cloud"""
    if STANDALONE_MODE or API_BASE_URL is None:
        # Standalone mode - create basic visualizations without backend
        return create_standalone_visualization(data)
    
    try:
        response = requests.post(f"{API_BASE_URL}/{endpoint}", json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return {"error": str(e)}

def upload_data():
    """Handle file upload with enhanced error handling"""
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
            st.success("✅ File uploaded successfully!")
            
            # Display basic info in a compact way
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
    st.subheader("🤖 AI Data Visualization Assistant")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first to start creating visualizations.")
        return
    
    # Use form to handle query submission properly
    with st.form("chat_form", clear_on_submit=False):
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
                                
                                # Add refinement controls after showing the visualization
                                if st.session_state.current_visualization:
                                    with st.expander("🎨 Refine This Visualization", expanded=False):
                                        columns = list(st.session_state.df.columns)
                                        chart_type = st.session_state.current_visualization['chart_type']
                                        render_refinement_controls(chart_type, columns)
                                        
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
        
        # Clear the query after successful processing
        st.session_state.current_query = ""

def direct_visualization():
    """Direct visualization interface with enhanced options"""
    st.subheader("🎯 Direct Visualization")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
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
                            
                            # Add refinement controls after showing the visualization
                            if st.session_state.current_visualization:
                                with st.expander("🎨 Refine This Visualization", expanded=False):
                                    render_refinement_controls(chart_type, columns)
                                    
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

def ai_agent_interface():
    """AI Agent interface for complex analysis - Standalone mode"""
    st.subheader("🤖 Data Analysis")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    if STANDALONE_MODE:
        st.info("🔧 **Standalone Mode**: Basic analysis available. Deploy with backend for advanced AI features.")
    
    analysis_type = st.selectbox(
        "Analysis Type",
        [
            "Full Data Overview",
            "Statistical Analysis", 
            "Correlation Analysis",
            "Basic Insights"
        ]
    )
    
    if st.button("🔍 Start Analysis", type="primary"):
        df = st.session_state.df
        
        with st.spinner("Analyzing your data..."):
            if analysis_type == "Full Data Overview":
                st.markdown("### 📊 Dataset Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", f"{len(df.columns):,}")
                with col3:
                    st.metric("Numeric Cols", f"{len(df.select_dtypes(include=[np.number]).columns):,}")
                with col4:
                    st.metric("Text Cols", f"{len(df.select_dtypes(include=['object']).columns):,}")
                
                # Data types
                st.markdown("### 📋 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Null %': ((df.isnull().sum() / len(df)) * 100).round(2)
                })
                st.dataframe(col_info, use_container_width=True)
                
            elif analysis_type == "Statistical Analysis":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.markdown("### 📈 Statistical Summary")
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    
                    # Create histogram for first numeric column
                    if len(numeric_cols) > 0:
                        fig = px.histogram(df, x=numeric_cols[0], 
                                         title=f"Distribution of {numeric_cols[0]}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns found for statistical analysis.")
                    
            elif analysis_type == "Correlation Analysis":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    st.markdown("### � Correlation Matrix")
                    corr_matrix = df[numeric_cols].corr()
                    
                    fig = px.imshow(corr_matrix, 
                                  title="Correlation Heatmap",
                                  color_continuous_scale="RdBu_r",
                                  aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show highest correlations
                    st.markdown("### 🔍 Strongest Correlations")
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_pairs.append({
                                'Column 1': corr_matrix.columns[i],
                                'Column 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                    
                    corr_df = pd.DataFrame(corr_pairs)
                    corr_df = corr_df.reindex(corr_df.Correlation.abs().sort_values(ascending=False).index)
                    st.dataframe(corr_df.head(10), use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for correlation analysis.")
                    
            elif analysis_type == "Basic Insights":
                st.markdown("### 💡 Basic Data Insights")
                
                insights = []
                insights.append(f"📊 Your dataset has {len(df):,} rows and {len(df.columns)} columns")
                
                # Missing data
                missing_cols = df.columns[df.isnull().any()].tolist()
                if missing_cols:
                    insights.append(f"⚠️ {len(missing_cols)} columns have missing values: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}")
                else:
                    insights.append("✅ No missing values detected")
                
                # Numeric insights
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    insights.append(f"🔢 {len(numeric_cols)} numeric columns available for analysis")
                    
                # Categorical insights  
                cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                if cat_cols:
                    insights.append(f"📝 {len(cat_cols)} text/categorical columns found")
                
                for insight in insights:
                    st.write(f"• {insight}")

def ml_interface():
    """ML interface for automated machine learning with data analysis"""
    st.subheader("🧠 Machine Learning Assistant")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    df = st.session_state.df.copy()
    
    # Auto-detect column types
    st.markdown("### 🔍 Data Analysis")
    
    # Column type detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Try to detect datetime columns
    datetime_cols = []
    for col in categorical_cols.copy():
        try:
            pd.to_datetime(df[col].head(10))
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except:
            continue
    
    # Display column analysis
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔢 Numeric Columns", len(numeric_cols))
        if numeric_cols:
            with st.expander("View Numeric Columns"):
                for col in numeric_cols:
                    nulls = df[col].isnull().sum()
                    st.write(f"**{col}**: {nulls} nulls" if nulls > 0 else f"**{col}**: complete")
    
    with col2:
        st.metric("📝 Categorical Columns", len(categorical_cols))
        if categorical_cols:
            with st.expander("View Categorical Columns"):
                for col in categorical_cols:
                    unique_vals = df[col].nunique()
                    st.write(f"**{col}**: {unique_vals} unique values")
    
    with col3:
        st.metric("📅 DateTime Columns", len(datetime_cols))
        if datetime_cols:
            with st.expander("View DateTime Columns"):
                for col in datetime_cols:
                    st.write(f"**{col}**: detected as datetime")
    
    # ML Task Selection
    st.markdown("### 🎯 ML Task Selection")
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found. ML tasks require at least one numeric column.")
        return
    
    # Suggest ML tasks based on data characteristics
    suggested_tasks = []
    
    # Check for potential target columns for supervised learning
    potential_targets = []
    for col in numeric_cols:
        unique_vals = df[col].nunique()
        if 2 <= unique_vals <= 10:  # Likely classification target
            potential_targets.append((col, "Classification"))
        elif unique_vals > 10:  # Likely regression target
            potential_targets.append((col, "Regression"))
    
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        if 2 <= unique_vals <= 20:  # Good for classification
            potential_targets.append((col, "Classification"))
    
    if potential_targets:
        suggested_tasks.append("Supervised Learning (Prediction)")
    
    if len(numeric_cols) >= 2:
        suggested_tasks.append("Clustering (Unsupervised)")
    
    # Task selection
    ml_task = st.selectbox(
        "Choose ML Task:",
        ["Data Exploration"] + suggested_tasks,
        help="Select the type of machine learning task to perform"
    )
    
    if ml_task == "Data Exploration":
        # Basic data exploration with ML insights
        st.markdown("### 📊 Exploratory Data Analysis")
        
        # Correlation analysis for numeric columns
        if len(numeric_cols) >= 2:
            st.markdown("#### 🔗 Correlation Analysis")
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix,
                          labels=dict(x="Variables", y="Variables", color="Correlation"),
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          color_continuous_scale="RdBu_r",
                          title="Feature Correlation Heatmap")
            fig.update_traces(text=np.around(corr_matrix.values, decimals=2),
                            texttemplate="%{text}", textfont_size=8)
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight strong correlations
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if strong_corrs:
                st.markdown("**🔥 Strong Correlations (|r| > 0.7):**")
                for col1, col2, corr in strong_corrs:
                    st.write(f"• {col1} ↔ {col2}: {corr:.3f}")
        
        # Distribution analysis
        if numeric_cols:
            st.markdown("#### 📈 Data Distributions")
            selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                # Histogram
                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.markdown(f"**Statistical Summary for {selected_col}:**")
            summary = df[selected_col].describe()
            st.dataframe(summary.to_frame().T, use_container_width=True)
    
    elif ml_task == "Supervised Learning (Prediction)":
        st.markdown("### 🎯 Supervised Learning")
        
        # Target selection
        target_col = st.selectbox(
            "Select Target Column (what to predict):",
            potential_targets,
            format_func=lambda x: f"{x[0]} ({x[1]})",
            help="Choose the column you want to predict"
        )
        
        if target_col:
            target_name, task_type = target_col
            
            # Feature selection
            available_features = [col for col in numeric_cols + categorical_cols if col != target_name]
            selected_features = st.multiselect(
                "Select Feature Columns:",
                available_features,
                default=available_features[:5],
                help="Choose the columns to use as input features"
            )
            
            if selected_features and st.button("🚀 Train Model", type="primary"):
                with st.spinner(f"Training {task_type} model..."):
                    # Prepare data
                    X = df[selected_features].copy()
                    y = df[target_name].copy()
                    
                    # Handle missing values
                    X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
                    
                    # Encode categorical variables
                    le_dict = {}
                    for col in X.select_dtypes(include=['object']).columns:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        le_dict[col] = le
                    
                    # Encode target if needed
                    target_le = None
                    if y.dtype == 'object':
                        target_le = LabelEncoder()
                        y = target_le.fit_transform(y)
                    
                    try:
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Train model
                        if task_type == "Classification":
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                        else:
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                        # Display results
                        st.success("✅ Model trained successfully!")
                        
                        # Model performance
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 Model Performance")
                            if task_type == "Classification":
                                accuracy = (predictions == y_test).mean()
                                st.metric("Accuracy", f"{accuracy:.3f}")
                                
                                # Confusion matrix
                                cm = confusion_matrix(y_test, predictions)
                                fig = px.imshow(cm, title="Confusion Matrix", 
                                              color_continuous_scale="Blues",
                                              aspect="auto")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                r2 = r2_score(y_test, predictions)
                                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                                st.metric("R² Score", f"{r2:.3f}")
                                st.metric("RMSE", f"{rmse:.3f}")
                                
                                # Actual vs Predicted
                                fig = px.scatter(x=y_test, y=predictions, 
                                               title="Actual vs Predicted",
                                               labels={'x': 'Actual', 'y': 'Predicted'})
                                fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
                                            x1=y_test.max(), y1=y_test.max(),
                                            line=dict(dash='dash', color='red'))
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### 🎯 Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': selected_features,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(importance_df, x='Importance', y='Feature',
                                       orientation='h', title="Feature Importance")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        # Model insights
                        st.markdown("#### 💡 Model Insights")
                        insights = []
                        
                        if task_type == "Classification":
                            insights.append(f"🎯 Model achieved {accuracy:.1%} accuracy on test data")
                        else:
                            insights.append(f"🎯 Model explains {r2:.1%} of variance in {target_name}")
                        
                        top_features = importance_df.head(3)['Feature'].tolist()
                        insights.append(f"🔝 Most important features: {', '.join(top_features)}")
                        
                        for insight in insights:
                            st.write(f"• {insight}")
                            
                    except Exception as e:
                        st.error(f"❌ Error training model: {str(e)}")
    
    elif ml_task == "Clustering (Unsupervised)":
        st.markdown("### 🔍 Clustering Analysis")
        
        # Feature selection for clustering
        selected_features = st.multiselect(
            "Select Features for Clustering:",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
            help="Choose numeric columns for clustering"
        )
        
        n_clusters = st.slider("Number of Clusters:", 2, 10, 3)
        
        if selected_features and st.button("🔍 Perform Clustering", type="primary"):
            with st.spinner("Performing clustering analysis..."):
                try:
                    # Prepare data
                    X = df[selected_features].fillna(df[selected_features].mean())
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Perform clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # Calculate silhouette score
                    sil_score = silhouette_score(X_scaled, clusters)
                    
                    # Add clusters to dataframe
                    df_clustered = df.copy()
                    df_clustered['Cluster'] = clusters
                    
                    # Display results
                    st.success("✅ Clustering completed!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Silhouette Score", f"{sil_score:.3f}")
                        st.write("*Higher is better (max: 1.0)*")
                        
                        # Cluster sizes
                        cluster_counts = pd.Series(clusters).value_counts().sort_index()
                        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                                   title="Cluster Sizes",
                                   labels={'x': 'Cluster', 'y': 'Number of Points'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Visualization based on number of features
                        if len(selected_features) >= 2:
                            fig = px.scatter(df_clustered, x=selected_features[0], y=selected_features[1],
                                           color='Cluster', title="Cluster Visualization")
                            st.plotly_chart(fig, use_container_width=True)
                        
                    # Cluster analysis
                    st.markdown("#### 📋 Cluster Analysis")
                    cluster_summary = df_clustered.groupby('Cluster')[selected_features].mean()
                    st.dataframe(cluster_summary, use_container_width=True)
                    
                    # Insights
                    st.markdown("#### 💡 Clustering Insights")
                    insights = []
                    insights.append(f"🎯 Found {n_clusters} clusters with silhouette score of {sil_score:.3f}")
                    
                    # Find the largest cluster
                    largest_cluster = cluster_counts.idxmax()
                    largest_size = cluster_counts.max()
                    insights.append(f"📊 Cluster {largest_cluster} is the largest with {largest_size} points")
                    
                    for insight in insights:
                        st.write(f"• {insight}")
                        
                except Exception as e:
                    st.error(f"❌ Error performing clustering: {str(e)}")

def main():
    """Main application"""
    
    # Apply CSS theme
    st.markdown(load_css(st.session_state.theme), unsafe_allow_html=True)
    
    # Header
    if STANDALONE_MODE:
        st.markdown('<h1 class="main-header">🎯 Omni-Data: Smart Data Visualization Platform</h1>', unsafe_allow_html=True)
        st.markdown("Transform your data into insights with AI-powered visualizations")
    else:
        st.markdown('<h1 class="main-header">🎯 Omni-Data: GenAI Data Visualization Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("Transform your data into insights with AI-powered visualizations")
    
    # Sidebar
    with st.sidebar:
        # Theme Toggle
        st.markdown("### 🎨 Theme Settings")
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button("☀️ Light", key="light_theme"):
                st.session_state.theme = "light"
                st.rerun()
        with theme_col2:
            if st.button("🌙 Dark", key="dark_theme"):
                st.session_state.theme = "dark"
                st.rerun()
        
        current_theme = "Light Mode" if st.session_state.theme == "light" else "Dark Mode"
        st.info(f"Current: {current_theme}")
        st.markdown("---")
        
        # File upload section
        df = upload_data()
        
        if df is not None:            
            # Quick templates
            render_query_templates()
            
            # Query history
            render_query_history()
            
            # Export Panel
            if st.session_state.current_figures:
                st.markdown("---")
                st.markdown("### 📊 Export Dashboard")
                
                # Quick stats
                num_charts = len(st.session_state.current_figures)
                st.metric("📈 Active Charts", num_charts)
                
                # Compact export options
                if st.button("📄 Export All", help="Export comprehensive PDF report"):
                    dataset_info = {
                        'rows': len(st.session_state.df),
                        'columns': len(st.session_state.df.columns),
                        'numeric_columns': st.session_state.df.select_dtypes(include=[np.number]).columns.tolist(),
                        'categorical_columns': st.session_state.df.select_dtypes(include=['object']).columns.tolist()
                    }
                    
                    display_export_options(
                        st.session_state.current_figures, 
                        dataset_info, 
                        st.session_state.dataset_insights
                    )
                
                # Clear charts button
                if st.button("🗑️ Clear Charts", help="Clear all current visualizations"):
                    st.session_state.current_figures = []
                    st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("**🚀 Omni-Data Dashboard**")
        st.markdown("*Enhanced Smart Parser with AI*")
    
    # Main content tabs
    if st.session_state.df is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Smart Chat", "🎯 Direct Viz", "🤖 AI Agent", "🧠 ML"])
        
        with tab1:
            chat_interface()
        
        with tab2:
            direct_visualization()
        
        with tab3:
            ai_agent_interface()
            
        with tab4:
            ml_interface()
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