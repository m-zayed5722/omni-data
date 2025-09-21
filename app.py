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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, mean_squared_error, r2_score, silhouette_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import zscore, pearsonr
from scipy.signal import find_peaks
import warnings
from datetime import datetime
import uuid
import hashlib
import hmac
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import os
import time
import threading
import queue
from datetime import timedelta
import asyncio
try:
    from sqlalchemy import create_engine, text
    import psycopg2
    import pymongo
    import redis
    DATABASE_SUPPORT = True
except ImportError:
    DATABASE_SUPPORT = False
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

class RealTimeAnalytics:
    """Real-time data streaming and live analytics system"""
    
    @staticmethod
    def generate_live_data(data_type="random", base_df=None, anomaly_rate=0.05):
        """Generate simulated real-time data points"""
        current_time = datetime.now()
        
        if data_type == "random" or base_df is None:
            # Generate random data
            data_point = {
                'timestamp': current_time,
                'value': np.random.normal(100, 20),
                'category': np.random.choice(['A', 'B', 'C', 'D']),
                'metric_1': np.random.uniform(0, 100),
                'metric_2': np.random.exponential(2),
                'is_anomaly': np.random.random() < anomaly_rate
            }
        else:
            # Generate data similar to base dataset
            numeric_cols = base_df.select_dtypes(include=[np.number]).columns
            categorical_cols = base_df.select_dtypes(include=['object']).columns
            
            data_point = {'timestamp': current_time}
            
            # Generate numeric data based on existing patterns
            for col in numeric_cols:
                mean_val = base_df[col].mean()
                std_val = base_df[col].std()
                
                # Add some time-based trend
                trend = np.sin(current_time.minute / 10) * std_val * 0.1
                value = np.random.normal(mean_val + trend, std_val * 0.5)
                
                # Inject anomalies
                if np.random.random() < anomaly_rate:
                    value += np.random.choice([-1, 1]) * std_val * 3
                    data_point['is_anomaly'] = True
                else:
                    data_point['is_anomaly'] = False
                    
                data_point[col] = value
            
            # Generate categorical data
            for col in categorical_cols:
                unique_vals = base_df[col].unique()
                data_point[col] = np.random.choice(unique_vals)
        
        return data_point
    
    @staticmethod
    def detect_anomalies(data_stream, window_size=50, threshold=2.5):
        """Simple anomaly detection using rolling statistics"""
        if len(data_stream) < window_size:
            return []
        
        recent_data = data_stream[-window_size:]
        values = [point['value'] if 'value' in point else list(point.values())[1] for point in recent_data]
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        anomalies = []
        for i, point in enumerate(recent_data):
            value = point['value'] if 'value' in point else list(point.values())[1]
            z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0
            
            if z_score > threshold:
                anomalies.append({
                    'timestamp': point['timestamp'],
                    'value': value,
                    'z_score': z_score,
                    'severity': 'high' if z_score > 3.5 else 'medium'
                })
        
        return anomalies
    
    @staticmethod
    def create_live_chart(data_stream, chart_type='line', max_points=100):
        """Create live updating charts"""
        if not data_stream:
            return None
        
        # Limit data points for performance
        recent_data = data_stream[-max_points:]
        df_stream = pd.DataFrame(recent_data)
        
        if 'timestamp' not in df_stream.columns:
            return None
        
        if chart_type == 'line':
            fig = px.line(df_stream, x='timestamp', y='value' if 'value' in df_stream.columns else df_stream.columns[1],
                         title='Real-time Data Stream',
                         template='plotly_white')
            
            # Highlight anomalies
            if 'is_anomaly' in df_stream.columns:
                anomaly_data = df_stream[df_stream['is_anomaly'] == True]
                if not anomaly_data.empty:
                    fig.add_scatter(x=anomaly_data['timestamp'], 
                                  y=anomaly_data['value' if 'value' in df_stream.columns else df_stream.columns[1]],
                                  mode='markers', marker=dict(color='red', size=8),
                                  name='Anomalies')
            
        elif chart_type == 'gauge':
            current_value = recent_data[-1]['value'] if 'value' in recent_data[-1] else list(recent_data[-1].values())[1]
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = current_value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Current Value"},
                delta = {'reference': recent_data[-2]['value'] if len(recent_data) > 1 and 'value' in recent_data[-2] else current_value},
                gauge = {'axis': {'range': [None, 200]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}],
                        'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 150}}))
        
        elif chart_type == 'histogram':
            values = [point['value'] if 'value' in point else list(point.values())[1] for point in recent_data]
            fig = px.histogram(x=values, title='Real-time Value Distribution', template='plotly_white')
        
        # Update layout for real-time feel
        fig.update_layout(
            showlegend=True,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig

class AIInsightsEngine:
    """Advanced AI-powered insights and analysis system"""
    
    @staticmethod
    def detect_anomalies_advanced(df, method='isolation_forest', contamination=0.1):
        """Advanced anomaly detection using multiple algorithms"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return pd.DataFrame(), {}
        
        results = {}
        
        if method == 'isolation_forest':
            # Isolation Forest
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_columns].fillna(df[numeric_columns].mean()))
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(scaled_data)
            
            df_anomalies = df.copy()
            df_anomalies['anomaly_score'] = iso_forest.score_samples(scaled_data)
            df_anomalies['is_anomaly'] = outliers == -1
            
            results['method'] = 'Isolation Forest'
            results['anomaly_count'] = sum(outliers == -1)
            results['anomaly_percentage'] = (sum(outliers == -1) / len(df)) * 100
            
        elif method == 'statistical':
            # Statistical (Z-score based)
            df_anomalies = df.copy()
            z_scores = np.abs(zscore(df[numeric_columns].fillna(df[numeric_columns].mean())))
            threshold = 3
            
            df_anomalies['is_anomaly'] = (z_scores > threshold).any(axis=1)
            df_anomalies['max_z_score'] = z_scores.max(axis=1)
            
            results['method'] = 'Statistical (Z-score > 3)'
            results['anomaly_count'] = sum(df_anomalies['is_anomaly'])
            results['anomaly_percentage'] = (sum(df_anomalies['is_anomaly']) / len(df)) * 100
            
        elif method == 'dbscan':
            # DBSCAN clustering for anomaly detection
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_columns].fillna(df[numeric_columns].mean()))
            
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(scaled_data)
            
            df_anomalies = df.copy()
            df_anomalies['cluster'] = clusters
            df_anomalies['is_anomaly'] = clusters == -1
            
            results['method'] = 'DBSCAN Clustering'
            results['anomaly_count'] = sum(clusters == -1)
            results['anomaly_percentage'] = (sum(clusters == -1) / len(df)) * 100
            results['n_clusters'] = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        return df_anomalies, results
    
    @staticmethod
    def trend_analysis(df, date_column=None, value_columns=None):
        """Analyze trends in time series data"""
        insights = {}
        
        # Auto-detect date column if not provided
        if date_column is None:
            date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns
            if len(date_cols) > 0:
                date_column = date_cols[0]
            else:
                # Try to parse date-like columns
                for col in df.columns:
                    try:
                        pd.to_datetime(df[col].head())
                        date_column = col
                        df[col] = pd.to_datetime(df[col])
                        break
                    except:
                        continue
        
        if date_column is None:
            return {"error": "No date column found for trend analysis"}
        
        # Auto-detect value columns
        if value_columns is None:
            value_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in value_columns:
            if col == date_column:
                continue
                
            # Sort by date
            df_sorted = df.sort_values(date_column)
            
            # Calculate trend using linear regression
            x_numeric = np.arange(len(df_sorted))
            y_values = df_sorted[col].fillna(df_sorted[col].mean())
            
            if len(y_values) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_values)
                
                # Determine trend direction
                if p_value < 0.05:  # Significant trend
                    if slope > 0:
                        trend_direction = "Increasing"
                    else:
                        trend_direction = "Decreasing"
                else:
                    trend_direction = "No significant trend"
                
                # Calculate percentage change
                start_value = y_values.iloc[0]
                end_value = y_values.iloc[-1]
                percent_change = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0
                
                # Detect seasonality (simple approach)
                seasonality = "None detected"
                if len(y_values) >= 12:  # Need at least 12 points
                    # Check for periodic patterns
                    autocorr_12 = y_values.autocorr(lag=min(12, len(y_values)//2))
                    if autocorr_12 > 0.3:
                        seasonality = "Potential seasonal pattern detected"
                
                insights[col] = {
                    'trend_direction': trend_direction,
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'percent_change': percent_change,
                    'seasonality': seasonality,
                    'mean': y_values.mean(),
                    'std': y_values.std(),
                    'volatility': y_values.std() / y_values.mean() if y_values.mean() != 0 else 0
                }
        
        return insights
    
    @staticmethod
    def correlation_insights(df):
        """Generate correlation insights and identify strong relationships"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}
        
        corr_matrix = numeric_df.corr()
        insights = {}
        strong_correlations = []
        
        # Find strong correlations (absolute value > 0.7)
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    
                    relationship_type = "Strong positive" if corr_value > 0 else "Strong negative"
                    strong_correlations.append({
                        'variable_1': col1,
                        'variable_2': col2,
                        'correlation': corr_value,
                        'relationship': relationship_type
                    })
        
        insights['correlation_matrix'] = corr_matrix
        insights['strong_correlations'] = strong_correlations
        insights['summary'] = f"Found {len(strong_correlations)} strong correlations"
        
        return insights
    
    @staticmethod
    def predictive_forecasting(df, target_column, n_periods=10, method='linear'):
        """Generate predictive forecasts for time series data"""
        if target_column not in df.columns:
            return {"error": f"Column '{target_column}' not found"}
        
        # Prepare data
        df_clean = df.dropna(subset=[target_column])
        if len(df_clean) < 3:
            return {"error": "Insufficient data for forecasting"}
        
        y = df_clean[target_column].values
        x = np.arange(len(y))
        
        if method == 'linear':
            # Simple linear regression forecast
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)
            
            # Generate future predictions
            future_x = np.arange(len(y), len(y) + n_periods)
            predictions = model.predict(future_x.reshape(-1, 1))
            
            # Calculate confidence intervals (rough approximation)
            residuals = y - model.predict(x.reshape(-1, 1))
            std_error = np.std(residuals)
            confidence_interval = 1.96 * std_error  # 95% CI
            
            forecast_result = {
                'method': 'Linear Regression',
                'predictions': predictions.tolist(),
                'confidence_lower': (predictions - confidence_interval).tolist(),
                'confidence_upper': (predictions + confidence_interval).tolist(),
                'model_score': model.score(x.reshape(-1, 1), y),
                'trend': 'Increasing' if model.coef_[0] > 0 else 'Decreasing'
            }
            
        elif method == 'moving_average':
            # Simple moving average forecast
            window = min(5, len(y) // 2)
            if window < 1:
                window = 1
                
            recent_avg = np.mean(y[-window:])
            predictions = [recent_avg] * n_periods
            
            # Calculate historical volatility for confidence intervals
            historical_std = np.std(y)
            confidence_interval = 1.96 * historical_std
            
            forecast_result = {
                'method': f'{window}-Period Moving Average',
                'predictions': predictions,
                'confidence_lower': [p - confidence_interval for p in predictions],
                'confidence_upper': [p + confidence_interval for p in predictions],
                'trend': 'Stable (Moving Average)'
            }
        
        return forecast_result
    
    @staticmethod
    def generate_smart_recommendations(df, insights_data):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Data quality recommendations
        missing_data = df.isnull().sum()
        high_missing = missing_data[missing_data > len(df) * 0.1]
        
        if len(high_missing) > 0:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'High',
                'recommendation': f"Address missing data in columns: {', '.join(high_missing.index)}",
                'action': 'Consider data imputation or removal of sparse columns'
            })
        
        # Anomaly recommendations
        if 'anomaly_results' in insights_data:
            anomaly_count = insights_data['anomaly_results'].get('anomaly_count', 0)
            if anomaly_count > len(df) * 0.05:  # More than 5% anomalies
                recommendations.append({
                    'category': 'Data Integrity',
                    'priority': 'Medium',
                    'recommendation': f"High anomaly rate detected ({anomaly_count} outliers)",
                    'action': 'Review data collection process and investigate outliers'
                })
        
        # Correlation recommendations
        if 'correlation_insights' in insights_data:
            strong_corrs = insights_data['correlation_insights'].get('strong_correlations', [])
            if len(strong_corrs) > 0:
                recommendations.append({
                    'category': 'Analysis',
                    'priority': 'Medium',
                    'recommendation': f"Found {len(strong_corrs)} strong correlations",
                    'action': 'Consider feature selection to avoid multicollinearity in modeling'
                })
        
        # Trend recommendations
        if 'trend_insights' in insights_data:
            trends = insights_data['trend_insights']
            for col, trend_info in trends.items():
                if isinstance(trend_info, dict) and trend_info.get('r_squared', 0) > 0.8:
                    direction = trend_info['trend_direction']
                    recommendations.append({
                        'category': 'Business Insight',
                        'priority': 'High',
                        'recommendation': f"{col} shows {direction.lower()} trend (R² = {trend_info['r_squared']:.3f})",
                        'action': f"Monitor {col} closely and consider trend-based strategies"
                    })
        
        # Performance recommendations
        if len(df) > 10000:
            recommendations.append({
                'category': 'Performance',
                'priority': 'Low',
                'recommendation': f"Large dataset detected ({len(df)} rows)",
                'action': 'Consider data sampling for faster analysis or use data aggregation'
            })
        
        return recommendations

def render_ai_insights_dashboard():
    """Render the AI-powered insights dashboard"""
    st.markdown("### 🧠 AI-Powered Insights Engine")
    
    if st.session_state.df is None:
        st.info("👆 Please upload a dataset to generate AI insights!")
        return
    
    df = st.session_state.df
    
    # Initialize insights session state
    if 'ai_insights' not in st.session_state:
        st.session_state.ai_insights = {}
    
    # Insights generation controls
    st.markdown("#### 🔍 Analysis Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_types = st.multiselect(
            "Select Analysis Types",
            ["Anomaly Detection", "Trend Analysis", "Correlation Analysis", "Predictive Forecasting"],
            default=["Anomaly Detection", "Correlation Analysis"]
        )
    
    with col2:
        if "Anomaly Detection" in analysis_types:
            anomaly_method = st.selectbox(
                "Anomaly Detection Method",
                ["isolation_forest", "statistical", "dbscan"],
                help="Choose the algorithm for anomaly detection"
            )
    
    with col3:
        if "Predictive Forecasting" in analysis_types:
            forecast_periods = st.number_input(
                "Forecast Periods", 
                min_value=1, max_value=50, value=10,
                help="Number of periods to forecast"
            )
    
    # Generate insights button
    if st.button("🚀 Generate AI Insights", type="primary"):
        with st.spinner("🧠 AI is analyzing your data..."):
            insights_results = {}
            
            # Anomaly Detection
            if "Anomaly Detection" in analysis_types:
                df_anomalies, anomaly_results = AIInsightsEngine.detect_anomalies_advanced(
                    df, method=anomaly_method
                )
                insights_results['anomalies_df'] = df_anomalies
                insights_results['anomaly_results'] = anomaly_results
            
            # Trend Analysis
            if "Trend Analysis" in analysis_types:
                trend_results = AIInsightsEngine.trend_analysis(df)
                insights_results['trend_insights'] = trend_results
            
            # Correlation Analysis
            if "Correlation Analysis" in analysis_types:
                corr_results = AIInsightsEngine.correlation_insights(df)
                insights_results['correlation_insights'] = corr_results
            
            # Predictive Forecasting
            if "Predictive Forecasting" in analysis_types:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_col = numeric_cols[0]  # Use first numeric column
                    forecast_results = AIInsightsEngine.predictive_forecasting(
                        df, target_col, n_periods=forecast_periods
                    )
                    insights_results['forecast_results'] = forecast_results
                    insights_results['forecast_target'] = target_col
            
            # Generate recommendations
            recommendations = AIInsightsEngine.generate_smart_recommendations(df, insights_results)
            insights_results['recommendations'] = recommendations
            
            st.session_state.ai_insights = insights_results
        
        st.success("✅ AI Insights generated successfully!")
    
    # Display results
    if st.session_state.ai_insights:
        insights = st.session_state.ai_insights
        
        # Smart Recommendations
        if 'recommendations' in insights:
            st.markdown("#### 💡 Smart Recommendations")
            recommendations = insights['recommendations']
            
            for rec in recommendations:
                priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[rec['priority']]
                with st.expander(f"{priority_color} {rec['category']}: {rec['recommendation']}"):
                    st.write(f"**Action:** {rec['action']}")
        
        # Anomaly Detection Results
        if 'anomaly_results' in insights:
            st.markdown("#### 🚨 Anomaly Detection")
            anomaly_results = insights['anomaly_results']
            df_anomalies = insights['anomalies_df']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Method", anomaly_results['method'])
            with col2:
                st.metric("Anomalies Found", anomaly_results['anomaly_count'])
            with col3:
                st.metric("Percentage", f"{anomaly_results['anomaly_percentage']:.1f}%")
            
            # Anomaly visualization
            if sum(df_anomalies['is_anomaly']) > 0:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    fig = px.scatter(
                        df_anomalies, 
                        x=numeric_cols[0], 
                        y=numeric_cols[1],
                        color='is_anomaly',
                        title="Anomaly Detection Results",
                        color_discrete_map={True: 'red', False: 'blue'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Trend Analysis Results
        if 'trend_insights' in insights:
            trend_results = insights['trend_insights']
            if not trend_results.get('error'):
                st.markdown("#### 📈 Trend Analysis")
                
                for col, trend_info in trend_results.items():
                    if isinstance(trend_info, dict):
                        with st.expander(f"📊 {col} Trend Analysis"):
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Trend Direction", trend_info['trend_direction'])
                            with col_b:
                                st.metric("R² Score", f"{trend_info['r_squared']:.3f}")
                            with col_c:
                                st.metric("% Change", f"{trend_info['percent_change']:.1f}%")
                            
                            st.write(f"**Seasonality:** {trend_info['seasonality']}")
                            st.write(f"**Volatility:** {trend_info['volatility']:.3f}")
        
        # Correlation Analysis Results
        if 'correlation_insights' in insights:
            corr_results = insights['correlation_insights']
            if not corr_results.get('error'):
                st.markdown("#### 🔗 Correlation Analysis")
                
                # Correlation heatmap
                fig = px.imshow(
                    corr_results['correlation_matrix'],
                    title="Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Strong correlations
                if corr_results['strong_correlations']:
                    st.markdown("**Strong Correlations Found:**")
                    for corr in corr_results['strong_correlations']:
                        st.write(f"• **{corr['variable_1']}** ↔ **{corr['variable_2']}**: "
                                f"{corr['correlation']:.3f} ({corr['relationship']})")
        
        # Predictive Forecasting Results
        if 'forecast_results' in insights:
            forecast_results = insights['forecast_results']
            target_col = insights['forecast_target']
            
            if not forecast_results.get('error'):
                st.markdown("#### 🔮 Predictive Forecasting")
                
                # Display forecast metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Method", forecast_results['method'])
                with col2:
                    if 'model_score' in forecast_results:
                        st.metric("Model Score", f"{forecast_results['model_score']:.3f}")
                
                # Forecast visualization
                historical_data = df[target_col].values
                predictions = forecast_results['predictions']
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=list(range(len(historical_data))),
                    y=historical_data,
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='blue')
                ))
                
                # Predictions
                future_x = list(range(len(historical_data), len(historical_data) + len(predictions)))
                fig.add_trace(go.Scatter(
                    x=future_x,
                    y=predictions,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Confidence intervals
                if 'confidence_lower' in forecast_results:
                    fig.add_trace(go.Scatter(
                        x=future_x + future_x[::-1],
                        y=forecast_results['confidence_upper'] + forecast_results['confidence_lower'][::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval'
                    ))
                
                fig.update_layout(
                    title=f"Predictive Forecast for {target_col}",
                    xaxis_title="Time Period",
                    yaxis_title="Value"
                )
                
                st.plotly_chart(fig, use_container_width=True)

class CollaborationSystem:
    """Multi-user collaboration and workspace management system"""
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> tuple:
        """Hash password with salt for secure storage"""
        if salt is None:
            salt = str(uuid.uuid4())
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt.encode('utf-8'), 
            100000
        )
        return base64.b64encode(password_hash).decode(), salt
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """Verify password against hash"""
        password_hash, _ = CollaborationSystem.hash_password(password, salt)
        return hmac.compare_digest(password_hash, hashed)
    
    @staticmethod
    def create_user(username: str, password: str, email: str = None) -> dict:
        """Create a new user account"""
        user_id = str(uuid.uuid4())
        password_hash, salt = CollaborationSystem.hash_password(password)
        
        user = {
            'user_id': user_id,
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'salt': salt,
            'created_at': datetime.now().isoformat(),
            'role': 'user',
            'workspaces': [],
            'preferences': {
                'theme': 'dark',
                'notifications': True,
                'default_chart_type': 'auto'
            }
        }
        
        return user
    
    @staticmethod
    def create_workspace(name: str, description: str, owner_id: str) -> dict:
        """Create a new collaborative workspace"""
        workspace_id = str(uuid.uuid4())
        
        workspace = {
            'workspace_id': workspace_id,
            'name': name,
            'description': description,
            'owner_id': owner_id,
            'created_at': datetime.now().isoformat(),
            'members': [{'user_id': owner_id, 'role': 'owner', 'joined_at': datetime.now().isoformat()}],
            'datasets': [],
            'charts': [],
            'comments': [],
            'settings': {
                'visibility': 'private',
                'allow_comments': True,
                'allow_exports': True,
                'auto_save': True
            }
        }
        
        return workspace
    
    @staticmethod
    def add_chart_comment(chart_id: str, user_id: str, username: str, comment_text: str) -> dict:
        """Add comment to a chart"""
        comment = {
            'comment_id': str(uuid.uuid4()),
            'chart_id': chart_id,
            'user_id': user_id,
            'username': username,
            'text': comment_text,
            'timestamp': datetime.now().isoformat(),
            'replies': [],
            'likes': 0,
            'resolved': False
        }
        
        return comment
    
    @staticmethod
    def save_chart_to_workspace(chart_data: dict, workspace_id: str, user_id: str) -> dict:
        """Save a chart to workspace for collaboration"""
        chart_entry = {
            'chart_id': str(uuid.uuid4()),
            'workspace_id': workspace_id,
            'created_by': user_id,
            'created_at': datetime.now().isoformat(),
            'chart_type': chart_data.get('type', 'unknown'),
            'title': chart_data.get('title', 'Untitled Chart'),
            'config': chart_data,
            'version': 1,
            'is_shared': False,
            'permissions': {'view': True, 'edit': False, 'comment': True}
        }
        
        return chart_entry

class UserAuthentication:
    """Simple user authentication system for demo purposes"""
    
    @staticmethod
    def initialize_auth_state():
        """Initialize authentication session state"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'users_db' not in st.session_state:
            # Demo users database (in production, use proper database)
            st.session_state.users_db = {
                'admin': CollaborationSystem.create_user('admin', 'admin123', 'admin@example.com'),
                'demo': CollaborationSystem.create_user('demo', 'demo123', 'demo@example.com')
            }
            st.session_state.users_db['admin']['role'] = 'admin'
        if 'workspaces_db' not in st.session_state:
            st.session_state.workspaces_db = {}
        if 'charts_db' not in st.session_state:
            st.session_state.charts_db = {}
        if 'comments_db' not in st.session_state:
            st.session_state.comments_db = {}
    
    @staticmethod
    def login_form():
        """Render login form"""
        st.markdown("### 🔐 User Authentication")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col1, col2 = st.columns(2)
            with col1:
                login_submitted = st.form_submit_button("🔑 Login", type="primary")
            with col2:
                register_submitted = st.form_submit_button("📝 Register")
            
            if login_submitted:
                if username in st.session_state.users_db:
                    user = st.session_state.users_db[username]
                    if CollaborationSystem.verify_password(password, user['password_hash'], user['salt']):
                        st.session_state.authenticated = True
                        st.session_state.current_user = user
                        st.success(f"✅ Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid password!")
                else:
                    st.error("❌ User not found!")
            
            if register_submitted:
                if username and password:
                    if username not in st.session_state.users_db:
                        new_user = CollaborationSystem.create_user(username, password, f"{username}@example.com")
                        st.session_state.users_db[username] = new_user
                        st.success(f"✅ Account created for {username}! Please login.")
                    else:
                        st.error("❌ Username already exists!")
                else:
                    st.error("❌ Please fill in all fields!")
        
        st.markdown("---")
        st.markdown("**Demo Accounts:**")
        st.code("Username: admin, Password: admin123")
        st.code("Username: demo, Password: demo123")
    
    @staticmethod
    def logout():
        """Handle user logout"""
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.success("✅ Logged out successfully!")
        st.rerun()

def render_collaboration_dashboard():
    """Render the collaboration and workspace management dashboard"""
    UserAuthentication.initialize_auth_state()
    
    if not st.session_state.authenticated:
        UserAuthentication.login_form()
        return
    
    # User info sidebar
    with st.sidebar:
        st.markdown("---")
        user = st.session_state.current_user
        st.markdown(f"👤 **{user['username']}**")
        st.markdown(f"📧 {user.get('email', 'No email')}")
        st.markdown(f"🎭 Role: {user.get('role', 'user').title()}")
        
        if st.button("🚪 Logout"):
            UserAuthentication.logout()
    
    st.markdown("### 👥 Collaboration Hub")
    
    # Main collaboration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Workspaces", "💬 Comments", "📊 Shared Charts", "⚙️ Settings"])
    
    with tab1:
        render_workspaces_tab()
    
    with tab2:
        render_comments_tab()
    
    with tab3:
        render_shared_charts_tab()
    
    with tab4:
        render_user_settings_tab()

def render_workspaces_tab():
    """Render workspaces management tab"""
    st.markdown("#### 🏢 My Workspaces")
    
    user_id = st.session_state.current_user['user_id']
    user_workspaces = [w for w in st.session_state.workspaces_db.values() 
                      if any(m['user_id'] == user_id for m in w['members'])]
    
    # Create new workspace
    with st.expander("➕ Create New Workspace"):
        with st.form("new_workspace"):
            workspace_name = st.text_input("Workspace Name", placeholder="Enter workspace name")
            workspace_desc = st.text_area("Description", placeholder="Describe your workspace")
            
            if st.form_submit_button("🚀 Create Workspace", type="primary"):
                if workspace_name:
                    new_workspace = CollaborationSystem.create_workspace(
                        workspace_name, workspace_desc, user_id
                    )
                    st.session_state.workspaces_db[new_workspace['workspace_id']] = new_workspace
                    st.success(f"✅ Workspace '{workspace_name}' created!")
                    st.rerun()
                else:
                    st.error("❌ Please enter a workspace name!")
    
    # Display existing workspaces
    if user_workspaces:
        for workspace in user_workspaces:
            with st.expander(f"🏢 {workspace['name']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {workspace['description']}")
                    st.write(f"**Created:** {workspace['created_at'][:10]}")
                    st.write(f"**Members:** {len(workspace['members'])}")
                
                with col2:
                    if st.button(f"👥 Manage Members", key=f"manage_{workspace['workspace_id']}"):
                        st.info("Member management coming soon!")
                    
                    if st.button(f"📁 Open Workspace", key=f"open_{workspace['workspace_id']}"):
                        st.info("Workspace navigation coming soon!")
                
                # Show recent activity
                st.markdown("**Recent Activity:**")
                if workspace.get('charts'):
                    for chart in workspace['charts'][-3:]:  # Show last 3 charts
                        st.write(f"• 📊 {chart.get('title', 'Untitled')} - {chart.get('created_at', '')[:10]}")
                else:
                    st.write("• No recent activity")
    else:
        st.info("🏢 No workspaces found. Create your first workspace above!")

def render_comments_tab():
    """Render comments and discussion tab"""
    st.markdown("#### 💬 Chart Comments & Discussions")
    
    # Mock comment system for demo
    if 'demo_comments' not in st.session_state:
        st.session_state.demo_comments = [
            {
                'chart_title': 'Sales Trend Analysis',
                'user': 'admin',
                'comment': 'The Q3 spike looks interesting. Can we drill down into regional data?',
                'timestamp': '2024-01-15 10:30',
                'replies': 1
            },
            {
                'chart_title': 'Customer Segmentation',
                'user': 'demo',
                'comment': 'Great insights! The premium segment growth is promising.',
                'timestamp': '2024-01-14 15:45',
                'replies': 0
            }
        ]
    
    # Add new comment
    with st.expander("💬 Add New Comment"):
        with st.form("new_comment"):
            chart_selection = st.selectbox(
                "Select Chart", 
                ["Sales Trend Analysis", "Customer Segmentation", "Revenue Dashboard"],
                help="Choose which chart to comment on"
            )
            comment_text = st.text_area("Comment", placeholder="Share your insights...")
            
            if st.form_submit_button("📝 Post Comment", type="primary"):
                if comment_text:
                    new_comment = {
                        'chart_title': chart_selection,
                        'user': st.session_state.current_user['username'],
                        'comment': comment_text,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'replies': 0
                    }
                    st.session_state.demo_comments.insert(0, new_comment)
                    st.success("✅ Comment posted!")
                    st.rerun()
    
    # Display comments
    st.markdown("**Recent Comments:**")
    for comment in st.session_state.demo_comments:
        with st.container():
            st.markdown(f"**📊 {comment['chart_title']}**")
            st.markdown(f"💬 *{comment['user']}* - {comment['timestamp']}")
            st.markdown(f"> {comment['comment']}")
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                st.button("👍", key=f"like_{comment['timestamp']}")
            with col2:
                st.button("💬 Reply", key=f"reply_{comment['timestamp']}")
            
            st.markdown("---")

def render_shared_charts_tab():
    """Render shared charts tab"""
    st.markdown("#### 📊 Shared Charts")
    
    # Mock shared charts for demo
    shared_charts = [
        {
            'title': 'Q4 Sales Performance',
            'creator': 'admin',
            'shared_date': '2024-01-15',
            'views': 24,
            'chart_type': 'Line Chart'
        },
        {
            'title': 'Customer Demographics',
            'creator': 'demo',
            'shared_date': '2024-01-14',
            'views': 18,
            'chart_type': 'Bar Chart'
        },
        {
            'title': 'Revenue Forecast',
            'creator': 'admin',
            'shared_date': '2024-01-13',
            'views': 31,
            'chart_type': 'Area Chart'
        }
    ]
    
    # Share current chart
    if st.session_state.df is not None:
        with st.expander("📤 Share Current Analysis"):
            with st.form("share_chart"):
                share_title = st.text_input("Chart Title", placeholder="Enter a descriptive title")
                share_desc = st.text_area("Description", placeholder="Describe your analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    share_public = st.checkbox("Make Public", help="Allow others to discover this chart")
                with col2:
                    allow_comments = st.checkbox("Allow Comments", value=True)
                
                if st.form_submit_button("📤 Share Chart", type="primary"):
                    if share_title:
                        st.success(f"✅ Chart '{share_title}' shared successfully!")
                        shared_charts.insert(0, {
                            'title': share_title,
                            'creator': st.session_state.current_user['username'],
                            'shared_date': datetime.now().strftime('%Y-%m-%d'),
                            'views': 0,
                            'chart_type': 'Custom Analysis'
                        })
                    else:
                        st.error("❌ Please enter a chart title!")
    
    # Display shared charts
    st.markdown("**Available Shared Charts:**")
    for chart in shared_charts:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**📊 {chart['title']}**")
                st.markdown(f"👤 By {chart['creator']} • {chart['shared_date']} • {chart['chart_type']}")
            
            with col2:
                st.metric("👀 Views", chart['views'])
            
            with col3:
                if st.button("🔍 View", key=f"view_{chart['title']}"):
                    st.info("Chart viewer coming soon!")
            
            st.markdown("---")

def render_user_settings_tab():
    """Render user settings and preferences tab"""
    st.markdown("#### ⚙️ User Settings & Preferences")
    
    user = st.session_state.current_user
    
    with st.form("user_settings"):
        st.markdown("**Profile Settings:**")
        new_email = st.text_input("Email", value=user.get('email', ''))
        
        st.markdown("**Dashboard Preferences:**")
        theme_pref = st.selectbox(
            "Default Theme", 
            ["dark", "light"],
            index=0 if user.get('preferences', {}).get('theme') == 'dark' else 1
        )
        
        notifications = st.checkbox(
            "Enable Notifications", 
            value=user.get('preferences', {}).get('notifications', True)
        )
        
        default_chart = st.selectbox(
            "Default Chart Type",
            ["auto", "bar", "line", "scatter", "pie"],
            index=0
        )
        
        st.markdown("**Privacy Settings:**")
        profile_visibility = st.selectbox(
            "Profile Visibility",
            ["public", "workspace_members", "private"],
            index=2
        )
        
        if st.form_submit_button("💾 Save Settings", type="primary"):
            # Update user preferences
            user['email'] = new_email
            user['preferences'] = {
                'theme': theme_pref,
                'notifications': notifications,
                'default_chart_type': default_chart,
                'profile_visibility': profile_visibility
            }
            st.session_state.users_db[user['username']] = user
            st.success("✅ Settings saved successfully!")
    
    # Account management
    st.markdown("---")
    st.markdown("**Account Management:**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 Change Password"):
            st.info("Password change coming soon!")
    
    with col2:
        if st.button("📊 Export My Data"):
            st.info("Data export coming soon!")

class PerformanceOptimizer:
    """Performance optimization and caching system"""
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def cached_data_preprocessing(df_hash: str, df_json: str) -> dict:
        """Cache expensive data preprocessing operations"""
        df = pd.read_json(df_json)
        
        # Expensive operations that benefit from caching
        preprocessing_results = {
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'basic_stats': df.describe(include='all').to_dict(),
            'correlation_matrix': df.select_dtypes(include=[np.number]).corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 1 else {},
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        return preprocessing_results
    
    @staticmethod
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def cached_chart_generation(chart_type: str, df_json: str, **params) -> dict:
        """Cache expensive chart generation operations"""
        df = pd.read_json(df_json)
        
        # This would contain the heavy chart generation logic
        # For now, return chart configuration
        chart_config = {
            'type': chart_type,
            'data_shape': df.shape,
            'timestamp': datetime.now().isoformat(),
            'params': params
        }
        
        return chart_config
    
    @staticmethod
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def cached_ml_training(df_json: str, model_type: str, target_column: str, **params) -> dict:
        """Cache expensive ML model training"""
        df = pd.read_json(df_json)
        
        try:
            # This is a simplified version - in practice would include actual training
            results = {
                'model_type': model_type,
                'target_column': target_column,
                'feature_count': len(df.columns) - 1,
                'sample_count': len(df),
                'trained_at': datetime.now().isoformat(),
                'params': params,
                'status': 'completed'
            }
        except Exception as e:
            results = {'status': 'error', 'error': str(e)}
        
        return results
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame, memory_threshold_mb: int = 100) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        if initial_memory < memory_threshold_mb:
            return df
        
        df_optimized = df.copy()
        
        # Optimize integer columns
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)
        
        # Optimize float columns
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Optimize object columns to categorical where beneficial
        for col in df_optimized.select_dtypes(include=['object']).columns:
            unique_count = df_optimized[col].nunique()
            total_count = len(df_optimized[col])
            
            if unique_count / total_count < 0.5:  # Less than 50% unique values
                df_optimized[col] = df_optimized[col].astype('category')
        
        final_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        return df_optimized
    
    @staticmethod
    def paginate_data(df: pd.DataFrame, page_size: int = 1000, page: int = 0) -> tuple:
        """Implement data pagination for large datasets"""
        start_idx = page * page_size
        end_idx = start_idx + page_size
        
        paginated_df = df.iloc[start_idx:end_idx]
        total_pages = (len(df) - 1) // page_size + 1
        
        return paginated_df, total_pages, page
    
    @staticmethod
    def lazy_load_columns(df: pd.DataFrame, selected_columns: list = None) -> pd.DataFrame:
        """Implement lazy loading for column selection"""
        if selected_columns is None:
            # Default to first 10 columns for performance
            selected_columns = df.columns[:min(10, len(df.columns))].tolist()
        
        return df[selected_columns]
    
    @staticmethod
    def monitor_performance() -> dict:
        """Monitor application performance metrics"""
        import psutil
        
        metrics = {
            'memory_usage_mb': psutil.virtual_memory().used / 1024 / 1024,
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(),
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics

class DataManager:
    """Advanced data management with performance optimizations"""
    
    @staticmethod
    def smart_sample_data(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
        """Intelligently sample large datasets for analysis"""
        if len(df) <= max_rows:
            return df
        
        # For very large datasets, use stratified sampling if possible
        sample_size = min(max_rows, len(df))
        
        # Try to maintain data distribution
        if len(df) > 50000:
            # Use systematic sampling for very large datasets
            step = len(df) // sample_size
            indices = range(0, len(df), step)[:sample_size]
            sampled_df = df.iloc[indices]
        else:
            # Use random sampling for moderately large datasets
            sampled_df = df.sample(n=sample_size, random_state=42)
        
        return sampled_df
    
    @staticmethod
    def detect_large_dataset(df: pd.DataFrame) -> dict:
        """Detect if dataset is large and suggest optimizations"""
        size_info = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'is_large': False,
            'suggestions': []
        }
        
        # Define thresholds
        if size_info['row_count'] > 100000:
            size_info['is_large'] = True
            size_info['suggestions'].append("Consider using data sampling for faster analysis")
        
        if size_info['column_count'] > 50:
            size_info['suggestions'].append("Consider selecting relevant columns only")
        
        if size_info['memory_mb'] > 500:
            size_info['is_large'] = True
            size_info['suggestions'].append("Consider data type optimization")
        
        return size_info
    
    @staticmethod
    def auto_optimize_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Automatically optimize dataset for better performance"""
        # Step 1: Detect large dataset characteristics
        size_info = DataManager.detect_large_dataset(df)
        
        optimized_df = df.copy()
        
        # Step 2: Apply optimizations based on size
        if size_info['is_large']:
            # Memory optimization
            optimized_df = PerformanceOptimizer.optimize_dataframe(optimized_df)
            
            # Smart sampling if too large
            if len(optimized_df) > 50000:
                optimized_df = DataManager.smart_sample_data(optimized_df, 25000)
        
        return optimized_df

def render_performance_dashboard():
    """Render performance monitoring and optimization dashboard"""
    st.markdown("### ⚡ Performance & Scaling")
    
    # Performance monitoring
    try:
        perf_metrics = PerformanceOptimizer.monitor_performance()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💾 Memory Usage", f"{perf_metrics['memory_usage_mb']:.1f} MB")
        with col2:
            st.metric("🧠 Memory %", f"{perf_metrics['memory_percent']:.1f}%")
        with col3:
            st.metric("⚙️ CPU %", f"{perf_metrics['cpu_percent']:.1f}%")
    except ImportError:
        st.info("📊 Install 'psutil' for detailed performance monitoring")
    
    # Dataset optimization
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("#### 📊 Dataset Analysis & Optimization")
        
        # Current dataset info
        size_info = DataManager.detect_large_dataset(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Rows", f"{size_info['row_count']:,}")
        with col2:
            st.metric("📊 Columns", size_info['column_count'])
        with col3:
            st.metric("💾 Memory", f"{size_info['memory_mb']:.1f} MB")
        with col4:
            status = "🔴 Large" if size_info['is_large'] else "🟢 Optimal"
            st.metric("Status", status)
        
        # Optimization suggestions
        if size_info['suggestions']:
            st.markdown("#### 💡 Optimization Suggestions")
            for suggestion in size_info['suggestions']:
                st.write(f"• {suggestion}")
        
        # Optimization controls
        st.markdown("#### 🛠️ Optimization Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Auto-Optimize Dataset", type="primary"):
                with st.spinner("Optimizing dataset..."):
                    optimized_df = DataManager.auto_optimize_dataset(df)
                    st.session_state.df = optimized_df
                    
                    # Show optimization results
                    new_size_info = DataManager.detect_large_dataset(optimized_df)
                    memory_saved = size_info['memory_mb'] - new_size_info['memory_mb']
                    
                    if memory_saved > 0:
                        st.success(f"✅ Optimization complete! Saved {memory_saved:.1f} MB")
                    else:
                        st.info("ℹ️ Dataset was already optimized")
                    
                    st.rerun()
        
        with col2:
            if st.button("📋 Reset to Original"):
                st.info("Original dataset reset functionality would go here")
        
        # Data pagination demo
        st.markdown("#### 📄 Data Pagination")
        
        page_size = st.selectbox("Page Size", [100, 500, 1000, 5000], index=1)
        
        paginated_df, total_pages, current_page = PerformanceOptimizer.paginate_data(
            df, page_size=page_size, page=0
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous") and current_page > 0:
                # Previous page logic would go here
                pass
        with col2:
            st.write(f"Page 1 of {total_pages} ({len(paginated_df)} rows shown)")
        with col3:
            if st.button("➡️ Next") and current_page < total_pages - 1:
                # Next page logic would go here
                pass
        
        st.dataframe(paginated_df, use_container_width=True)
        
        # Column selection for lazy loading
        st.markdown("#### 🎯 Column Selection")
        
        all_columns = df.columns.tolist()
        default_selection = all_columns[:min(10, len(all_columns))]
        
        selected_columns = st.multiselect(
            "Select columns to analyze",
            all_columns,
            default=default_selection,
            help="Select fewer columns for better performance"
        )
        
        if selected_columns:
            lazy_df = PerformanceOptimizer.lazy_load_columns(df, selected_columns)
            
            st.write(f"**Selected Data:** {len(lazy_df)} rows × {len(lazy_df.columns)} columns")
            
            # Show memory usage comparison
            original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            lazy_memory = lazy_df.memory_usage(deep=True).sum() / 1024 / 1024
            memory_saved = original_memory - lazy_memory
            
            if memory_saved > 0:
                st.success(f"💾 Memory saved: {memory_saved:.1f} MB ({memory_saved/original_memory*100:.1f}%)")
    
    else:
        st.info("👆 Upload a dataset to see optimization tools!")
    
    # Cache management
    st.markdown("#### 🗄️ Cache Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Data Cache"):
            st.cache_data.clear()
            st.success("✅ Data cache cleared!")
    
    with col2:
        if st.button("📊 Cache Statistics"):
            st.info("Cache statistics would be displayed here")
    
    with col3:
        cache_enabled = st.checkbox("🚀 Enable Caching", value=True)
        if cache_enabled:
            st.success("✅ Caching enabled")
        else:
            st.warning("⚠️ Caching disabled")

def render_realtime_dashboard():
    """Render the real-time analytics dashboard"""
    st.markdown("### ⚡ Real-time Analytics Dashboard")
    
    # Initialize real-time session state
    if 'realtime_data' not in st.session_state:
        st.session_state.realtime_data = []
    if 'is_streaming' not in st.session_state:
        st.session_state.is_streaming = False
    if 'stream_interval' not in st.session_state:
        st.session_state.stream_interval = 2  # seconds
    if 'anomaly_alerts' not in st.session_state:
        st.session_state.anomaly_alerts = []
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start Stream" if not st.session_state.is_streaming else "⏸️ Pause Stream"):
            st.session_state.is_streaming = not st.session_state.is_streaming
    
    with col2:
        if st.button("🗑️ Clear Data"):
            st.session_state.realtime_data = []
            st.session_state.anomaly_alerts = []
            st.rerun()
    
    with col3:
        interval = st.selectbox("Update Interval", [1, 2, 5, 10], 
                               index=1, help="Seconds between updates")
        st.session_state.stream_interval = interval
    
    with col4:
        chart_type = st.selectbox("Chart Type", ["line", "gauge", "histogram"])
    
    # Data generation controls
    with st.expander("🎛️ Data Source Settings", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            data_source = st.selectbox("Data Source", 
                                     ["Random Data", "Based on Current Dataset"],
                                     help="Generate random data or simulate from current dataset")
        with col_b:
            anomaly_rate = st.slider("Anomaly Rate", 0.0, 0.2, 0.05, 0.01,
                                   help="Probability of generating anomalies")
    
    # Auto-refresh mechanism
    if st.session_state.is_streaming:
        # Generate new data point
        if data_source == "Based on Current Dataset" and st.session_state.df is not None:
            new_point = RealTimeAnalytics.generate_live_data(
                "dataset", st.session_state.df, anomaly_rate
            )
        else:
            new_point = RealTimeAnalytics.generate_live_data("random", None, anomaly_rate)
        
        st.session_state.realtime_data.append(new_point)
        
        # Limit data points for performance
        if len(st.session_state.realtime_data) > 500:
            st.session_state.realtime_data = st.session_state.realtime_data[-500:]
        
        # Detect anomalies
        anomalies = RealTimeAnalytics.detect_anomalies(st.session_state.realtime_data)
        if anomalies:
            for anomaly in anomalies:
                if anomaly not in st.session_state.anomaly_alerts:
                    st.session_state.anomaly_alerts.append(anomaly)
        
        # Auto-refresh
        time.sleep(st.session_state.stream_interval)
        st.rerun()
    
    # Display metrics
    if st.session_state.realtime_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Data Points", len(st.session_state.realtime_data))
        
        with col2:
            recent_val = st.session_state.realtime_data[-1]['value'] if 'value' in st.session_state.realtime_data[-1] else 0
            prev_val = st.session_state.realtime_data[-2]['value'] if len(st.session_state.realtime_data) > 1 and 'value' in st.session_state.realtime_data[-2] else recent_val
            st.metric("📈 Current Value", f"{recent_val:.2f}", f"{recent_val - prev_val:.2f}")
        
        with col3:
            st.metric("🚨 Anomalies", len(st.session_state.anomaly_alerts))
        
        with col4:
            status = "🟢 Streaming" if st.session_state.is_streaming else "🔴 Stopped"
            st.metric("Status", status)
        
        # Display chart
        fig = RealTimeAnalytics.create_live_chart(st.session_state.realtime_data, chart_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly alerts
        if st.session_state.anomaly_alerts:
            st.markdown("### 🚨 Anomaly Alerts")
            recent_anomalies = sorted(st.session_state.anomaly_alerts, 
                                    key=lambda x: x['timestamp'], reverse=True)[:5]
            
            for anomaly in recent_anomalies:
                severity_color = "🔴" if anomaly['severity'] == 'high' else "🟡"
                st.warning(f"{severity_color} Anomaly detected at {anomaly['timestamp'].strftime('%H:%M:%S')} - "
                          f"Value: {anomaly['value']:.2f} (Z-score: {anomaly['z_score']:.2f})")
        
        # Data table
        if st.checkbox("Show Raw Data"):
            recent_data = st.session_state.realtime_data[-20:]  # Show last 20 points
            df_display = pd.DataFrame(recent_data)
            if 'timestamp' in df_display.columns:
                df_display['timestamp'] = df_display['timestamp'].dt.strftime('%H:%M:%S')
            st.dataframe(df_display, use_container_width=True)
    else:
        st.info("🎯 Click 'Start Stream' to begin real-time data generation")

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "💬 Smart Chat", "🎯 Direct Viz", "🤖 AI Agent", "🧠 ML", 
        "⚡ Real-time", "🔮 AI Insights", "👥 Collaborate", "🚀 Performance"
    ])
    
    with tab1:
        if st.session_state.df is not None:
            chat_interface()
        else:
            st.info("👆 Please upload a CSV file to use Smart Chat mode!")
    
    with tab2:
        if st.session_state.df is not None:
            direct_visualization()
        else:
            st.info("👆 Please upload a CSV file to create visualizations!")
    
    with tab3:
        if st.session_state.df is not None:
            ai_agent_interface()
        else:
            st.info("👆 Please upload a CSV file to use AI Agent!")
        
    with tab4:
        if st.session_state.df is not None:
            ml_interface()
        else:
            st.info("👆 Please upload a CSV file to use ML features!")
        
    with tab5:
        # Real-time analytics works with or without uploaded data
        render_realtime_dashboard()
    
    with tab6:
        # AI Insights requires uploaded data
        render_ai_insights_dashboard()
        
    with tab7:
        # Collaboration works with or without uploaded data
        render_collaboration_dashboard()
        
    with tab8:
        # Performance monitoring and optimization
        render_performance_dashboard()

    # Show sample data section if no data is uploaded
    if st.session_state.df is None:
        st.markdown("---")
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