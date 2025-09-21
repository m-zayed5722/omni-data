#!/usr/bin/env python3

"""
BI Dashboard with Matplotlib/Seaborn instead of Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
import zipfile

# Configure matplotlib for better display
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="BI Dashboard (Matplotlib)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze dataset and return summary statistics"""
    analysis = {
        'shape': df.shape,
        'columns': list(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Basic statistics for numeric columns
    if analysis['numeric_columns']:
        analysis['numeric_summary'] = df[analysis['numeric_columns']].describe().to_dict()
        analysis['correlation_matrix'] = df[analysis['numeric_columns']].corr()
    
    # Summary for categorical columns
    analysis['categorical_summary'] = {}
    for col in analysis['categorical_columns']:
        analysis['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    
    return analysis

def create_matplotlib_figure(chart_type: str, data: pd.DataFrame, config: dict) -> tuple:
    """Create matplotlib figure and return figure + base64 image"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        if chart_type == 'bar_chart':
            column = config['column']
            limit = config.get('limit', 20)
            
            value_counts = data[column].value_counts().head(limit)
            if config.get('sort_by') == 'alphabetical':
                value_counts = value_counts.sort_index()
            
            bars = ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
            ax.set_title(config['title'])
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{int(height)}', ha='center', va='bottom')
            
        elif chart_type == 'box_plot':
            numeric_col = config['numeric_column']
            categorical_col = config.get('categorical_column')
            
            if categorical_col and categorical_col in data.columns:
                # Grouped box plot
                clean_data = data[[numeric_col, categorical_col]].dropna()
                sns.boxplot(data=clean_data, x=categorical_col, y=numeric_col, ax=ax)
                ax.set_xlabel(categorical_col)
                ax.set_ylabel(numeric_col)
                plt.xticks(rotation=45, ha='right')
            else:
                # Single box plot
                clean_data = data[numeric_col].dropna()
                ax.boxplot(clean_data)
                ax.set_ylabel(numeric_col)
                ax.set_xlabel('Distribution')
            
            ax.set_title(config['title'])
            
        elif chart_type == 'pie_chart':
            column = config['column']
            limit = config.get('limit', 10)
            
            value_counts = data[column].value_counts().head(limit)
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index, 
                                             autopct='%1.1f%%', startangle=90)
            
            ax.set_title(config['title'])
            
        elif chart_type == 'histogram':
            column = config['column']
            bins = config.get('bins', 30)
            
            clean_data = data[column].dropna()
            n, bins, patches = ax.hist(clean_data, bins=bins, alpha=0.7, edgecolor='black')
            
            # Add mean and median lines
            mean_val = clean_data.mean()
            median_val = clean_data.median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.set_title(config['title'])
            ax.legend()
            
        elif chart_type == 'scatter_plot':
            x_col = config['x_column']
            y_col = config['y_column']
            
            clean_data = data[[x_col, y_col]].dropna()
            ax.scatter(clean_data[x_col], clean_data[y_col], alpha=0.6)
            
            # Add correlation line
            if len(clean_data) > 1:
                z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
                p = np.poly1d(z)
                ax.plot(clean_data[x_col], p(clean_data[x_col]), "r--", alpha=0.8)
                
                correlation = clean_data[x_col].corr(clean_data[y_col])
                ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(config['title'])
            
        elif chart_type == 'correlation_heatmap':
            numeric_cols = config.get('columns', data.select_dtypes(include=[np.number]).columns)
            corr_matrix = data[numeric_cols].corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.2f', ax=ax)
            ax.set_title(config['title'])
            
        elif chart_type == 'line_plot':
            x_col = config['x_column']
            y_col = config['y_column']
            
            clean_data = data[[x_col, y_col]].dropna().sort_values(x_col)
            ax.plot(clean_data[x_col], clean_data[y_col], marker='o', markersize=3, linewidth=2)
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(config['title'])
            ax.grid(True, alpha=0.3)
            
        elif chart_type == 'stacked_bar_chart':
            numeric_col = config['numeric_column']
            category_col = config['category_column']
            segment_col = config['segment_column']
            
            # Create pivot table for stacked bar chart
            pivot_data = data.pivot_table(values=numeric_col, index=category_col, 
                                        columns=segment_col, aggfunc='mean', fill_value=0)
            
            pivot_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            ax.set_xlabel(category_col)
            ax.set_ylabel(f'Average {numeric_col}')
            ax.set_title(config['title'])
            ax.legend(title=segment_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == 'violin_plot':
            numeric_col = config['numeric_column']
            categorical_col = config.get('categorical_column')
            
            if categorical_col:
                clean_data = data[[numeric_col, categorical_col]].dropna()
                sns.violinplot(data=clean_data, x=categorical_col, y=numeric_col, ax=ax)
                ax.set_xlabel(categorical_col)
                ax.set_ylabel(numeric_col)
                plt.xticks(rotation=45, ha='right')
            else:
                clean_data = data[numeric_col].dropna()
                sns.violinplot(y=clean_data, ax=ax)
                ax.set_ylabel(numeric_col)
                ax.set_xlabel('Distribution')
            
            ax.set_title(config['title'])
            
        elif chart_type == 'donut_chart':
            column = config['column']
            limit = config.get('limit', 10)
            
            value_counts = data[column].value_counts().head(limit)
            
            # Create donut chart
            wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index, 
                                             autopct='%1.1f%%', startangle=90, 
                                             wedgeprops=dict(width=0.5))
            
            # Add center circle for donut effect
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            ax.add_artist(centre_circle)
            
            ax.set_title(config['title'])
            
        elif chart_type == 'pair_plot':
            columns = config['columns']
            hue_col = config.get('hue_column')
            
            if len(columns) >= 2:
                # Create a simple pair plot with selected columns
                plot_data = data[columns].dropna()
                
                if len(columns) == 2:
                    ax.scatter(plot_data[columns[0]], plot_data[columns[1]], alpha=0.6)
                    ax.set_xlabel(columns[0])
                    ax.set_ylabel(columns[1])
                else:
                    # For more than 2 columns, show correlation matrix
                    corr_matrix = plot_data.corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                               square=True, fmt='.2f', ax=ax)
                
                ax.set_title(config['title'])
            
        elif chart_type == 'statistical_summary':
            columns = config['columns']
            
            # Create a summary statistics visualization
            summary_data = data[columns].describe().T
            
            # Plot means with error bars (std)
            means = summary_data['mean']
            stds = summary_data['std']
            
            x_pos = range(len(means))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            
            ax.set_xlabel('Variables')
            ax.set_ylabel('Mean ± Standard Deviation')
            ax.set_title(config['title'])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(means.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + stds.iloc[i],
                       f'{means.iloc[i]:.1f}', ha='center', va='bottom', fontsize=8)
                       
        elif chart_type == 'outlier_analysis':
            column = config['column']
            
            clean_data = data[column].dropna()
            
            # Calculate outliers using IQR method
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            
            # Create box plot with outlier highlighting
            ax.boxplot(clean_data, vert=True)
            
            # Highlight outliers
            if len(outliers) > 0:
                ax.scatter([1] * len(outliers), outliers, color='red', s=50, alpha=0.7, label='Outliers')
                ax.legend()
            
            ax.set_ylabel(column)
            ax.set_xlabel('Distribution')
            ax.set_title(f'{config["title"]} ({len(outliers)} outliers found)')
            
            # Add statistics text
            stats_text = f'Q1: {Q1:.2f}\nMedian: {clean_data.median():.2f}\nQ3: {Q3:.2f}\nOutliers: {len(outliers)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Convert to base64 image
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return fig, img_base64
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error creating chart:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle="round", facecolor='red', alpha=0.3))
        ax.set_title(f'Error: {config["title"]}')
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return fig, img_base64

def get_visualization_recommendations(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate diverse and insightful visualizations"""
    recommendations = []
    
    numeric_cols = analysis['numeric_columns']
    categorical_cols = analysis['categorical_columns']
    
    print(f"DEBUG: Found {len(numeric_cols)} numeric columns: {numeric_cols}")
    print(f"DEBUG: Found {len(categorical_cols)} categorical columns: {categorical_cols}")
    
    # 1. SCATTER PLOT with trend line - Always add if we have 2+ numeric columns
    if len(numeric_cols) >= 2:
        recommendations.append({
            'type': 'scatter_plot',
            'title': f'Relationship Analysis: {numeric_cols[0]} vs {numeric_cols[1]}',
            'description': f'Correlation and trend between {numeric_cols[0]} and {numeric_cols[1]}',
            'x_column': numeric_cols[0],
            'y_column': numeric_cols[1],
        })
    
    # 2. STACKED BAR CHART - Multi-dimensional categorical analysis
    if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
        recommendations.append({
            'type': 'stacked_bar_chart',
            'title': f'Segmented Analysis: {numeric_cols[0]} by {categorical_cols[0]} & {categorical_cols[1]}',
            'description': f'Average {numeric_cols[0]} across {categorical_cols[0]} segments by {categorical_cols[1]}',
            'numeric_column': numeric_cols[0],
            'category_column': categorical_cols[0],
            'segment_column': categorical_cols[1],
        })
    
    # 3. VIOLIN PLOT - Distribution shape analysis
    if len(numeric_cols) >= 1:
        if len(categorical_cols) >= 1 and analysis['categorical_summary'][categorical_cols[0]]['unique_count'] <= 8:
            recommendations.append({
                'type': 'violin_plot',
                'title': f'Distribution Shapes: {numeric_cols[0]} by {categorical_cols[0]}',
                'description': f'Detailed distribution comparison of {numeric_cols[0]} across {categorical_cols[0]} groups',
                'numeric_column': numeric_cols[0],
                'categorical_column': categorical_cols[0],
            })
        else:
            recommendations.append({
                'type': 'histogram',
                'title': f'Distribution Analysis: {numeric_cols[0]}',
                'description': f'Statistical distribution of {numeric_cols[0]} with key metrics',
                'column': numeric_cols[0],
                'bins': 30,
            })
    
    # 4. HEATMAP - Multi-variable correlation
    if len(numeric_cols) >= 3:
        recommendations.append({
            'type': 'correlation_heatmap',
            'title': 'Variable Correlation Matrix',
            'description': 'Strength of relationships between all numeric variables',
            'columns': numeric_cols[:8],  # Limit for readability
        })
    
    # 5. PAIR PLOT - Multiple relationships
    if len(numeric_cols) >= 3:
        recommendations.append({
            'type': 'pair_plot',
            'title': f'Multi-Variable Relationships',
            'description': f'Pairwise relationships between top numeric variables',
            'columns': numeric_cols[:4],  # Limit to 4 for performance
            'hue_column': categorical_cols[0] if categorical_cols and analysis['categorical_summary'][categorical_cols[0]]['unique_count'] <= 6 else None,
        })
    
    # 6. TIME SERIES or DONUT CHART
    time_cols = [col for col in analysis['columns'] if any(word in col.lower() for word in ['date', 'time', 'year', 'month'])]
    if time_cols and len(numeric_cols) >= 1:
        recommendations.append({
            'type': 'line_plot',
            'title': f'Trend Analysis: {numeric_cols[0]} Over Time',
            'description': f'Time-based progression of {numeric_cols[0]}',
            'x_column': time_cols[0],
            'y_column': numeric_cols[0],
        })
    elif categorical_cols:
        col = categorical_cols[0]
        if 3 <= analysis['categorical_summary'][col]['unique_count'] <= 10:
            recommendations.append({
                'type': 'donut_chart',
                'title': f'Composition Analysis: {col}',
                'description': f'Proportional breakdown of {col} categories with percentages',
                'column': col,
                'limit': 10
            })
    
    # 7. STATISTICAL SUMMARY VISUALIZATION
    if len(numeric_cols) >= 2:
        recommendations.append({
            'type': 'statistical_summary',
            'title': 'Statistical Overview',
            'description': 'Key statistics and distributions for numeric variables',
            'columns': numeric_cols[:6],
        })
    
    # 8. OUTLIER ANALYSIS
    if len(numeric_cols) >= 1:
        recommendations.append({
            'type': 'outlier_analysis',
            'title': f'Outlier Detection: {numeric_cols[0]}',
            'description': f'Identifying unusual values in {numeric_cols[0]}',
            'column': numeric_cols[0],
        })
    
    print(f"DEBUG: Total recommendations: {len(recommendations)}")
    for i, rec in enumerate(recommendations):
        print(f"  {i+1}. {rec['type']}: {rec['title']}")
    
    # Return up to 8 visualizations for more insights
    return recommendations[:8]

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    st.title("📊 BI Dashboard (Matplotlib Version)")
    st.markdown("---")
    
    # Initialize session state
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = {}
    if 'viz_counter' not in st.session_state:
        st.session_state.viz_counter = 0
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file to get started with visualization analysis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.dataset = df
                st.session_state.analysis = analyze_dataset(df)
                st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Show basic info
                st.write("**Dataset Info:**")
                st.write(f"• Shape: {df.shape}")
                st.write(f"• Numeric columns: {len(st.session_state.analysis['numeric_columns'])}")
                st.write(f"• Categorical columns: {len(st.session_state.analysis['categorical_columns'])}")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    # Main content
    if st.session_state.dataset is None:
        st.info("👈 Please upload a CSV file to get started")
        return
    
    # Dataset overview
    st.header("📋 Dataset Overview & Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", st.session_state.dataset.shape[0])
    with col2:
        st.metric("Columns", st.session_state.dataset.shape[1])
    with col3:
        st.metric("Numeric Cols", len(st.session_state.analysis['numeric_columns']))
    with col4:
        st.metric("Categorical Cols", len(st.session_state.analysis['categorical_columns']))
    
    # Detailed Data Summary
    st.subheader("🔍 Detailed Data Summary")
    
    # Create tabs for different types of analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sample Data", "📈 Numeric Summary", "🏷️ Categorical Summary", "🧮 Missing Values"])
    
    with tab1:
        st.write("**First 10 rows of your dataset:**")
        st.dataframe(st.session_state.dataset.head(10), use_container_width=True)
        
        st.write("**Dataset Info:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Column Names & Types:**")
            dtype_df = pd.DataFrame({
                'Column': st.session_state.dataset.columns,
                'Data Type': [str(dtype) for dtype in st.session_state.dataset.dtypes],
                'Non-Null Count': [st.session_state.dataset[col].count() for col in st.session_state.dataset.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.write("**Memory Usage:**")
            memory_usage = st.session_state.dataset.memory_usage(deep=True)
            memory_df = pd.DataFrame({
                'Column': ['Index'] + list(st.session_state.dataset.columns),
                'Memory (bytes)': memory_usage.values
            })
            memory_df['Memory (MB)'] = memory_df['Memory (bytes)'] / (1024 * 1024)
            st.dataframe(memory_df, use_container_width=True)
    
    with tab2:
        if st.session_state.analysis['numeric_columns']:
            st.write("**Statistical Summary for Numeric Columns:**")
            numeric_summary = st.session_state.dataset[st.session_state.analysis['numeric_columns']].describe()
            st.dataframe(numeric_summary, use_container_width=True)
            
            # Correlation insights
            if len(st.session_state.analysis['numeric_columns']) > 1:
                st.write("**Key Correlations:**")
                corr_matrix = st.session_state.analysis['correlation_matrix']
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                
                st.dataframe(corr_df.head(10), use_container_width=True)
                
                # Insights
                strongest = corr_df.iloc[0]
                st.info(f"� **Strongest correlation:** {strongest['Variable 1']} and {strongest['Variable 2']} (r = {strongest['Correlation']:.3f})")
        else:
            st.info("No numeric columns found in the dataset.")
    
    with tab3:
        if st.session_state.analysis['categorical_columns']:
            st.write("**Categorical Variables Summary:**")
            
            for col in st.session_state.analysis['categorical_columns'][:5]:  # Show first 5
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    unique_count = st.session_state.analysis['categorical_summary'][col]['unique_count']
                    st.metric(f"{col}", f"{unique_count} categories")
                    
                    # Show top categories
                    top_values = st.session_state.analysis['categorical_summary'][col]['top_values']
                    st.write("**Top Categories:**")
                    for cat, count in list(top_values.items())[:3]:
                        st.write(f"• {cat}: {count}")
                
                with col2:
                    # Quick bar chart for this categorical variable
                    value_counts = st.session_state.dataset[col].value_counts().head(10)
                    
                    fig_mini, ax_mini = plt.subplots(figsize=(6, 3))
                    bars = ax_mini.bar(range(len(value_counts)), value_counts.values)
                    ax_mini.set_xticks(range(len(value_counts)))
                    ax_mini.set_xticklabels(value_counts.index, rotation=45, ha='right')
                    ax_mini.set_title(f'{col} Distribution')
                    ax_mini.set_ylabel('Count')
                    
                    # Add value labels on bars
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax_mini.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                   f'{int(height)}', ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig_mini)
                    plt.close(fig_mini)
                
                st.markdown("---")
        else:
            st.info("No categorical columns found in the dataset.")
    
    with tab4:
        st.write("**Missing Values Analysis:**")
        missing_data = st.session_state.dataset.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': (missing_data.values / len(st.session_state.dataset) * 100).round(2)
            })
            
            st.dataframe(missing_df, use_container_width=True)
            
            # Visualization of missing data
            if len(missing_data) > 0:
                fig_missing, ax_missing = plt.subplots(figsize=(8, 4))
                bars = ax_missing.bar(range(len(missing_df)), missing_df['Missing %'])
                ax_missing.set_xticks(range(len(missing_df)))
                ax_missing.set_xticklabels(missing_df['Column'], rotation=45, ha='right')
                ax_missing.set_title('Missing Data Percentage by Column')
                ax_missing.set_ylabel('Missing %')
                
                # Color bars based on severity
                for i, bar in enumerate(bars):
                    pct = missing_df.iloc[i]['Missing %']
                    if pct > 50:
                        bar.set_color('red')
                    elif pct > 20:
                        bar.set_color('orange')
                    else:
                        bar.set_color('yellow')
                
                plt.tight_layout()
                st.pyplot(fig_missing)
                plt.close(fig_missing)
                
                # Recommendations
                high_missing = missing_df[missing_df['Missing %'] > 50]
                if len(high_missing) > 0:
                    st.warning(f"⚠️ **High missing data:** {', '.join(high_missing['Column'].tolist())} have >50% missing values. Consider removing these columns or imputing values.")
                
        else:
            st.success("✅ **Great!** No missing values found in the dataset.")
            
        # Data quality score
        total_cells = st.session_state.dataset.shape[0] * st.session_state.dataset.shape[1]
        missing_cells = st.session_state.dataset.isnull().sum().sum()
        quality_score = ((total_cells - missing_cells) / total_cells * 100)
        
        st.metric("📊 Data Quality Score", f"{quality_score:.1f}%", 
                 help="Percentage of non-missing values in the dataset")
    
    # Auto-generate visualizations
    st.header("📈 Automatic Visualizations")
    
    if st.button("🚀 Generate Visualizations"):
        recommendations = get_visualization_recommendations(st.session_state.analysis)
        
        st.session_state.visualizations = {}
        
        for i, rec in enumerate(recommendations[:6]):  # Limit to 6 charts
            viz_id = f"auto_{i}"
            st.session_state.visualizations[viz_id] = rec
    
    # Display visualizations
    if st.session_state.visualizations:
        st.markdown("### 🎨 Generated Visualizations")
        
        # Create grid layout
        cols_per_row = 2
        viz_items = list(st.session_state.visualizations.items())
        
        for i in range(0, len(viz_items), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                if i + j < len(viz_items):
                    viz_id, viz_config = viz_items[i + j]
                    
                    with cols[j]:
                        st.markdown(f"""
                        <div style="
                            border: 2px solid #e1e5e9;
                            border-radius: 15px;
                            padding: 20px;
                            margin: 15px 0;
                            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        ">
                        """, unsafe_allow_html=True)
                        
                        # Chart type badge
                        chart_type_colors = {
                            'scatter_plot': '🔵', 'histogram': '📊', 'bar_chart': '📈',
                            'box_plot': '📦', 'correlation_heatmap': '🔥', 'pie_chart': '🥧',
                            'violin_plot': '🎻', 'stacked_bar_chart': '📊', 'donut_chart': '🍩',
                            'pair_plot': '🔗', 'statistical_summary': '📋', 'outlier_analysis': '🎯',
                            'line_plot': '📉'
                        }
                        
                        chart_emoji = chart_type_colors.get(viz_config['type'], '📊')
                        st.markdown(f"### {chart_emoji} {viz_config['title']}")
                        st.caption(viz_config['description'])
                        
                        try:
                            fig, img_base64 = create_matplotlib_figure(
                                viz_config['type'], 
                                st.session_state.dataset, 
                                viz_config
                            )
                            
                            # Display image
                            st.image(f"data:image/png;base64,{img_base64}", use_column_width=True)
                            
                            # Add insights based on chart type
                            if viz_config['type'] == 'scatter_plot':
                                x_col, y_col = viz_config['x_column'], viz_config['y_column']
                                correlation = st.session_state.dataset[x_col].corr(st.session_state.dataset[y_col])
                                if abs(correlation) > 0.7:
                                    st.success(f"💡 **Strong correlation** (r={correlation:.3f})")
                                elif abs(correlation) > 0.3:
                                    st.info(f"💡 **Moderate correlation** (r={correlation:.3f})")
                                else:
                                    st.warning(f"💡 **Weak correlation** (r={correlation:.3f})")
                            
                            elif viz_config['type'] == 'histogram':
                                col_data = st.session_state.dataset[viz_config['column']].dropna()
                                skewness = col_data.skew()
                                if abs(skewness) < 0.5:
                                    st.success("💡 **Nearly normal distribution**")
                                elif skewness > 0:
                                    st.info("💡 **Right-skewed distribution**")
                                else:
                                    st.info("💡 **Left-skewed distribution**")
                            
                            elif viz_config['type'] == 'outlier_analysis':
                                col_data = st.session_state.dataset[viz_config['column']].dropna()
                                Q1, Q3 = col_data.quantile([0.25, 0.75])
                                IQR = Q3 - Q1
                                outliers = col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]
                                outlier_pct = len(outliers) / len(col_data) * 100
                                
                                if outlier_pct > 10:
                                    st.warning(f"💡 **High outlier rate:** {outlier_pct:.1f}%")
                                elif outlier_pct > 5:
                                    st.info(f"💡 **Moderate outliers:** {outlier_pct:.1f}%")
                                else:
                                    st.success(f"💡 **Low outlier rate:** {outlier_pct:.1f}%")
                            
                            # Download button with enhanced styling
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    "📥 Download PNG",
                                    data=base64.b64decode(img_base64),
                                    file_name=f"{viz_config['title'].replace(' ', '_')}.png",
                                    mime="image/png",
                                    key=f"download_{viz_id}",
                                    use_container_width=True
                                )
                            
                            with col2:
                                # Add a "View Details" expander
                                with st.expander("📋 Chart Info"):
                                    st.write(f"**Type:** {viz_config['type'].replace('_', ' ').title()}")
                                    if 'column' in viz_config:
                                        st.write(f"**Column:** {viz_config['column']}")
                                    if 'x_column' in viz_config:
                                        st.write(f"**X-axis:** {viz_config['x_column']}")
                                        st.write(f"**Y-axis:** {viz_config['y_column']}")
                            
                            st.success("✅ Chart generated successfully!")
                            
                            plt.close(fig)  # Clean up
                            
                        except Exception as e:
                            st.error(f"❌ Error creating {viz_config['type']}: {str(e)}")
                            st.code(f"Config: {viz_config}", language='python')
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export all option
        if st.button("📦 Download All Charts (ZIP)"):
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for viz_id, viz_config in st.session_state.visualizations.items():
                    try:
                        fig, img_base64 = create_matplotlib_figure(
                            viz_config['type'],
                            st.session_state.dataset, 
                            viz_config
                        )
                        
                        img_data = base64.b64decode(img_base64)
                        filename = f"{viz_config['title'].replace(' ', '_')}.png"
                        zip_file.writestr(filename, img_data)
                        
                        plt.close(fig)
                        
                    except Exception as e:
                        st.warning(f"Skipped {viz_config['title']}: {str(e)}")
            
            zip_buffer.seek(0)
            
            st.download_button(
                "📥 Download ZIP File",
                data=zip_buffer.getvalue(),
                file_name=f"charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()