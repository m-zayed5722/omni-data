import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import base64
import io
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

class DataVisualizationTools:
    """Tools for generating data visualizations"""
    
    def __init__(self):
        # Set default style for matplotlib/seaborn
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _validate_columns(self, df: pd.DataFrame, columns: list) -> Tuple[bool, str]:
        """Validate that columns exist in dataframe"""
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            available_cols = list(df.columns)
            return False, f"Columns {missing_cols} not found. Available columns: {available_cols}"
        return True, ""
    
    def _clean_data(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Basic data cleaning for specified columns"""
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                # Remove rows with missing values in this column
                df_clean = df_clean.dropna(subset=[col])
        
        return df_clean
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_base64
    
    def _plotly_to_json(self, fig) -> Dict:
        """Convert plotly figure to JSON"""
        return json.loads(pio.to_json(fig))
    
    def show_histogram(self, df: pd.DataFrame, feature: str, bins: int = 30) -> Dict[str, Any]:
        """
        Generate histogram for a single feature
        
        Args:
            df: DataFrame containing the data
            feature: Column name for the histogram
            bins: Number of bins for the histogram
            
        Returns:
            Dictionary with visualization data and metadata
        """
        # Validate input
        valid, error_msg = self._validate_columns(df, [feature])
        if not valid:
            return {"error": error_msg, "type": "histogram"}
        
        # Clean data
        df_clean = self._clean_data(df, [feature])
        if df_clean.empty:
            return {"error": f"No valid data found for column '{feature}'", "type": "histogram"}
        
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df_clean[feature]):
            return {"error": f"Column '{feature}' is not numeric. Histograms require numeric data.", "type": "histogram"}
        
        try:
            # Create plotly histogram
            fig = px.histogram(
                df_clean, 
                x=feature, 
                nbins=bins,
                title=f'Distribution of {feature}',
                labels={feature: feature, 'count': 'Frequency'}
            )
            fig.update_layout(
                showlegend=False,
                template="plotly_white"
            )
            
            # Create matplotlib version as backup
            plt_fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_clean[feature].dropna(), bins=bins, alpha=0.7, edgecolor='black')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feature}')
            ax.grid(True, alpha=0.3)
            
            return {
                "type": "histogram",
                "feature": feature,
                "plotly_json": self._plotly_to_json(fig),
                "image_base64": self._fig_to_base64(plt_fig),
                "data_info": {
                    "total_rows": len(df_clean),
                    "mean": float(df_clean[feature].mean()),
                    "std": float(df_clean[feature].std()),
                    "min": float(df_clean[feature].min()),
                    "max": float(df_clean[feature].max())
                }
            }
            
        except Exception as e:
            return {"error": f"Error creating histogram: {str(e)}", "type": "histogram"}
    
    def show_scatter(self, df: pd.DataFrame, feature_x: str, feature_y: str) -> Dict[str, Any]:
        """
        Generate scatter plot for two features
        
        Args:
            df: DataFrame containing the data
            feature_x: Column name for x-axis
            feature_y: Column name for y-axis
            
        Returns:
            Dictionary with visualization data and metadata
        """
        # Validate input
        valid, error_msg = self._validate_columns(df, [feature_x, feature_y])
        if not valid:
            return {"error": error_msg, "type": "scatter"}
        
        # Clean data
        df_clean = self._clean_data(df, [feature_x, feature_y])
        if df_clean.empty:
            return {"error": f"No valid data found for columns '{feature_x}' and '{feature_y}'", "type": "scatter"}
        
        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(df_clean[feature_x]):
            return {"error": f"Column '{feature_x}' is not numeric. Scatter plots require numeric data.", "type": "scatter"}
        if not pd.api.types.is_numeric_dtype(df_clean[feature_y]):
            return {"error": f"Column '{feature_y}' is not numeric. Scatter plots require numeric data.", "type": "scatter"}
        
        try:
            # Create plotly scatter plot
            fig = px.scatter(
                df_clean,
                x=feature_x,
                y=feature_y,
                title=f'{feature_y} vs {feature_x}',
                labels={feature_x: feature_x, feature_y: feature_y}
            )
            fig.update_layout(template="plotly_white")
            
            # Create matplotlib version
            plt_fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(df_clean[feature_x], df_clean[feature_y], alpha=0.6)
            ax.set_xlabel(feature_x)
            ax.set_ylabel(feature_y)
            ax.set_title(f'{feature_y} vs {feature_x}')
            ax.grid(True, alpha=0.3)
            
            # Calculate correlation
            correlation = df_clean[feature_x].corr(df_clean[feature_y])
            
            return {
                "type": "scatter",
                "feature_x": feature_x,
                "feature_y": feature_y,
                "plotly_json": self._plotly_to_json(fig),
                "image_base64": self._fig_to_base64(plt_fig),
                "data_info": {
                    "total_points": len(df_clean),
                    "correlation": float(correlation),
                    "x_range": [float(df_clean[feature_x].min()), float(df_clean[feature_x].max())],
                    "y_range": [float(df_clean[feature_y].min()), float(df_clean[feature_y].max())]
                }
            }
            
        except Exception as e:
            return {"error": f"Error creating scatter plot: {str(e)}", "type": "scatter"}
    
    def show_bar(self, df: pd.DataFrame, feature_x: str, feature_y: str) -> Dict[str, Any]:
        """
        Generate bar chart (categorical vs numeric)
        
        Args:
            df: DataFrame containing the data
            feature_x: Column name for categories (x-axis)
            feature_y: Column name for values (y-axis)
            
        Returns:
            Dictionary with visualization data and metadata
        """
        # Validate input
        valid, error_msg = self._validate_columns(df, [feature_x, feature_y])
        if not valid:
            return {"error": error_msg, "type": "bar"}
        
        # Clean data
        df_clean = self._clean_data(df, [feature_x, feature_y])
        if df_clean.empty:
            return {"error": f"No valid data found for columns '{feature_x}' and '{feature_y}'", "type": "bar"}
        
        # Check if y column is numeric
        if not pd.api.types.is_numeric_dtype(df_clean[feature_y]):
            return {"error": f"Column '{feature_y}' is not numeric. Bar charts require numeric values.", "type": "bar"}
        
        try:
            # Aggregate data by category (mean)
            agg_data = df_clean.groupby(feature_x)[feature_y].mean().reset_index()
            
            # Limit categories to top 20 to avoid overcrowded plots
            if len(agg_data) > 20:
                agg_data = agg_data.nlargest(20, feature_y)
            
            # Create plotly bar chart
            fig = px.bar(
                agg_data,
                x=feature_x,
                y=feature_y,
                title=f'Average {feature_y} by {feature_x}',
                labels={feature_x: feature_x, feature_y: f'Average {feature_y}'}
            )
            fig.update_layout(template="plotly_white")
            fig.update_xaxes(tickangle=45)
            
            # Create matplotlib version
            plt_fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(agg_data[feature_x].astype(str), agg_data[feature_y])
            ax.set_xlabel(feature_x)
            ax.set_ylabel(f'Average {feature_y}')
            ax.set_title(f'Average {feature_y} by {feature_x}')
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            return {
                "type": "bar",
                "feature_x": feature_x,
                "feature_y": feature_y,
                "plotly_json": self._plotly_to_json(fig),
                "image_base64": self._fig_to_base64(plt_fig),
                "data_info": {
                    "categories": len(agg_data),
                    "total_rows": len(df_clean),
                    "aggregation": "mean",
                    "top_category": agg_data.loc[agg_data[feature_y].idxmax(), feature_x],
                    "max_value": float(agg_data[feature_y].max())
                }
            }
            
        except Exception as e:
            return {"error": f"Error creating bar chart: {str(e)}", "type": "bar"}
    
    def show_boxplot(self, df: pd.DataFrame, feature_x: str, feature_y: str) -> Dict[str, Any]:
        """
        Generate box plot (distribution across categories)
        
        Args:
            df: DataFrame containing the data
            feature_x: Column name for categories
            feature_y: Column name for values
            
        Returns:
            Dictionary with visualization data and metadata
        """
        # Validate input
        valid, error_msg = self._validate_columns(df, [feature_x, feature_y])
        if not valid:
            return {"error": error_msg, "type": "boxplot"}
        
        # Clean data
        df_clean = self._clean_data(df, [feature_x, feature_y])
        if df_clean.empty:
            return {"error": f"No valid data found for columns '{feature_x}' and '{feature_y}'", "type": "boxplot"}
        
        # Check if y column is numeric
        if not pd.api.types.is_numeric_dtype(df_clean[feature_y]):
            return {"error": f"Column '{feature_y}' is not numeric. Box plots require numeric values.", "type": "boxplot"}
        
        try:
            # Limit categories for readability
            categories = df_clean[feature_x].value_counts()
            if len(categories) > 15:
                top_categories = categories.head(15).index
                df_clean = df_clean[df_clean[feature_x].isin(top_categories)]
            
            # Create plotly box plot
            fig = px.box(
                df_clean,
                x=feature_x,
                y=feature_y,
                title=f'Distribution of {feature_y} by {feature_x}',
                labels={feature_x: feature_x, feature_y: feature_y}
            )
            fig.update_layout(template="plotly_white")
            fig.update_xaxes(tickangle=45)
            
            # Create matplotlib/seaborn version
            plt_fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=df_clean, x=feature_x, y=feature_y, ax=ax)
            ax.set_xlabel(feature_x)
            ax.set_ylabel(feature_y)
            ax.set_title(f'Distribution of {feature_y} by {feature_x}')
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            return {
                "type": "boxplot",
                "feature_x": feature_x,
                "feature_y": feature_y,
                "plotly_json": self._plotly_to_json(fig),
                "image_base64": self._fig_to_base64(plt_fig),
                "data_info": {
                    "categories": len(df_clean[feature_x].unique()),
                    "total_rows": len(df_clean),
                    "y_median": float(df_clean[feature_y].median()),
                    "y_mean": float(df_clean[feature_y].mean()),
                    "y_std": float(df_clean[feature_y].std())
                }
            }
            
        except Exception as e:
            return {"error": f"Error creating box plot: {str(e)}", "type": "boxplot"}
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary information about the dataset"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            return {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "missing_values": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.astype(str).to_dict()
            }
        except Exception as e:
            return {"error": f"Error getting data summary: {str(e)}"}
    
    def suggest_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Suggest appropriate visualizations based on data types"""
        try:
            summary = self.get_data_summary(df)
            if "error" in summary:
                return summary
            
            suggestions = []
            
            # Histogram suggestions for numeric columns
            for col in summary["numeric_columns"]:
                suggestions.append({
                    "type": "histogram",
                    "description": f"Distribution of {col}",
                    "example_query": f"Show me a histogram of {col}"
                })
            
            # Scatter plot suggestions for pairs of numeric columns
            numeric_cols = summary["numeric_columns"]
            if len(numeric_cols) >= 2:
                for i, col1 in enumerate(numeric_cols[:3]):  # Limit suggestions
                    for col2 in numeric_cols[i+1:4]:
                        suggestions.append({
                            "type": "scatter",
                            "description": f"Relationship between {col1} and {col2}",
                            "example_query": f"Show me a scatter plot of {col1} vs {col2}"
                        })
            
            # Bar chart suggestions for categorical vs numeric
            for cat_col in summary["categorical_columns"][:2]:  # Limit suggestions
                for num_col in summary["numeric_columns"][:2]:
                    suggestions.append({
                        "type": "bar",
                        "description": f"Average {num_col} by {cat_col}",
                        "example_query": f"Show me a bar chart of {num_col} by {cat_col}"
                    })
            
            return {
                "suggestions": suggestions[:10],  # Limit to 10 suggestions
                "data_summary": summary
            }
        except Exception as e:
            return {"error": f"Error generating suggestions: {str(e)}"}