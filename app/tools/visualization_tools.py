from langchain.tools import BaseTool
from typing import Type, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
from backend.visualizations import DataVisualizationTools

# Global storage for the current dataset and last visualization result
current_dataset: pd.DataFrame = None
viz_tools = DataVisualizationTools()
last_viz_result: Dict[str, Any] = None

class HistogramInput(BaseModel):
    input: str = Field(description="Column name and optional bins count, e.g., 'age' or 'age,30'")

class HistogramTool(BaseTool):
    name: str = "show_histogram"
    description: str = """
    Create a histogram showing the distribution of a single numeric column.
    Input should be the column name, optionally followed by number of bins (e.g., 'age' or 'age,30').
    Use this when users ask for:
    - Distribution of a column
    - Histogram of a feature
    - "Show me the spread of [column]"
    - "What does [column] look like?"
    """
    args_schema: Type[BaseModel] = HistogramInput
    
    def _run(self, input: str) -> str:
        global current_dataset, last_viz_result
        if current_dataset is None:
            return "No dataset loaded. Please upload a CSV file first."
        
        try:
            # Clean and parse input - handle extra text that agent might add
            cleaned_input = input.strip().split('\n')[0]  # Take only first line
            parts = cleaned_input.strip().split(',')
            feature = parts[0].strip()
            bins = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 30
            
            result = viz_tools.show_histogram(current_dataset, feature, bins)
            if "error" in result:
                last_viz_result = None
                return f"❌ {result['error']}"
            
            # Store the result globally for the API to access
            last_viz_result = result
            
            data_info = result.get('data_info', {})
            return f"""✅ Created histogram for '{feature}'
📊 Statistics:
- Total data points: {data_info.get('total_rows', 'N/A')}
- Mean: {data_info.get('mean', 'N/A'):.2f}
- Standard deviation: {data_info.get('std', 'N/A'):.2f}
- Range: {data_info.get('min', 'N/A'):.2f} to {data_info.get('max', 'N/A'):.2f}

The histogram shows the distribution of {feature} values. You can see the visualization in the dashboard."""
        except Exception as e:
            return f"❌ Error creating histogram: {str(e)}"

class ScatterInput(BaseModel):
    input: str = Field(description="Two column names separated by comma, e.g., 'age,salary'")

class ScatterTool(BaseTool):
    name: str = "show_scatter"
    description: str = """
    Create a scatter plot showing the relationship between two numeric columns.
    Input should be two column names separated by comma (e.g., 'age,salary').
    Use this when users ask for:
    - Relationship between two variables
    - Scatter plot of X vs Y
    - "How does [column1] relate to [column2]?"
    - "Plot [column1] against [column2]"
    - When they ask for line plots (use scatter as alternative)
    """
    args_schema: Type[BaseModel] = ScatterInput
    
    def _run(self, input: str) -> str:
        global current_dataset, last_viz_result
        if current_dataset is None:
            return "No dataset loaded. Please upload a CSV file first."
        
        try:
            # Clean and parse input - handle extra text that agent might add
            cleaned_input = input.strip().split('\n')[0]  # Take only first line
            parts = cleaned_input.strip().split(',')
            if len(parts) != 2:
                return "❌ Please provide two column names separated by comma (e.g., 'age,salary')"
            
            feature_x = parts[0].strip()
            feature_y = parts[1].strip()
            
            result = viz_tools.show_scatter(current_dataset, feature_x, feature_y)
            if "error" in result:
                last_viz_result = None
                return f"❌ {result['error']}"
            
            # Store the result globally for the API to access
            last_viz_result = result
            
            data_info = result.get('data_info', {})
            correlation = data_info.get('correlation', 0)
            
            # Interpret correlation
            if abs(correlation) > 0.7:
                corr_desc = "strong"
            elif abs(correlation) > 0.3:
                corr_desc = "moderate"
            else:
                corr_desc = "weak"
            
            corr_direction = "positive" if correlation > 0 else "negative"
            
            return f"""✅ Created scatter plot: {feature_y} vs {feature_x}
📊 Analysis:
- Total data points: {data_info.get('total_points', 'N/A')}
- Correlation: {correlation:.3f} ({corr_desc} {corr_direction} relationship)
- X range: {data_info.get('x_range', ['N/A', 'N/A'])[0]:.2f} to {data_info.get('x_range', ['N/A', 'N/A'])[1]:.2f}
- Y range: {data_info.get('y_range', ['N/A', 'N/A'])[0]:.2f} to {data_info.get('y_range', ['N/A', 'N/A'])[1]:.2f}

The scatter plot reveals the relationship between {feature_x} and {feature_y}. You can see the visualization in the dashboard."""
        except Exception as e:
            return f"❌ Error creating scatter plot: {str(e)}"

class BarInput(BaseModel):
    input: str = Field(description="Categorical and numeric column names separated by comma, e.g., 'department,salary'")

class BarTool(BaseTool):
    name: str = "show_bar"
    description: str = """
    Create a bar chart showing average values for each category.
    Input should be categorical column and numeric column separated by comma (e.g., 'department,salary').
    Use this when users ask for:
    - Bar chart of categorical vs numeric data
    - "Average [numeric] by [category]"
    - "Compare [numeric] across [categories]"
    - "Show [category] performance"
    """
    args_schema: Type[BaseModel] = BarInput
    
    def _run(self, input: str) -> str:
        global current_dataset
        if current_dataset is None:
            return "No dataset loaded. Please upload a CSV file first."
        
        try:
            # Parse input
            parts = input.strip().split(',')
            if len(parts) != 2:
                return "❌ Please provide categorical and numeric column names separated by comma (e.g., 'department,salary')"
            
            feature_x = parts[0].strip()
            feature_y = parts[1].strip()
            
            result = viz_tools.show_bar(current_dataset, feature_x, feature_y)
            if "error" in result:
                return f"❌ {result['error']}"
            
            data_info = result.get('data_info', {})
            
            return f"""✅ Created bar chart: Average {feature_y} by {feature_x}
📊 Summary:
- Number of categories: {data_info.get('categories', 'N/A')}
- Total data points: {data_info.get('total_rows', 'N/A')}
- Top performing category: {data_info.get('top_category', 'N/A')}
- Highest average value: {data_info.get('max_value', 'N/A'):.2f}

The bar chart shows how {feature_y} varies across different {feature_x} categories. You can see the visualization in the dashboard."""
        except Exception as e:
            return f"❌ Error creating bar chart: {str(e)}"

class BoxplotInput(BaseModel):
    input: str = Field(description="Categorical and numeric column names separated by comma, e.g., 'department,salary'")

class BoxplotTool(BaseTool):
    name: str = "show_boxplot"
    description: str = """
    Create a box plot showing the distribution of numeric values across different categories.
    Input should be categorical column and numeric column separated by comma (e.g., 'department,salary').
    Use this when users ask for:
    - Box plot of numeric data by categories
    - "Distribution of [numeric] by [category]"
    - "Compare [numeric] distributions across [categories]"
    - "Show spread of [numeric] for each [category]"
    """
    args_schema: Type[BaseModel] = BoxplotInput
    
    def _run(self, input: str) -> str:
        global current_dataset
        if current_dataset is None:
            return "No dataset loaded. Please upload a CSV file first."
        
        try:
            # Parse input
            parts = input.strip().split(',')
            if len(parts) != 2:
                return "❌ Please provide categorical and numeric column names separated by comma (e.g., 'department,salary')"
            
            feature_x = parts[0].strip()
            feature_y = parts[1].strip()
            
            result = viz_tools.show_boxplot(current_dataset, feature_x, feature_y)
            if "error" in result:
                return f"❌ {result['error']}"
            
            data_info = result.get('data_info', {})
            
            return f"""✅ Created box plot: Distribution of {feature_y} by {feature_x}
📊 Statistics:
- Number of categories: {data_info.get('categories', 'N/A')}
- Total data points: {data_info.get('total_rows', 'N/A')}
- Overall median: {data_info.get('y_median', 'N/A'):.2f}
- Overall mean: {data_info.get('y_mean', 'N/A'):.2f}
- Standard deviation: {data_info.get('y_std', 'N/A'):.2f}

The box plot shows how {feature_y} is distributed across different {feature_x} categories, including medians, quartiles, and outliers. You can see the visualization in the dashboard."""
        except Exception as e:
            return f"❌ Error creating box plot: {str(e)}"

class DataSummaryInput(BaseModel):
    input: str = Field(default="", description="No input required")

class DataSummaryTool(BaseTool):
    name: str = "get_data_summary"
    description: str = """
    Get information about the current dataset including columns, data types, and basic statistics.
    Use this when users ask:
    - "What's in the dataset?"
    - "Show me the data structure"
    - "What columns are available?"
    - "Describe the data"
    """
    args_schema: Type[BaseModel] = DataSummaryInput
    
    def _run(self, input: str = "") -> str:
        global current_dataset
        if current_dataset is None:
            return "No dataset loaded. Please upload a CSV file first."
        
        try:
            summary = viz_tools.get_data_summary(current_dataset)
            if "error" in summary:
                return f"❌ {summary['error']}"
            
            numeric_cols = summary.get('numeric_columns', [])
            categorical_cols = summary.get('categorical_columns', [])
            
            result = f"""📊 Dataset Summary:
📈 Dimensions: {summary.get('total_rows', 'N/A')} rows × {summary.get('total_columns', 'N/A')} columns

🔢 Numeric Columns ({len(numeric_cols)}):
{', '.join(numeric_cols) if numeric_cols else 'None'}

📝 Categorical Columns ({len(categorical_cols)}):
{', '.join(categorical_cols) if categorical_cols else 'None'}

⚠️ Missing Values:
"""
            missing_values = summary.get('missing_values', {})
            has_missing = False
            for col, missing in missing_values.items():
                if missing > 0:
                    result += f"- {col}: {missing} missing\n"
                    has_missing = True
            
            if not has_missing:
                result += "No missing values found ✅"
            
            return result
        except Exception as e:
            return f"❌ Error getting data summary: {str(e)}"

class VisualizationSuggestionInput(BaseModel):
    input: str = Field(default="", description="No input required")

class VisualizationSuggestionTool(BaseTool):
    name: str = "suggest_visualizations"
    description: str = """
    Get suggestions for appropriate visualizations based on the current dataset.
    Use this when users ask:
    - "What can I visualize?"
    - "What charts should I make?"
    - "Give me suggestions"
    - "What analysis can I do?"
    """
    args_schema: Type[BaseModel] = VisualizationSuggestionInput
    
    def _run(self, input: str = "") -> str:
        global current_dataset
        if current_dataset is None:
            return "No dataset loaded. Please upload a CSV file first."
        
        try:
            suggestions = viz_tools.suggest_visualizations(current_dataset)
            if "error" in suggestions:
                return f"❌ {suggestions['error']}"
            
            result = "🎯 Visualization Suggestions:\n\n"
            
            for i, suggestion in enumerate(suggestions.get('suggestions', [])[:8], 1):
                result += f"{i}. **{suggestion['type'].title()}**: {suggestion['description']}\n"
                result += f"   Try: \"{suggestion['example_query']}\"\n\n"
            
            return result
        except Exception as e:
            return f"❌ Error generating suggestions: {str(e)}"

def set_current_dataset(df: pd.DataFrame):
    """Set the global dataset for visualization tools"""
    global current_dataset
    current_dataset = df

def get_current_dataset() -> pd.DataFrame:
    """Get the current dataset"""
    global current_dataset
    return current_dataset

def get_visualization_tools():
    """Return list of visualization tools for LangChain agent"""
    return [
        HistogramTool(),
        ScatterTool(),
        BarTool(),
        BoxplotTool(),
        DataSummaryTool(),
        VisualizationSuggestionTool()
    ]