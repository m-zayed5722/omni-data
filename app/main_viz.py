from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.models import (
    QueryRequest, QueryResponse, ErrorResponse, UploadResponse, 
    DataSummaryResponse, ConversationHistoryItem, ConversationHistoryResponse
)
from app.agent import AIAgent
from app.tools.visualization_tools import set_current_dataset, get_current_dataset, viz_tools
from backend.visualizations import DataVisualizationTools
import pandas as pd
import os
import uuid
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GenAI Data Visualization Dashboard",
    description="AI Agent with Data Visualization capabilities using LangChain + FastAPI + Ollama",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for uploaded files)
if not os.path.exists("uploads"):
    os.makedirs("uploads")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Global variables
ai_agent = None
conversation_history = []
current_file_info = None
last_visualization = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI agent on startup"""
    global ai_agent
    try:
        logger.info("Initializing GenAI Data Visualization Agent...")
        ai_agent = AIAgent()
        logger.info("GenAI Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GenAI Agent: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GenAI Data Visualization Dashboard API", 
        "status": "online",
        "version": "2.0.0",
        "capabilities": ["data_upload", "visualization", "natural_language_queries"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global current_file_info
    dataset_status = "loaded" if current_file_info else "no_dataset"
    
    # Check AI agent status
    ai_status = "disabled"
    ai_message = "AI Agent not initialized - basic visualization features available"
    
    if ai_agent is not None:
        if ai_agent.is_healthy():
            ai_status = "healthy"
            ai_message = "GenAI Data Visualization Agent is ready"
        else:
            ai_status = "unhealthy"
            ai_message = "AI Agent is not healthy - check Ollama connection - basic visualization features available"
    
    return {
        "status": "healthy", 
        "message": "Data Visualization Dashboard is ready",
        "ai_status": ai_status,
        "ai_message": ai_message,
        "dataset_status": dataset_status,
        "current_dataset": current_file_info
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV dataset for visualization
    
    The uploaded dataset will be available for natural language queries
    and visualization generation.
    """
    global current_file_info
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Generate unique filename
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join("uploads", unique_filename)
        
        # Save uploaded file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Load and validate CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            # Clean up file if CSV reading fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        # Check if DataFrame is empty
        if df.empty:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Set as current dataset for visualization tools
        set_current_dataset(df)
        
        # Store file info
        current_file_info = {
            "filename": file.filename,
            "unique_filename": unique_filename,
            "file_path": file_path,
            "upload_time": datetime.now().isoformat(),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        }
        
        logger.info(f"Successfully uploaded dataset: {file.filename} ({len(df)} rows, {len(df.columns)} columns)")
        
        return UploadResponse(
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
            column_names=list(df.columns),
            message=f"Successfully uploaded {file.filename}. You can now ask questions about your data!"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/dataset/summary", response_model=DataSummaryResponse)
async def get_dataset_summary():
    """Get summary information about the current dataset"""
    df = get_current_dataset()
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a CSV file first.")
    
    try:
        viz_tools = DataVisualizationTools()
        summary = viz_tools.get_data_summary(df)
        
        if "error" in summary:
            raise HTTPException(status_code=500, detail=summary["error"])
        
        return DataSummaryResponse(**summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset summary: {str(e)}")

@app.get("/dataset/suggestions")
async def get_visualization_suggestions():
    """Get visualization suggestions based on the current dataset"""
    df = get_current_dataset()
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a CSV file first.")
    
    try:
        viz_tools = DataVisualizationTools()
        suggestions = viz_tools.suggest_visualizations(df)
        
        if "error" in suggestions:
            raise HTTPException(status_code=500, detail=suggestions["error"])
        
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating suggestions: {str(e)}")

async def process_query_without_ai(query: str, df):
    """
    Fallback function to process basic visualization queries without AI
    """
    import re
    from backend.visualizations import DataVisualizationTools
    
    query_lower = query.lower()
    columns = df.columns.tolist()
    
    # Try to extract column names from the query
    mentioned_columns = [col for col in columns if col.lower() in query_lower]
    
    try:
        viz_tools = DataVisualizationTools()
        
        # Handle scatter plot requests
        if any(word in query_lower for word in ['scatter', 'scatter plot', 'vs', 'versus', 'against']):
            if len(mentioned_columns) >= 2:
                x_col, y_col = mentioned_columns[0], mentioned_columns[1]
                result = viz_tools.create_scatter_plot(df, x_col, y_col)
                if result.get("success"):
                    return QueryResponse(
                        answer=f"Created a scatter plot of {y_col} vs {x_col}. The visualization shows the relationship between these two variables.",
                        tool_calls=[{"tool": "create_scatter_plot", "args": {"x_column": x_col, "y_column": y_col}}],
                        visualization_data=result
                    )
            else:
                # Try with first two numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                    result = viz_tools.create_scatter_plot(df, x_col, y_col)
                    if result.get("success"):
                        return QueryResponse(
                            answer=f"Created a scatter plot of {y_col} vs {x_col} using the first two numeric columns in your dataset.",
                            tool_calls=[{"tool": "create_scatter_plot", "args": {"x_column": x_col, "y_column": y_col}}],
                            visualization_data=result
                        )
        
        # Handle histogram requests
        elif any(word in query_lower for word in ['histogram', 'distribution', 'hist']):
            if mentioned_columns:
                col = mentioned_columns[0]
                result = viz_tools.create_histogram(df, col)
                if result.get("success"):
                    return QueryResponse(
                        answer=f"Created a histogram showing the distribution of {col}.",
                        tool_calls=[{"tool": "create_histogram", "args": {"column": col}}],
                        visualization_data=result
                    )
            else:
                # Use first numeric column
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    col = numeric_cols[0]
                    result = viz_tools.create_histogram(df, col)
                    if result.get("success"):
                        return QueryResponse(
                            answer=f"Created a histogram showing the distribution of {col}.",
                            tool_calls=[{"tool": "create_histogram", "args": {"column": col}}],
                            visualization_data=result
                        )
        
        # Handle bar chart requests
        elif any(word in query_lower for word in ['bar', 'bar chart', 'compare', 'count']):
            if mentioned_columns:
                col = mentioned_columns[0]
                result = viz_tools.create_bar_chart(df, col)
                if result.get("success"):
                    return QueryResponse(
                        answer=f"Created a bar chart showing the counts for {col}.",
                        tool_calls=[{"tool": "create_bar_chart", "args": {"column": col}}],
                        visualization_data=result
                    )
        
        # Default response if no specific pattern matched
        return QueryResponse(
            answer="I couldn't interpret your specific visualization request. Here are some examples you can try:\n\n" +
                   "• 'Create a scatter plot of column1 vs column2'\n" +
                   "• 'Show histogram of column_name'\n" +
                   "• 'Create bar chart of category_column'\n\n" +
                   f"Available columns in your dataset: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}",
            error="query_not_understood"
        )
        
    except Exception as e:
        logger.error(f"Error in fallback query processing: {str(e)}")
        return QueryResponse(
            answer=f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your query or use the manual visualization options.",
            error="processing_error"
        )

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Process a natural language query about your data
    
    Examples:
    - "Show me a histogram of Age"
    - "Create a scatter plot of Salary vs Experience"
    - "What's the distribution of Sales by Region?"
    - "Compare Price across different Categories"
    """
    global ai_agent, conversation_history, last_visualization
    
    # Check if dataset is loaded
    df = get_current_dataset()
    if df is None:
        return QueryResponse(
            answer="No dataset loaded. Please upload a CSV file first to start creating visualizations!",
            error="no_dataset_loaded"
        )

    try:
        logger.info(f"Processing visualization query: {request.query}")
        
        # If AI agent is not available, try basic query parsing
        if ai_agent is None or not ai_agent.is_healthy():
            logger.info("AI agent not available, using fallback query parsing")
            return await process_query_without_ai(request.query, df)
        
        # Process the query with enhanced prompt for data visualization
        enhanced_query = f"""
        Dataset context: We have a dataset with {len(df)} rows and {len(df.columns)} columns.
        Available columns: {', '.join(df.columns)}
        
        User query: {request.query}
        
        Please analyze this query and create appropriate visualizations or provide helpful information about the data.
        If the user asks for unavailable visualization types (like heatmap, line plot, etc.), 
        suggest the closest alternative from available options: histogram, scatter plot, bar chart, or box plot.
        """
        
        # Let AI agent take as much time as it needs - no timeout interruption
        try:
            result = ai_agent.query(enhanced_query)
            logger.info(f"AI agent result: {result}")
        except Exception as ai_error:
            logger.error(f"AI agent exception: {str(ai_error)}, falling back to manual parsing")
            return await process_query_without_ai(request.query, df)
        
        # Check if result is valid
        if not isinstance(result, dict):
            logger.error(f"Invalid result type from AI agent: {type(result)}, result: {result}")
            return await process_query_without_ai(request.query, df)
            
        if result.get("error"):
            logger.error(f"AI agent returned error: {result['error']}")
            return QueryResponse(
                answer=f"I encountered an error: {result['error']}",
                tool_calls=result.get("tool_calls", []),
                error=result["error"]
            )
        
        # Check if the AI agent result contains incomplete parsing (shows "Thought:" or "Action:")    
        answer = result.get("answer", "")
        if not answer or "Thought:" in answer or "Action:" in answer or len(answer.strip()) < 10:
            logger.warning(f"AI agent returned incomplete result, falling back to manual parsing. Result: {answer[:100]}...")
            return await process_query_without_ai(request.query, df)
        
        # Try to extract the last visualization from the agent's tool usage
        visualization_data = None
        try:
            from app.tools.visualization_tools import last_viz_result
            
            logger.info(f"Checking last_viz_result: {last_viz_result}")
            
            # Check if a visualization was generated
            if last_viz_result:
                visualization_data = {
                    "type": last_viz_result.get("type", "unknown"),
                    "image": last_viz_result.get("image", ""),
                    "plotly_json": last_viz_result.get("plotly_json", {}),
                    "data_info": last_viz_result.get("data_info", {})
                }
                last_visualization = visualization_data
                logger.info(f"Extracted visualization data: {visualization_data}")
            else:
                logger.warning("last_viz_result is None or empty")
                # Check if query mentions specific visualization types
                query_lower = request.query.lower()
                if any(viz_type in query_lower for viz_type in ['histogram', 'scatter', 'bar', 'box']):
                    last_visualization = {"type": "visualization_requested"}
        except Exception as e:
            logger.warning(f"Could not extract visualization data: {e}")
            last_visualization = None
        
        # Add to conversation history
        history_item = ConversationHistoryItem(
            query=request.query,
            response=result["answer"],
            timestamp=datetime.now().isoformat(),
            visualization=last_visualization
        )
        conversation_history.append(history_item)
        
        # Keep only last 50 conversations
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        
        return QueryResponse(
            answer=result["answer"],
            tool_calls=result["tool_calls"],
            visualization=last_visualization
        )
        
    except Exception as e:
        logger.error(f"Unexpected error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(limit: int = 10):
    """Get recent conversation history"""
    global conversation_history
    
    recent_history = conversation_history[-limit:] if limit > 0 else conversation_history
    
    return ConversationHistoryResponse(
        history=recent_history,
        total=len(conversation_history)
    )

@app.delete("/history")
async def clear_conversation_history():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return {"message": "Conversation history cleared"}

@app.delete("/dataset")
async def clear_dataset():
    """Clear current dataset and conversation history"""
    global current_file_info, conversation_history
    
    # Remove uploaded file if it exists
    if current_file_info and os.path.exists(current_file_info.get("file_path", "")):
        try:
            os.remove(current_file_info["file_path"])
        except:
            pass
    
    # Clear dataset and history
    set_current_dataset(None)
    current_file_info = None
    conversation_history = []
    
    return {"message": "Dataset and conversation history cleared"}

@app.get("/tools")
async def list_tools():
    """List available tools"""
    if ai_agent is None:
        raise HTTPException(status_code=503, detail="AI Agent not initialized")
    
    from app.tools import get_tools
    tools = get_tools()
    
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in tools
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)