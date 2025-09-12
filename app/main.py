from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import QueryRequest, QueryResponse, ErrorResponse
from app.agent import AIAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="AI Agent using LangChain + FastAPI + Ollama with external tools",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Agent (will be done on startup)
ai_agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI agent on startup"""
    global ai_agent
    try:
        logger.info("Initializing AI Agent...")
        ai_agent = AIAgent()
        logger.info("AI Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI Agent: {str(e)}")
        # Don't raise here, let the health check handle it

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Agent API is running", "status": "online"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if ai_agent is None:
        raise HTTPException(status_code=503, detail="AI Agent not initialized")
    
    if not ai_agent.is_healthy():
        raise HTTPException(status_code=503, detail="AI Agent is not healthy - check Ollama connection")
    
    return {"status": "healthy", "message": "AI Agent is ready"}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Process a natural language query using the AI agent
    
    Example request:
    {
        "query": "What's the weather in Toronto and what is 5 * 12?"
    }
    """
    if ai_agent is None:
        raise HTTPException(
            status_code=503, 
            detail="AI Agent not initialized. Please check if Ollama is running."
        )
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Process the query
        result = ai_agent.query(request.query)
        
        if result["error"]:
            # Return error in response format rather than raising exception
            return QueryResponse(
                answer=f"I encountered an error: {result['error']}",
                tool_calls=result["tool_calls"],
                error=result["error"]
            )
        
        return QueryResponse(
            answer=result["answer"],
            tool_calls=result["tool_calls"],
            error=None
        )
        
    except Exception as e:
        logger.error(f"Unexpected error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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