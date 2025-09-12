from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from langchain.schema import AgentAction, AgentFinish
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.tools import get_tools
import requests
import logging

logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        self.llm = None
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with Ollama LLM"""
        try:
            # Check if Ollama is running
            self._check_ollama_connection()
            
            # Initialize Ollama LLM
            self.llm = Ollama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1
            )
            
            # Get tools
            tools = get_tools()
            
            # Initialize agent
            self.agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            
            logger.info("AI Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Agent: {str(e)}")
            raise
    
    def _check_ollama_connection(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5)
            response.raise_for_status()
            logger.info("Ollama connection verified")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Ollama server not accessible at {OLLAMA_BASE_URL}. Please ensure Ollama is running. Error: {str(e)}")
    
    def query(self, question: str) -> dict:
        """Process a query using the AI agent"""
        if not self.agent:
            return {
                "answer": None,
                "tool_calls": [],
                "error": "Agent not initialized. Please check Ollama configuration."
            }
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Run the agent
            result = self.agent.run(question)
            
            return {
                "answer": result,
                "tool_calls": [],  # Would need to track this in custom callback
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": None,
                "tool_calls": [],
                "error": str(e)
            }
    
    def is_healthy(self) -> bool:
        """Check if the agent is healthy and ready"""
        try:
            self._check_ollama_connection()
            return self.agent is not None
        except:
            return False