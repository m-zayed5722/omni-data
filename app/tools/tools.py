from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import requests
import json
import ast
import operator
from app.config import TAVILY_API_KEY, SERP_API_KEY, OPENWEATHER_API_KEY, OPENWEATHER_BASE_URL

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Use this tool for mathematical calculations. Input should be a mathematical expression like '5 * 12' or '100 + 50 * 2'"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        try:
            # Safe evaluation of mathematical expressions
            # Remove any whitespace and validate expression
            expression = expression.strip()
            
            # Simple validation - only allow numbers, operators, parentheses, and spaces
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return f"Error: Invalid characters in expression '{expression}'. Only numbers and basic operators (+, -, *, /, (, )) are allowed."
            
            # Use eval with restricted builtins for basic math
            result = eval(expression, {"__builtins__": {}}, {})
            return f"The answer is {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query to look up on the web")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current information. Use this when you need up-to-date information about news, events, or general knowledge."
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str) -> str:
        try:
            # Try Tavily first if API key is available
            if TAVILY_API_KEY and TAVILY_API_KEY != "your_tavily_api_key_here":
                return self._tavily_search(query)
            # Fallback to SerpAPI if available
            elif SERP_API_KEY and SERP_API_KEY != "your_serpapi_key_here":
                return self._serp_search(query)
            else:
                return "Error: No web search API key configured. Please set TAVILY_API_KEY or SERP_API_KEY in your environment."
        except Exception as e:
            return f"Error searching for '{query}': {str(e)}"
    
    def _tavily_search(self, query: str) -> str:
        try:
            try:
                from tavily import TavilyClient
            except ImportError:
                return "Error: Tavily client not available. Please install with: pip install tavily-python"
            
            client = TavilyClient(api_key=TAVILY_API_KEY)
            results = client.search(query=query, max_results=3)
            
            if results and 'results' in results:
                search_results = []
                for result in results['results'][:3]:
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', 'No URL')
                    search_results.append(f"Title: {title}\nContent: {content}\nURL: {url}\n")
                return "Web search results:\n" + "\n".join(search_results)
            else:
                return "No search results found."
        except Exception as e:
            return f"Tavily search error: {str(e)}"
    
    def _serp_search(self, query: str) -> str:
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": SERP_API_KEY,
                "engine": "google",
                "num": 3
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'organic_results' in data:
                search_results = []
                for result in data['organic_results'][:3]:
                    title = result.get('title', 'No title')
                    snippet = result.get('snippet', 'No snippet')
                    link = result.get('link', 'No link')
                    search_results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}\n")
                return "Web search results:\n" + "\n".join(search_results)
            else:
                return "No search results found."
        except Exception as e:
            return f"SerpAPI search error: {str(e)}"

class WeatherInput(BaseModel):
    city: str = Field(description="City name to get weather information for")

class WeatherTool(BaseTool):
    name: str = "weather"
    description: str = "Get current weather information for a specific city. Input should be the city name."
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str) -> str:
        try:
            if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your_openweather_api_key_here":
                return "Error: OpenWeatherMap API key not configured. Please set OPENWEATHER_API_KEY in your environment."
            
            params = {
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            }
            
            response = requests.get(OPENWEATHER_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("cod") == 200:
                weather = data["weather"][0]
                main = data["main"]
                temp = main["temp"]
                feels_like = main["feels_like"]
                humidity = main["humidity"]
                description = weather["description"]
                
                return f"Current weather in {city}: {temp}°C (feels like {feels_like}°C), {description.title()}, humidity: {humidity}%"
            else:
                return f"Error: Could not find weather data for {city}. Please check the city name."
                
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data for {city}: {str(e)}"
        except Exception as e:
            return f"Unexpected error getting weather for {city}: {str(e)}"

def get_tools():
    """Return list of available tools"""
    return [
        CalculatorTool(),
        WebSearchTool(),
        WeatherTool()
    ]