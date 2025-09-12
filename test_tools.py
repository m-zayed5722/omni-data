#!/usr/bin/env python3
"""
Direct test of tools without Ollama
"""

from app.tools.tools import get_tools

def test_tools_directly():
    """Test tools directly without LLM"""
    
    print("🔧 Direct Tools Test")
    print("=" * 30)
    
    tools = get_tools()
    
    # Test Calculator
    print("\n1. Testing Calculator Tool...")
    calc_tool = next(tool for tool in tools if tool.name == "calculator")
    result = calc_tool._run("5 * 12")
    print(f"✅ Calculator: {result}")
    
    result = calc_tool._run("100 / 4 + 25")
    print(f"✅ Calculator: {result}")
    
    # Test Weather (will show error without API key)
    print("\n2. Testing Weather Tool...")
    weather_tool = next(tool for tool in tools if tool.name == "weather")
    result = weather_tool._run("London")
    print(f"⚠️  Weather: {result}")
    
    # Test Web Search (will show error without API key)
    print("\n3. Testing Web Search Tool...")
    search_tool = next(tool for tool in tools if tool.name == "web_search")
    result = search_tool._run("AI news")
    print(f"⚠️  Search: {result}")
    
    print("\n✅ Tools test completed!")
    print("   - Calculator works without additional setup")
    print("   - Weather and Search need API keys in .env file")

if __name__ == "__main__":
    test_tools_directly()