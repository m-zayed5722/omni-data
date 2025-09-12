#!/usr/bin/env python3
"""
Test client for AI Agent API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the AI Agent API"""
    
    print("🤖 AI Agent API Test Client")
    print("=" * 40)
    
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Root: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return
    
    # Test health endpoint
    print("\n2. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        health_data = response.json()
        print(f"✅ Health: {health_data}")
        
        if health_data.get("status") != "healthy":
            print("⚠️  Agent is not healthy. Check Ollama setup.")
            print("   Run: ollama serve")
            print("   Run: ollama pull mistral")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test tools endpoint
    print("\n3. Testing tools endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/tools")
        tools = response.json()
        print(f"✅ Available tools:")
        for tool in tools["tools"]:
            print(f"   - {tool['name']}: {tool['description']}")
    except Exception as e:
        print(f"❌ Tools endpoint failed: {e}")
        return
    
    # Test queries
    test_queries = [
        "What is 25 * 4?",
        "Calculate 100 / 5 + 10",
        "What's the weather in London?",
        "Search for latest AI news",
        "What's the weather in Toronto and what is 5 * 12?"
    ]
    
    print(f"\n4. Testing queries...")
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                json={"query": query},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Answer: {result['answer']}")
                if result.get('error'):
                    print(f"⚠️  Error: {result['error']}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏱️  Query timed out (30s)")
        except Exception as e:
            print(f"❌ Query failed: {e}")
        
        time.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    test_api()