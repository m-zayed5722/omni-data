#!/usr/bin/env python3
"""
Test client for GenAI Data Visualization API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8002"

def test_data_viz_api():
    """Test the GenAI Data Visualization API"""
    
    print("🤖 GenAI Data Visualization API Test Client")
    print("=" * 50)
    
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
    
    # Test file upload (with sample data)
    print("\n3. Testing file upload...")
    try:
        # Upload sample CSV file
        csv_file_path = "sample_data.csv"
        try:
            with open(csv_file_path, 'rb') as f:
                files = {'file': ('sample_data.csv', f, 'text/csv')}
                response = requests.post(f"{BASE_URL}/upload", files=files)
                
            if response.status_code == 200:
                upload_data = response.json()
                print(f"✅ File uploaded: {upload_data['filename']}")
                print(f"   📊 {upload_data['rows']} rows × {upload_data['columns']} columns")
                print(f"   Columns: {upload_data['column_names']}")
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                
        except FileNotFoundError:
            print("⚠️  sample_data.csv not found. Creating a simple test CSV...")
            # Create a simple test CSV for testing
            test_csv_content = """name,age,salary,department
Alice,25,50000,Engineering
Bob,30,60000,Marketing
Carol,35,70000,Sales
David,28,55000,Engineering
Eva,32,65000,Marketing"""
            
            with open("test_data.csv", "w") as f:
                f.write(test_csv_content)
            
            with open("test_data.csv", 'rb') as f:
                files = {'file': ('test_data.csv', f, 'text/csv')}
                response = requests.post(f"{BASE_URL}/upload", files=files)
                
            if response.status_code == 200:
                upload_data = response.json()
                print(f"✅ Test file uploaded: {upload_data['filename']}")
            else:
                print(f"❌ Test upload failed: {response.status_code}")
                
    except Exception as e:
        print(f"❌ File upload test failed: {e}")
        return
    
    # Test dataset summary
    print("\n4. Testing dataset summary...")
    try:
        response = requests.get(f"{BASE_URL}/dataset/summary")
        if response.status_code == 200:
            summary = response.json()
            print(f"✅ Dataset summary:")
            print(f"   Rows: {summary.get('total_rows')}")
            print(f"   Columns: {summary.get('total_columns')}")
            print(f"   Numeric: {summary.get('numeric_columns')}")
            print(f"   Categorical: {summary.get('categorical_columns')}")
        else:
            print(f"❌ Summary failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Summary test failed: {e}")
    
    # Test visualization suggestions
    print("\n5. Testing visualization suggestions...")
    try:
        response = requests.get(f"{BASE_URL}/dataset/suggestions")
        if response.status_code == 200:
            suggestions = response.json()
            print(f"✅ Got {len(suggestions.get('suggestions', []))} suggestions:")
            for i, suggestion in enumerate(suggestions.get('suggestions', [])[:3], 1):
                print(f"   {i}. {suggestion['type']}: {suggestion['description']}")
        else:
            print(f"❌ Suggestions failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Suggestions test failed: {e}")
    
    # Test visualization queries
    test_queries = [
        "What's in the dataset?",
        "Show me a histogram of age",
        "Create a scatter plot of age vs salary",
        "Show me average salary by department",
        "Give me visualization suggestions"
    ]
    
    print(f"\n6. Testing visualization queries...")
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
                print(f"✅ Answer: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
                
                if result.get('visualization'):
                    print(f"📊 Visualization generated: {result['visualization'].get('type', 'unknown')}")
                
                if result.get('error'):
                    print(f"⚠️  Error: {result['error']}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏱️  Query timed out (30s)")
        except Exception as e:
            print(f"❌ Query failed: {e}")
        
        time.sleep(1)  # Small delay between requests
    
    # Test conversation history
    print("\n7. Testing conversation history...")
    try:
        response = requests.get(f"{BASE_URL}/history?limit=5")
        if response.status_code == 200:
            history = response.json()
            print(f"✅ Got {len(history.get('history', []))} conversation items")
            for item in history.get('history', [])[:2]:
                print(f"   Q: {item['query'][:50]}...")
                print(f"   A: {item['response'][:50]}...")
        else:
            print(f"❌ History failed: {response.status_code}")
    except Exception as e:
        print(f"❌ History test failed: {e}")
    
    print(f"\n🎉 GenAI Data Visualization API tests completed!")
    print(f"💡 You can now:")
    print(f"   - Access the API docs at: http://localhost:8000/docs")
    print(f"   - Run the Streamlit frontend at: streamlit run frontend/streamlit_app.py")
    print(f"   - Upload your own CSV files for analysis")

if __name__ == "__main__":
    test_data_viz_api()