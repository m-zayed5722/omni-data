# 🎯 How to Use Your GenAI Data Visualization Dashboard

## 🚀 Quick Start (3 Simple Steps)

### Step 1: Start the System
```bash
python start_server.py
```
- API will be available at: **http://localhost:8002**
- Documentation at: **http://localhost:8002/docs** ← (Already opened for you!)

### Step 2: Upload Your Data
**Option A: Using the Web Interface (API Docs)**
1. Go to http://localhost:8002/docs
2. Find the **POST /upload** endpoint
3. Click "Try it out"
4. Choose your CSV file
5. Click "Execute"

**Option B: Using Python/curl**
```python
import requests

# Upload your CSV file
with open('your_data.csv', 'rb') as f:
    files = {'file': ('data.csv', f, 'text/csv')}
    response = requests.post('http://localhost:8002/upload', files=files)
    print(response.json())
```

### Step 3: Ask Questions in Natural Language
```python
import requests

# Ask for visualizations
response = requests.post('http://localhost:8002/ask', 
    json={'query': 'Create a scatter plot of age vs salary'})
    
result = response.json()
print("Answer:", result['answer'])

# Check if visualization was created
if result.get('visualization'):
    print("Visualization created!")
    viz = result['visualization']
    print("Type:", viz.get('type'))
    print("Has image:", bool(viz.get('image')))
```

## 💬 Example Questions You Can Ask

### 📊 **Data Exploration**
- "What's in this dataset?"
- "How many rows and columns do I have?"
- "What are the column names?"
- "Show me a summary of the data"

### 📈 **Histograms** (Distribution of one column)
- "Show me a histogram of age"
- "What's the distribution of prices?"
- "Create a histogram of customer ratings with 20 bins"

### 🔗 **Scatter Plots** (Relationship between two columns)
- "Create a scatter plot of age vs salary"
- "Plot height against weight"
- "Show me how price correlates with rating"

### 📊 **Bar Charts** (Categorical data)
- "Show me sales by department"
- "Create a bar chart of count by category"
- "Compare average salary by job title"

### 📦 **Box Plots** (Distribution comparison)
- "Make a box plot of salary by department"
- "Show me price distribution across categories"

## 🎨 Using the Test Data

I created sample data for you to test with:

```python
# Upload the test sample
import requests
with open('test_sample.csv', 'rb') as f:
    files = {'file': ('test.csv', f, 'text/csv')}
    requests.post('http://localhost:8002/upload', files=files)

# Try these example queries:
queries = [
    "What's in the dataset?",
    "Show me a histogram of age", 
    "Create a scatter plot of age vs salary",
    "Make a bar chart of count by department"
]

for query in queries:
    response = requests.post('http://localhost:8002/ask', json={'query': query})
    print(f"Q: {query}")
    print(f"A: {response.json()['answer'][:100]}...")
    print("---")
```

## 🌐 Complete Example Usage

```python
import requests
import json
import base64

# 1. Upload data
print("📤 Uploading data...")
with open('test_sample.csv', 'rb') as f:
    files = {'file': ('data.csv', f, 'text/csv')}
    upload_response = requests.post('http://localhost:8002/upload', files=files)
print("✅ Upload:", upload_response.json()['message'])

# 2. Get data summary
summary = requests.get('http://localhost:8002/dataset/summary').json()
print(f"📊 Dataset: {summary['total_rows']} rows, {summary['total_columns']} columns")

# 3. Create visualization
print("\n🎨 Creating scatter plot...")
viz_response = requests.post('http://localhost:8002/ask', 
    json={'query': 'Create a scatter plot of age vs salary'})

result = viz_response.json()
print("🤖 AI Response:", result['answer'][:200] + "...")

# 4. Check if visualization was generated
if result.get('visualization'):
    viz = result['visualization']
    print(f"\n✅ Visualization created!")
    print(f"   Type: {viz.get('type')}")
    print(f"   Has image data: {len(viz.get('image', '')) > 0}")
    print(f"   Has interactive plot: {bool(viz.get('plotly_json'))}")
    
    # Save the image if available
    if viz.get('image'):
        with open('my_plot.png', 'wb') as f:
            f.write(base64.b64decode(viz['image']))
        print("   💾 Saved plot as: my_plot.png")
```

## 🎯 **Your Dashboard is Ready!**

- **API Documentation**: http://localhost:8002/docs (already opened)
- **Upload endpoint**: POST /upload
- **Query endpoint**: POST /ask
- **Data summary**: GET /dataset/summary

**Try it now!** Upload your CSV file and start asking questions in natural language! 🚀