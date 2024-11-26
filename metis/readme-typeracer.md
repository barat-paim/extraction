

Here's a detailed README.md documentation:

```markdown
# TypeRacer Data Extractor

A Python program to extract and store typing race history from TypeRacer.

## Development Journey

### 1. Initial Approach: HTML Scraping (Failed)
**Problem:**
- Needed to extract race history data from TypeRacer user profile
- Initially tried direct HTML scraping from `/pit/race_history?user={username}`

**Solution Attempted:**
```python
from bs4 import BeautifulSoup
import requests

url = f'https://data.typeracer.com/pit/race_history?user={username}'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
```

**Result:**
- Failed because page content was dynamically loaded via JavaScript
- BeautifulSoup only saw empty initial HTML structure
- No race data was accessible

### 2. Second Approach: Selenium (Partially Working but Overcomplicated)
**Problem:**
- Needed to handle JavaScript-rendered content
- Tried browser automation to let JavaScript execute

**Solution Attempted:**
```python
from selenium import webdriver
from selenium.webdriver.common.by import By
driver = webdriver.Chrome()
driver.get(url)
```

**Result:**
- Would have worked but was:
  - Slow (needed to start browser)
  - Complex (required ChromeDriver installation)
  - Resource-heavy
  - Unreliable across different environments

### 3. Final Solution: Direct API Access (Success)
**Problem:**
- Needed a simpler, more reliable way to get data
- Discovered actual data source through Chrome Developer Tools

**Solution:**
```python
url = f'https://data.typeracer.com/games?playerId=tr:{username}&universe=play&startDate=0&n={n}'
response = requests.get(url)
data = response.json()
```

**Result:**
- Fast and reliable direct data access
- Clean JSON format
- No browser dependencies
- Minimal resource usage

### 4. Data Storage Evolution

#### Initial Storage: CSV (Temporary)
**Problem:**
- Needed to store extracted race data
- Started with simple CSV storage

**Solution:**
```python
df = pd.DataFrame(data)
df.to_csv('typeracer_data.csv')
```

**Limitations:**
- No data persistence
- Duplicate data on each run
- No historical tracking

#### Final Storage: SQLite Database
**Problem:**
- Needed persistent storage
- Wanted to avoid duplicates
- Required historical tracking

**Solution:**
```python
def init_database():
    conn = sqlite3.connect('typeracer.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS races (
            race_id INTEGER PRIMARY KEY,
            speed REAL,
            accuracy REAL,
            position TEXT,
            race_date TIMESTAMP
        )
    ''')
```

**Benefits:**
- Persistent storage
- Duplicate prevention
- Easy querying
- Progress tracking

## Current Architecture

### 1. Data Extraction
- Uses direct API endpoint
- Fetches JSON-formatted race data
- Configurable number of races to fetch

### 2. Data Processing
- Parses JSON response
- Validates data structure
- Transforms into database format

### 3. Data Storage
- SQLite database with two tables:
  - `races`: Stores race history
  - `fetch_history`: Tracks data collection

### 4. Features
- Duplicate prevention
- Error handling
- Fetch history tracking
- Basic statistics

## Usage

1. Install dependencies:
```bash
pip install requests pandas sqlite3
```

2. Run the program:
```bash
python web-extract.py
```

3. View data:
```sql
sqlite3 typeracer.db
SELECT * FROM races ORDER BY race_date DESC LIMIT 10;
```

## Lessons Learned

1. **Web Scraping Strategy:**
   - Always check Network tab in Developer Tools
   - Look for API endpoints before scraping HTML
   - Consider how data is loaded (static vs dynamic)

2. **Solution Evolution:**
   - Start simple
   - Iterate based on challenges
   - Don't overcomplicate if simpler solution exists

3. **Data Management:**
   - Plan for data persistence
   - Consider duplicate handling
   - Track data collection history

## Future Improvements

1. Add user authentication for private profiles
2. Implement rate limiting
3. Add data visualization
4. Create automated scheduling
5. Add more statistical analysis

## Files

- `web-extract.py`: Main program file
- `typeracer.db`: SQLite database
- `.gitignore`: Git ignore rules
```

This documentation:
1. Shows the development journey
2. Explains what worked and what didn't
3. Provides code examples
4. Includes setup instructions
5. Lists lessons learned
6. Suggests future improvements


# Merge Historical Data
1. Run `merge_historical.py` to combine recent data with historical data.
2. This will create a new file `typeracer_complete.csv` with the merged data.

Problems:
- Accuracy values are stored as percentages in the database, but as decimals in the CSV.
- Need to rename columns to match the database schema.

Solutions:
- Multiply accuracy values by 100 in the CSV to match the database.
- Rename columns to match the database schema.

# Data Verification

```bash
python verify_data.py
```

Data Verification Report
==================================================

1. Basic Information:
Total races: 345
Race ID range: 1 to 345

2. Sequence Check:
✓ All race numbers are sequential (no gaps)

3. Value Ranges:
Speed range: 57.00 to 98.00 WPM
Accuracy range: 0.922 to 1.000

4. Potential Anomalies:
✓ No unusual speeds found
✓ No unusual accuracies found

5. Format Consistency:
Speed decimals: [1 2]
Accuracy decimals: [3 1 2]

6. Statistical Summary:
             Item       Speed    Accuracy
count  345.000000  345.000000  345.000000
mean   173.000000   75.711710    0.976867
std     99.737155    7.825852    0.011722
min      1.000000   57.000000    0.922000
25%     87.000000   70.000000    0.970000
50%    173.000000   76.000000    0.978000
75%    259.000000   81.000000    0.986000
max    345.000000   98.000000    1.000000



# Structure of the Datalake

typeracer/
├── run.py                  # Main script to run pipeline
├── web-extract.py         # Fetches new data from API
├── update_datalake.py     # Manages datalake updates
├── typeracer_complete.csv # Main data storage
├── typeracer.db          # Temporary new data storage
└── .gitignore            # Git ignore file

Current Files:
1. run.py                    # ✓ KEEP (main entry point)
2. web-extract.py           # ✓ KEEP (data fetching)
3. update_datalake.py       # ✓ KEEP (datalake management)
4. merge_historical.py      # ✗ REMOVE (one-time use, already merged)
5. verify_data.py           # ✗ REMOVE (verification now in update_datalake.py)
6. analyze_data.py          # ✗ REMOVE (analysis now in run.py)
7. typeracer_pipeline.py    # ✗ REMOVE (replaced by run.py)
8. merge_data.py            # ✗ REMOVE (one-time use)
9. populate_db.py           # ✗ REMOVE (one-time use)

Data Files:
1. typeracer_complete.csv   # ✓ KEEP (our datalake)
2. typeracer.db            # ✓ KEEP (temporary storage for new data)
3. typeracer_data.csv      # ✗ REMOVE (temporary file)
4. typeracer_merged.csv    # ✗ REMOVE (temporary file)