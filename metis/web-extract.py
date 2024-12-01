# web extraction of data points
import sqlite3
from datetime import datetime
import requests
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore', category=Warning)

def init_database(username):
    """Initialize SQLite database with username-specific file"""
    db_name = f'typeracer_{username}.db'
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    # Create races table with username field
    c.execute('''
        CREATE TABLE IF NOT EXISTS races (
            race_id INTEGER,
            username TEXT,
            speed REAL,
            accuracy REAL,
            position TEXT,
            race_date TIMESTAMP,
            fetch_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (race_id, username)
        )
    ''')
    
    # Update fetch_history table to include username
    c.execute('''
        CREATE TABLE IF NOT EXISTS fetch_history (
            id INTEGER PRIMARY KEY,
            username TEXT,
            last_fetch TIMESTAMP,
            races_added INTEGER
        )
    ''')
    
    conn.commit()
    return conn

def get_last_race_id(conn):
    """Get the most recent race_id from database"""
    c = conn.cursor()
    c.execute('SELECT MAX(race_id) FROM races')
    result = c.fetchone()[0]
    return result if result is not None else 0

def store_races(conn, races_data, username):
    """Store new races in database"""
    c = conn.cursor()
    new_races = 0
    
    for race in races_data:
        # Check if race already exists for this user
        c.execute('SELECT 1 FROM races WHERE race_id = ? AND username = ?', 
                 (race['Item'], username))
        if not c.fetchone():
            try:
                c.execute('''
                    INSERT INTO races (race_id, username, speed, accuracy, position, race_date)
                    VALUES (?, ?, ?, ?, ?, datetime(?, 'unixepoch'))
                ''', (
                    race['Item'],
                    username,
                    race['Speed'],
                    race['Accuracy'],
                    race['Position'],
                    race['Date']
                ))
                new_races += 1
            except sqlite3.Error as e:
                print(f"Error inserting race {race['Item']}: {e}")
    
    # Record fetch history with username
    if new_races > 0:
        c.execute('''
            INSERT INTO fetch_history (username, last_fetch, races_added)
            VALUES (?, CURRENT_TIMESTAMP, ?)
        ''', (username, new_races))
    
    conn.commit()
    return new_races

def fetch_data(username='barat_paim', n=50):
    """Fetch typing race history data from API"""
    all_races = []
    batch_size = 50  # API limit per request
    total_requested = n
    start_date = 0
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    while len(all_races) < total_requested:
        url = f'https://data.typeracer.com/games?playerId=tr:{username}&universe=play&startDate={start_date}&n={batch_size}'
        
        try:
            print(f"Fetching batch of races from API: {url}")
            response = requests.get(url, headers=headers, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                if not data:  # No more races to fetch
                    break
                
                # Debug: Print raw accuracy values from first batch
                if start_date == 0:
                    print("\nRaw accuracy values from last 5 races:")
                    for race in data[:5]:
                        print(f"Race {race.get('gn')}: Raw accuracy = {race.get('ac'):.5f}")
                
                # Process the batch
                for race in data:
                    try:
                        race_data = {
                            'Item': race.get('gn', 0),
                            'Speed': race.get('wpm', 0),
                            'Accuracy': race.get('ac', 0),
                            'Position': f"{race.get('r', 0)}/{race.get('n', 0)}",
                            'Date': race.get('t', 'unknown')
                        }
                        all_races.append(race_data)
                    except Exception as e:
                        print(f"Error parsing race data: {e}")
                        continue
                
                # Update start_date for next batch
                if data:
                    start_date = data[-1]['t']  # Use timestamp of last race
                
                print(f"Fetched {len(data)} races (Total: {len(all_races)})")
                
            else:
                print(f"Failed to fetch data. Status code: {response.status_code}")
                break
                
        except Exception as e:
            print(f"Error: {str(e)}")
            break
    
    print(f"\nFetched a total of {len(all_races)} races")
    return all_races

def get_stats(conn):
    """Get basic statistics from the database"""
    c = conn.cursor()
    
    # Get total races
    c.execute('SELECT COUNT(*) FROM races')
    total_races = c.fetchone()[0]
    
    # Get average speed
    c.execute('SELECT AVG(speed) FROM races')
    avg_speed = c.fetchone()[0]
    
    # Get date range
    c.execute('SELECT MIN(race_date), MAX(race_date) FROM races')
    date_range = c.fetchone()
    
    # Get fetch history
    c.execute('''
        SELECT last_fetch, races_added 
        FROM fetch_history 
        ORDER BY last_fetch DESC 
        LIMIT 5
    ''')
    recent_fetches = c.fetchall()
    
    return {
        'total_races': total_races,
        'avg_speed': round(avg_speed, 2) if avg_speed else 0,
        'date_range': date_range,
        'recent_fetches': recent_fetches
    }

def main():
    """Main function to fetch and store race data"""
    username = input("Enter TypeRacer username: ")
    try:
        n_races = int(input("Enter number of races to fetch: "))  # Removed max limit
        n_races = max(1, n_races)  # Only ensure it's positive
    except ValueError:
        print("Invalid input. Using default value of 50 races.")
        n_races = 50

    # Initialize new database for this user
    conn = init_database(username)
    
    try:
        races_data = fetch_data(username, n_races)
        if races_data:
            new_races = store_races(conn, races_data, username)
            
            if new_races > 0:
                print(f"\nDatabase Update Summary:")
                print(f"- Fetched {len(races_data)} races from API")
                print(f"- Added {new_races} new races to database")
                print(f"- Skipped {len(races_data) - new_races} existing races")
                print(f"- Action recorded in fetch_history table")
            else:
                print("\nNo new races to add - all races already in database")
            
            stats = get_stats(conn)
            print("\nDatabase Statistics:")
            print(f"Total Races: {stats['total_races']}")
            print(f"Average Speed: {stats['avg_speed']} WPM")
            print(f"Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
            print("\nRecent Fetches:")
            for fetch in stats['recent_fetches']:
                print(f"- {fetch[0]}: Added {fetch[1]} races")
            
            print(f"\nData stored in: {os.path.abspath(f'typeracer_{username}.db')}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()

