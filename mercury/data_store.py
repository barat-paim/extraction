import sqlite3
import pandas as pd
from datetime import datetime
import requests
import os
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import load_config

class BusDataStore:
    def __init__(self, db_path='mercury/data/bus_data.db'):
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.config = load_config()
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
        # Configure requests session with retries
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def create_tables(self):
        with self.conn:
            # Real-time bus locations
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS bus_locations (
                    timestamp TEXT,
                    bus_line TEXT,
                    latitude REAL,
                    longitude REAL,
                    direction TEXT,
                    stop_name TEXT,
                    expected_arrival TEXT
                )
            ''')
            
            # Bus stops reference
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS bus_stops (
                    stop_id TEXT PRIMARY KEY,
                    stop_name TEXT,
                    latitude REAL,
                    longitude REAL
                )
            ''')
    
    def store_bus_data(self):
        start_time = datetime.now()
        url = f"{self.config['base_url']}/vehicle-monitoring.json"
        params = {"key": self.config['api_key']}
        
        try:
            # API call timing
            api_start = datetime.now()
            response = self.session.get(
                url, 
                params=params, 
                verify=False,
                timeout=30
            )
            response.raise_for_status()
            api_time = (datetime.now() - api_start).total_seconds()
            
            # Parse timing
            parse_start = datetime.now()
            data = response.json()
            if not data.get('Siri', {}).get('ServiceDelivery', {}).get('VehicleMonitoringDelivery'):
                raise ValueError("Invalid API response format")
            
            vehicles = data['Siri']['ServiceDelivery']['VehicleMonitoringDelivery'][0]['VehicleActivity']
            parse_time = (datetime.now() - parse_start).total_seconds()
            
            # DB operation timing
            db_start = datetime.now()
            current_time = datetime.now().isoformat()
            
            with self.conn:
                self.conn.execute('DELETE FROM bus_locations')
                
                for bus in vehicles:
                    journey = bus['MonitoredVehicleJourney']
                    location = journey['VehicleLocation']
                    
                    bus_data = {
                        'timestamp': current_time,
                        'bus_line': journey['LineRef'].split('_')[-1],
                        'latitude': location['Latitude'],
                        'longitude': location['Longitude'],
                        'direction': journey.get('DirectionRef', ''),
                        'stop_name': journey.get('MonitoredCall', {}).get('StopPointName', ''),
                        'expected_arrival': journey.get('MonitoredCall', {}).get('ExpectedArrivalTime', '')
                    }
                    
                    self.conn.execute('''
                        INSERT INTO bus_locations 
                        VALUES (:timestamp, :bus_line, :latitude, :longitude, 
                                :direction, :stop_name, :expected_arrival)
                    ''', bus_data)
            
            db_time = (datetime.now() - db_start).total_seconds()
            total_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\nTiming Breakdown:")
            print(f"API Call: {api_time:.2f}s")
            print(f"Parse JSON: {parse_time:.2f}s")
            print(f"DB Operations: {db_time:.2f}s")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Processed {len(vehicles)} buses\n")
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {str(e)}")
            return False
        except Exception as e:
            print(f"Error storing bus data: {str(e)}")
            return False
    
    def query_bus_location(self, bus_line=None, stop_name=None):
        query_start = datetime.now()
        
        query = '''
            SELECT * FROM bus_locations 
            WHERE timestamp = (SELECT MAX(timestamp) FROM bus_locations)
        '''
        
        if bus_line:
            query += f" AND bus_line = '{bus_line}'"
        if stop_name:
            query += f" AND stop_name LIKE '%{stop_name}%'"
        
        # Execute query and measure time
        try:
            df = pd.read_sql_query(query, self.conn)
            query_time = (datetime.now() - query_start).total_seconds()
            
            print(f"\nQuery Timing:")
            print(f"Query execution: {query_time:.2f}s")
            print(f"Results found: {len(df)} buses\n")
            
            return df
            
        except Exception as e:
            print(f"Query Error: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

if __name__ == "__main__":
    store = BusDataStore()
    store.store_bus_data()
    print(store.query_bus_location(bus_line='M57')) 