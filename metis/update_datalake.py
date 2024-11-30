import pandas as pd
import os
import sqlite3
from datetime import datetime
import subprocess
from pathlib import Path

class TypeRacerDatalake:
    def __init__(self):
        self.datalake_path = 'metis/typeracer_complete.csv'
        self.db_path = 'typeracer.db'
        
    def update_datalake(self):
        """Update datalake with new data"""
        print("\nUpdating TypeRacer Datalake...")
        
        # Step 1: Run web-extract.py to fetch new races into SQLite
        try:
            extract_script = Path('metis/web-extract.py')
            result = subprocess.run(
                ['python', str(extract_script)],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running web-extract: {e}")
            return False
            
        # Step 2: Load current datalake
        current_data = pd.read_csv(self.datalake_path) if os.path.exists(self.datalake_path) else pd.DataFrame(columns=['Item', 'Speed', 'Accuracy'])
        
        # Step 3: Get new data from SQLite
        conn = sqlite3.connect(self.db_path)
        new_data = pd.read_sql_query('''
            SELECT race_id as Item, 
                   speed as Speed, 
                   ROUND(accuracy, 3) as Accuracy  -- Round to 3 decimal places in SQL
            FROM races
            ORDER BY race_id DESC
        ''', conn)
        conn.close()
        
        # Format accuracy to match existing data (3 decimal places)
        new_data['Accuracy'] = new_data['Accuracy'].round(3)
        
        # Step 4: Filter only new races
        if len(current_data) > 0:
            last_race = current_data['Item'].max()
            print(f"Last race in datalake: {last_race}")
            new_data = new_data[new_data['Item'] > last_race]
        
        # Step 5: Update the datalake if there are new races
        if len(new_data) > 0:
            print(f"\nFound {len(new_data)} new races:")
            print(f"Race numbers {new_data['Item'].min()} to {new_data['Item'].max()}")
            
            # Append new data
            updated_data = pd.concat([current_data, new_data], ignore_index=True)
            updated_data = updated_data.sort_values('Item').reset_index(drop=True)
            
            # Save updated datalake
            updated_data.to_csv(self.datalake_path, index=False)
            
            print("\nDatalake Update Summary:")
            print(f"Previous size: {len(current_data)} races")
            print(f"New size: {len(updated_data)} races")
            print(f"Added: {len(new_data)} races")
            print(f"Race range: 1 to {updated_data['Item'].max()}")
        else:
            print("\nNo new races to add. Datalake is up to date!")

def main():
    datalake = TypeRacerDatalake()
    datalake.update_datalake()

if __name__ == "__main__":
    main() 