import pandas as pd
import os
import sqlite3
from datetime import datetime

class TypeRacerDatalake:
    def __init__(self):
        self.datalake_path = 'typeracer_complete.csv'
        self.db_path = 'typeracer.db'
        
    def load_datalake(self):
        """Load existing datalake"""
        if os.path.exists(self.datalake_path):
            return pd.read_csv(self.datalake_path)
        return None
        
    def get_new_data(self):
        """Get new data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        new_data = pd.read_sql_query('''
            SELECT race_id as Item, speed as Speed, accuracy as Accuracy 
            FROM races
            ORDER BY race_id DESC
        ''', conn)
        conn.close()
        return new_data
        
    def update_datalake(self):
        """Update datalake with new data"""
        print("\nUpdating TypeRacer Datalake...")
        
        # Load current datalake
        current_data = self.load_datalake()
        if current_data is None:
            print("No existing datalake found. Creating new one.")
            current_data = pd.DataFrame(columns=['Item', 'Speed', 'Accuracy'])
        
        # Get new data
        new_data = self.get_new_data()
        
        # Get last race number in datalake
        if len(current_data) > 0:
            last_race = current_data['Item'].max()
            print(f"Last race in datalake: {last_race}")
            
            # Filter only new races
            new_data = new_data[new_data['Item'] > last_race]
        
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
            
            # Verify data consistency
            self.verify_datalake()
        else:
            print("\nNo new races to add. Datalake is up to date!")
    
    def verify_datalake(self):
        """Verify datalake consistency"""
        df = pd.read_csv(self.datalake_path)
        
        print("\nVerification Results:")
        # Check sequence
        expected_races = set(range(df['Item'].min(), df['Item'].max() + 1))
        actual_races = set(df['Item'])
        missing_races = sorted(expected_races - actual_races)
        
        if not missing_races:
            print("✓ Race sequence is complete")
        else:
            print(f"! Missing races: {missing_races}")
            
        # Check value ranges
        print(f"✓ Speed range: {df['Speed'].min():.2f} to {df['Speed'].max():.2f} WPM")
        print(f"✓ Accuracy range: {df['Accuracy'].min():.3f} to {df['Accuracy'].max():.3f}")

def main():
    datalake = TypeRacerDatalake()
    datalake.update_datalake()

if __name__ == "__main__":
    main() 