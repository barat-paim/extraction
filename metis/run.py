from datetime import datetime
from metis.update_datalake import TypeRacerDatalake
import pandas as pd
import subprocess
from pathlib import Path
from metis.analytics.datanalysis import main as analytics_main
from metis.analytics.datanalysis import show_stats

def show_stats():
    """Display basic statistics from the datalake"""
    try:
        df = pd.read_csv('metis/typeracer_complete.csv')
        recent_df = df.tail(10)  # Last 10 races
        
        print("\n📊 TYPERACER STATISTICS")
        print("-" * 20)
        print(f"Total races: {len(df):,}")
        print(f"Race range: 1 to {df['Item'].max()}")
        print(f"Average speed: {df['Speed'].mean():.2f} WPM")
        print(f"Best speed: {df['Speed'].max():.2f} WPM")
        
        print("\nRecent Performance:")
        print(f"Last 10 races avg: {recent_df['Speed'].mean():.2f} WPM")
        print(f"Recent best: {recent_df['Speed'].max():.2f} WPM")
        
    except Exception as e:
        print(f"Error showing stats: {e}")

def main():
    print("\n🏎  Welcome to Metis TypeRacer Analytics")
    print("=" * 39)
    
    while True:
        print("\n1. Update Data (fetch new races)")
        print("2. Show Statistics")
        print("3. Run Analytics")
        print("4. Exit")
        print("-" * 39)
        
        choice = input("Choose an option (1-4): ")
        
        if choice == "1":
            datalake = TypeRacerDatalake()
            datalake.update_datalake()
            
        elif choice == "2":
            show_stats()
            
        elif choice == "3":
            print("\n🔍 Running Analytics...")
            subprocess.run(['python', 'metis/analytics/datanalysis.py'])
            print("✓ Analytics complete")

        elif choice == "4":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option. Please choose 1-4.")

if __name__ == "__main__":
    main() 