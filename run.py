import subprocess
import os
from datetime import datetime
from metis.update_datalake import TypeRacerDatalake
import pandas as pd

def format_status(status, message):
    """Format status messages consistently"""
    if status == "success":
        return f"✓ {message}"
    elif status == "info":
        return f"ℹ {message}"
    elif status == "warning":
        return f"⚠ {message}"
    else:
        return f"✗ {message}"

def run_pipeline():
    """Run the complete TypeRacer data pipeline"""
    
    print("\n🏎  TYPERACER DATA PIPELINE")
    print("=" * 50)
    print(format_status("info", f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    
    try:
        # Step 1: Fetch new data
        print("\n1️⃣  FETCH NEW DATA")
        print("-" * 20)
        fetch_result = subprocess.run(['python', 'web-extract.py'], 
                                    capture_output=True, 
                                    text=True, 
                                    check=True)
        
        # Extract key information from fetch output
        for line in fetch_result.stdout.split('\n'):
            if any(key in line for key in ['Added', 'No new races', 'Total Races']):
                print(format_status("success", line.strip()))
        
        # Step 2: Update datalake
        print("\n2️⃣  UPDATE DATALAKE")
        print("-" * 20)
        datalake = TypeRacerDatalake()
        datalake.update_datalake()
        
        # Final Status Report
        print("\n📊 PIPELINE SUMMARY")
        print("-" * 20)
        
        if os.path.exists('typeracer_complete.csv'):
            df = pd.read_csv('typeracer_complete.csv')
            recent_df = df.tail(10)  # Last 10 races
            
            print(format_status("success", "Pipeline completed successfully"))
            print("\nDatalake Status:")
            print(f"• Total races: {len(df):,}")
            print(f"• Race range: 1 to {df['Item'].max()}")
            print(f"• Average speed: {df['Speed'].mean():.2f} WPM")
            print(f"• Best speed: {df['Speed'].max():.2f} WPM")
            
            print("\nRecent Performance:")
            print(f"• Last 10 races avg: {recent_df['Speed'].mean():.2f} WPM")
            print(f"• Recent best: {recent_df['Speed'].max():.2f} WPM")
            
            # Progress indicators
            overall_avg = df['Speed'].mean()
            recent_avg = recent_df['Speed'].mean()
            if recent_avg > overall_avg:
                print(format_status("success", f"Recent average is {recent_avg - overall_avg:.2f} WPM above overall average"))
            
    except subprocess.CalledProcessError as e:
        print(format_status("error", f"Error in data fetch step: {e}"))
    except Exception as e:
        print(format_status("error", f"Error in pipeline: {e}"))
    finally:
        print("\n" + "=" * 50)
        print(format_status("info", f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))

if __name__ == "__main__":
    run_pipeline() 