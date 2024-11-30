import pandas as pd

def fix_accuracy_format():
    print("Fixing accuracy format in typeracer_complete.csv...")
    
    # Read the CSV
    df = pd.read_csv('metis/typeracer_complete.csv')
    
    # Store original values for verification
    original_accuracies = df['Accuracy'].head()
    
    # Round accuracy to 3 decimal places
    df['Accuracy'] = df['Accuracy'].round(3)
    
    # Save back to CSV
    df.to_csv('metis/typeracer_complete.csv', index=False)
    
    # Print verification
    print("\nVerification of first few entries:")
    print("Original -> Updated")
    for orig, new in zip(original_accuracies, df['Accuracy'].head()):
        print(f"{orig} -> {new}")
    
    print("\n✓ Fixed accuracy format in CSV file")

if __name__ == "__main__":
    fix_accuracy_format() 