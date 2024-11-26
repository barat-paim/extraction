# TypeRacer Data Tracker - How to Use

## Setup
1. Clone repository
2. Install requirements:
```bash
pip install -r requirements.txt
```

## Quick Start
Run everything with one command:
```bash
python run.py
```

This will:
- Fetch new races
- Update your data
- Show current stats

## Data Files
- `typeracer_complete.csv`: All your race history
- `typeracer.db`: Temporary storage for new races

## Common Tasks

### 1. Check Latest Stats
```bash
python run.py
```
Shows:
- Total races
- Recent performance
- Speed averages

### 2. Manual Data Update
If you need to force a new data fetch:
```bash
python web-extract.py
python update_datalake.py
```

### 3. Data Location
All data is stored in:
- Main data: `typeracer_complete.csv`
- Recent races: `typeracer.db`

## Troubleshooting

1. No new data?
   - Check your internet connection
   - Verify you've raced since last update

2. Error running script?
   - Ensure all requirements are installed
   - Check you're in the correct directory

## Need Help?
Check `readme-typeracer.md` for detailed documentation 