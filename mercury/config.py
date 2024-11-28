import os

def load_config():
    # Get API key from environment
    api_key = os.getenv('MTA_API_KEY')
    
    if not api_key:
        raise ValueError("MTA_API_KEY environment variable not set. Please set it with: export MTA_API_KEY='your-api-key'")
    
    return {
        'api_key': api_key,
        'base_url': "https://bustime.mta.info/api/siri"
    } 