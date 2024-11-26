import requests
import os

# Test the API
url = f"https://bustime.mta.info/api/siri/vehicle-monitoring.json"
params = {
    "key": os.getenv('MTA_API_KEY'),
    'LineRef': 'M57'
}

try:
    response = requests.get(url, params=params)
    print("API Response Status: ", response.status_code)
    print("API Response Data: ", response.json())
except Exception as e:
    print("Error: ", e)

