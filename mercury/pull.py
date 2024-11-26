import requests
import os
from datetime import datetime

def get_next_bus(stop_name, direction=None):
    url = "https://bustime.mta.info/api/siri/vehicle-monitoring.json"
    params = {
        "key": os.getenv('MTA_API_KEY'),
        'LineRef': 'M57'
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        vehicles = data['Siri']['ServiceDelivery']['VehicleMonitoringDelivery'][0]['VehicleActivity']
        
        next_arrival = None
        earliest_time = None

        for bus in vehicles:
            journey = bus['MonitoredVehicleJourney']
            if 'MonitoredCall' in journey:
                stop = journey['MonitoredCall']
                if stop_name.lower() in stop['StopPointName'].lower():
                    if direction:
                        dir_ref = "0" if direction.lower() == "eastbound" else "1"
                        if journey['DirectionRef'] != dir_ref:
                            continue
                    
                    arrival_time = datetime.fromisoformat(stop['ExpectedArrivalTime'].replace('Z', '+00:00'))
                    if earliest_time is None or arrival_time < earliest_time:
                        earliest_time = arrival_time
                        next_arrival = {
                            'stop': stop['StopPointName'],
                            'direction': "Eastbound" if journey['DirectionRef'] == '0' else "Westbound",
                            'arriving': stop['Extensions']['Distances']['PresentableDistance'],
                            'expected_time': arrival_time.strftime('%I:%M %p')
                        }

        return next_arrival or "No buses found for this stop"

    except Exception as e:
        return f"Error: {str(e)}"

# Test case
if __name__ == "__main__":
    print("\nNext bus at Madison Ave:")
    print(get_next_bus("MADISON"))

