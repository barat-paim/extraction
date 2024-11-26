import requests
import os

url = "https://bustime.mta.info/api/siri/vehicle-monitoring.json"
params = {
    "key": os.getenv('MTA_API_KEY'),
    'LineRef': 'M57'
}

try:
    response = requests.get(url, params=params)
    data = response.json()
    
    # Get vehicle activities
    vehicles = data['Siri']['ServiceDelivery']['VehicleMonitoringDelivery'][0]['VehicleActivity']
    
    print(f"\nFound {len(vehicles)} M57 buses:")
    for bus in vehicles:
        journey = bus['MonitoredVehicleJourney']
        location = journey['VehicleLocation']
        direction = "Eastbound" if journey['DirectionRef'] == '0' else "Westbound"
        
        print(f"\nBus {journey['VehicleRef']}:")
        print(f"Direction: {direction}")
        print(f"Location: {location['Latitude']}, {location['Longitude']}")
        
        if 'MonitoredCall' in journey:
            stop = journey['MonitoredCall']
            print(f"Next Stop: {stop['StopPointName']}")
            if 'EstimatedPassengerCount' in stop['Extensions']['Capacities']:
                print(f"Passengers: {stop['Extensions']['Capacities']['EstimatedPassengerCount']}/80")

except Exception as e:
    print("Error: ", e)

