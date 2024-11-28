import requests
import os
import folium
from folium import plugins
from datetime import datetime
import urllib3
import ssl

# Disable SSL warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_all_bus_locations():
    api_key = os.getenv('MTA_API_KEY')
    if not api_key:
        print("Error: MTA_API_KEY environment variable not set")
        print("Please set it with: export MTA_API_KEY='your-api-key'")
        return []

    url = "https://bustime.mta.info/api/siri/vehicle-monitoring.json"
    params = {
        "key": api_key,
    }

    try:
        # Use verify=False to bypass SSL verification
        response = requests.get(url, params=params, verify=False)
        if response.status_code != 200:
            print(f"API Error: Status code {response.status_code}")
            return []
            
        data = response.json()
        vehicles = data['Siri']['ServiceDelivery']['VehicleMonitoringDelivery'][0]['VehicleActivity']
        
        locations = []
        for bus in vehicles:
            journey = bus['MonitoredVehicleJourney']
            location = journey['VehicleLocation']
            line = journey['LineRef'].split('_')[-1]
            locations.append({
                'coords': [location['Latitude'], location['Longitude']],
                'line': line,
                'weight': 1
            })
        
        return locations
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {str(e)}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {str(e)}")
        return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def create_heatmap():
    # Get bus locations
    bus_data = get_all_bus_locations()
    
    if not bus_data:
        print("No bus data received. Check your API key and connection.")
        return None
    
    print(f"Found {len(bus_data)} buses")
    
    # Create base map centered around midtown Manhattan
    m = folium.Map(
        location=[40.7589, -73.9851],
        zoom_start=12
    )
    
    # Add both markers and heatmap
    for bus in bus_data:
        coords = bus['coords']
        folium.CircleMarker(
            coords,
            radius=8,
            popup=f"Bus Line: {bus['line']}",
            color='red',
            fill=True
        ).add_to(m)
    
    # Add heatmap layer
    locations = [item['coords'] for item in bus_data]
    plugins.HeatMap(
        locations,
        min_opacity=0.4,
        radius=25,
        blur=15,
        max_zoom=16
    ).add_to(m)
    
    # Add legend
    legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 150px; height: 90px; 
                    border:2px solid grey; z-index:9999; 
                    background-color:white;
                    padding: 10px;
                    font-size: 14px;">
            <b>Bus Count</b><br>
            Total Buses: {len(bus_data)}<br>
            Last Updated:<br>
            {datetime.now().strftime('%I:%M:%S %p')}
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'mercury/heatmaps/all_buses_heatmap_{timestamp}.html'
    os.makedirs('mercury/heatmaps', exist_ok=True)
    m.save(filename)
    return filename

if __name__ == "__main__":
    filename = create_heatmap()
    if filename:
        print(f"Heatmap saved as: {filename}")
        print("Opening map in browser...")
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(filename)}") 