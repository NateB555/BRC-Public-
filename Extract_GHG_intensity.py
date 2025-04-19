import requests
import pandas as pd
import datetime
import time

# Define API endpoint format
API_URL = "https://www.eirgrid.ie/api/graph-data?area=co2intensity&region=ROI&date={}"

# Output CSV file
#!# Define the output CSV file name #!#
OUTPUT_FILE = "eirgrid_ghg_intensity_2023.csv"

# Function to fetch CO2 intensity data from API
def fetch_ghg_data(date):
    formatted_date = date.strftime(r"%d %b %Y")  # Format: "03 Jan 2023"
    url = API_URL.format(formatted_date)

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()  # Raise error if request fails
        data = response.json()

        if data["ErrorMessage"] is None and "Rows" in data:
            return [
                {
                    "DateTime": entry["EffectiveTime"],
                    "CO2 Intensity (gCO2/kWh)": entry["Value"]
                }
                for entry in data["Rows"]
            ]
        else:
            print(f"Warning: No data for {formatted_date}")
            return []
    
    except Exception as e:
        print(f"Error fetching data for {formatted_date}: {e}")
        return []

#!# Change year/ date range here if needed #!#
# Generate a list of dates for 2024
start_date = datetime.date(2023, 1, 1)
end_date = datetime.date(2024, 1, 1)
date_list = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]

# Store all data
all_data = []

for date in date_list:
    print(f"Fetching data for {date}...")
    day_data = fetch_ghg_data(date)
    
    if day_data:
        all_data.extend(day_data)  # Append daily data to main list

    time.sleep(1)  # Prevent overwhelming the server

# Convert to DataFrame and save to CSV
if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data saved to {OUTPUT_FILE}")
else:
    print("No data collected.")