from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc

import base64
import datetime
import io

import pvlib
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance

import pandas as pd

# Importing Electricity Plans
class Elec_Plan_parameters:
    def __init__(self, provider, plan_name, rate_type, rates, extras, website_link, notes):
        self.provider = provider
        self.plan_name = plan_name
        self.rate_type = rate_type
        self.rates = rates
        self.extras = extras
        self.website_link = website_link
        self.notes = notes

Elec_plans = [
    Elec_Plan_parameters(
        provider="Electric Ireland",
        plan_name="Home Electric+ Night Boost",
        rate_type= "Day Night Night Boost",
        rates = {
            "day": {"rate": 0.3634, "hours": (8, 23), "priority": 2},   # priority indicates whether another rate during this rate range will supercede it
            "night_boost": {"rate": 0.1052, "hours": (2, 4), "priority": 1},
            "night": {"rate": 0.1792, "hours": (23, 8), "priority": 2}
        },
        extras = {"urban": 328.58, "rural": 400.48, "1yr_discount": 0.16, "welcome_bonus": 0, "PSO_levy": 42.25},
        website_link='https://www.electricireland.ie/switch/new-customer/price-plans?priceType=E',
        notes=f'''price type E. 16% discount implemented'''
    ),
    # EV and Microgen
    Elec_Plan_parameters(
        provider="Bord Gais",
        plan_name="EV Smart Electricity Discount with €50 Welcome Bonus",
        rate_type="Day Night Peak EV",
        rates = {
            "peak": {"rate": 0.3464, "hours": (17, 19), "priority": 1},
            "day": {"rate": 0.2484, "hours": (8, 23), "priority": 2},
            "EV": {"rate": 0.0633, "hours": (2, 5), "priority": 1},
            "night": {"rate": 0.1873, "hours": (23, 8), "priority": 2},
    },
        extras = {"urban": 325.52, "rural": 325.52, "1yr_discount": 0.32, "welcome_bonus": 50, "PSO_levy": 42.25},
        website_link='https://www.bordgaisenergy.ie/home/our-plans/a0pSh000000DPAHIA4',
        notes= f'''32% discount implemented. Standing Charge: €325.52.'''
    ),
    Elec_Plan_parameters(
        provider="Bord Gais",
        plan_name="Green EV Smart Electricity Discount with €50 Welcome Bonus",
        rate_type="Day Night Peak EV",
        rates = {
            "peak": {"rate": 0.3616, "hours": (17, 19), "priority": 1},
            "day": {"rate": 0.2593, "hours": (8, 23), "priority": 2},
            "EV": {"rate": 0.0661, "hours": (2, 5), "priority": 1},
            "night": {"rate": 0.1956, "hours": (23, 8), "priority": 2},
        },
        extras = {"urban": 325.52, "rural": 325.52, "1yr_discount": 0.29, "welcome_bonus": 50, "PSO_levy": 42.25},
        website_link='https://www.bordgaisenergy.ie/home/our-plans/a0pSh0000009TkRIAU',
        notes= f'''32% discount implemented. Standing Charge: €325.52.'''
    ),
    Elec_Plan_parameters(
        provider="energia",
        plan_name="Smart Data & 33% Discount",
        rate_type="Day Night Peak EV",
        rates = {
            "peak": {"rate": 0.2642, "hours": (17, 19), "priority": 1}, # assumed peak hours
            "day": {"rate": 0.2521, "hours": (8, 23), "priority": 2},
            "night": {"rate": 0.1349, "hours": (23, 8), "priority": 2}
        },
        extras = {"urban": 236.62, "rural": 300.91, "1yr_discount": 0.33, "welcome_bonus": 0, "PSO_levy": 42.25},
        website_link='https://switchto.energia.ie/MyPlanDetails',
        notes=''''''
    ),
    Elec_Plan_parameters(
        provider="energia",
        plan_name="EV Smart Drive Plus",
        rate_type="Day Night Peak",
        rates = {
            "peak": {"rate": 0.4021, "hours": (17, 19), "priority": 1}, # assumed peak hours
            "day": {"rate": 0.3677, "hours": (8, 23), "priority": 2},
            "EV": {"rate": 0.0883, "hours": (2, 6), "priority": 1},
            "night": {"rate": 0.1969, "hours": (23, 8), "priority": 2}
        },
        extras = {"urban": 236.62, "rural": 300.91, "1yr_discount": 0.15, "welcome_bonus": 0, "PSO_levy": 42.25},
        website_link='https://switchto.energia.ie/MyPlanDetails',
        notes='''Standing Charge not specified'''
    ),
    Elec_Plan_parameters(
        provider="energia",
        plan_name="EV Smart Drive & €50",
        rate_type="Day EV",
        rates = {
            "day": {"rate": 0.3386, "hours": (6, 2), "priority": 2},
            "EV": {"rate": 0.1349, "hours": (2, 6), "priority": 1}
        },
        extras = {"urban": 236.62, "rural": 300.91, "1yr_discount": 0.15, "welcome_bonus": 50, "PSO_levy": 42.25},
        website_link='https://switchto.energia.ie/MyPlanDetails',
        notes=f'''Standing Charge not specified. 100% renewable energy claimed (this claim is achieved using carbon credits => N/A)'''
    ),
]

# County List:
irish_counties = {
    "Carlow": {"latitude": 52.6680, "longitude": -6.9284, "altitude": 90},
    "Cavan": {"latitude": 53.9890, "longitude": -7.3606, "altitude": 100},
    "Clare": {"latitude": 52.9469, "longitude": -8.9402, "altitude": 50},
    "Cork": {"latitude": 51.8985, "longitude": -8.4756, "altitude": 10},
    "Dublin": {"latitude": 53.3498, "longitude": -6.2603, "altitude": 20},
    "Galway": {"latitude": 53.2707, "longitude": -9.0568, "altitude": 5},
    "Kerry": {"latitude": 52.2663, "longitude": -9.5605, "altitude": 10},
    "Kildare": {"latitude": 53.0241, "longitude": -6.9115, "altitude": 120},
    "Kilkenny": {"latitude": 52.6580, "longitude": -7.2516, "altitude": 80},
    "Laois": {"latitude": 53.0039, "longitude": -7.3159, "altitude": 70},
    "Leitrim": {"latitude": 54.1061, "longitude": -8.0731, "altitude": 50},
    "Limerick": {"latitude": 52.6638, "longitude": -8.6267, "altitude": 10},
    "Louth": {"latitude": 53.7265, "longitude": -6.3796, "altitude": 30},
    "Mayo": {"latitude": 53.8000, "longitude": -9.2000, "altitude": 100},
    "Meath": {"latitude": 53.6439, "longitude": -6.5211, "altitude": 70},
    "Monaghan": {"latitude": 54.2506, "longitude": -6.9524, "altitude": 50},
    "Offaly": {"latitude": 53.3249, "longitude": -7.6122, "altitude": 90},
    "Roscommon": {"latitude": 53.6342, "longitude": -8.1900, "altitude": 80},
    "Sligo": {"latitude": 54.2769, "longitude": -8.4760, "altitude": 60},
    "Tipperary": {"latitude": 52.7100, "longitude": -7.8283, "altitude": 80},
    "Waterford": {"latitude": 52.2580, "longitude": -7.1107, "altitude": 20},
    "Westmeath": {"latitude": 53.4020, "longitude": -7.3352, "altitude": 90},
    "Wexford": {"latitude": 52.3500, "longitude": -6.4583, "altitude": 40},
    "Wicklow": {"latitude": 52.9884, "longitude": -6.3733, "altitude": 120}
}
county_dropdown_options = [{"label": county, "value": county} for county in irish_counties.keys()]

# Functions:

def reformat_hdf(hdf):
    '''Reformats a HDF file to a single-column dataframe'''
    # Extracting the read value (kWh consumption) column and removing NaN values
    hdf = hdf[["Read Value"]]
    hdf = hdf*0.5  # converting to kWh
    hdf = hdf.dropna()
    # Ensuring index is in datetime format and sorted in ascending order (oldest to newest)
    hdf.index = pd.to_datetime(hdf.index, format=r'%d-%m-%Y %H:%M')
    hdf = hdf.sort_index()
    
    return hdf

def process_to_single_year(df):
    '''Processes consumption df of minimum a year of data into a single
    representative year (here taking 2023 as the representative year)'''
    
    # removing leap year data
    df = df[~((df.index.month == 2) & (df.index.day == 29))].copy()
    
    # extracting month/day/HH from index
    df.loc[:,"Month"] = df.index.month
    df.loc[:,"Day"] = df.index.day
    df.loc[:,"Half_Hour"] = df.index.strftime('%H:%M')
    
    # grouping by month/day/HH, then averaging values 
    averaged_df = (
        df.groupby(['Month', 'Day', 'Half_Hour'])['Read Value']
        .mean()
        .reset_index()
    )
    # compiling averaged data to a dataframe
    averaged_df['Datetime'] = pd.to_datetime(
        '2023-' + # assuming 2023 as the representative year (to be changed)
        averaged_df['Month'].astype(str) + '-' + 
        averaged_df['Day'].astype(str) + ' ' + 
        averaged_df['Half_Hour']
    )
    averaged_df.set_index('Datetime', inplace=True)
    
    # dropping month/day/HH columns
    df.drop(['Month', 'Day', 'Half_Hour'], axis=1, inplace=True)
    averaged_df.drop(['Month', 'Day', 'Half_Hour'], axis=1, inplace=True)
    
    return averaged_df

def link_time_to_rate(time, plan, no_discount=False):
    '''Links a time to corresponding rate in an electricity plan'''
    if isinstance(time, str): hour = int(time.split(":")[0])
    elif isinstance(time, datetime.time): hour = time.hour
    
    if no_discount == True: one_year_discount = 1.0 - plan.extras["1yr_discount"]
    else: one_year_discount = 1.0
    
    for rate_category, details in plan.rates.items():
        
        start, end = details["hours"]
        rate_priority = details.get("priority", 2)

        if rate_priority == 1:
            if start < end:   # normal (during the same day)
                if start <= hour < end: return details["rate"]/one_year_discount
            else:
                if hour >= start or hour < end: return details["rate"]/one_year_discount
                
        elif rate_priority == 2:
            if start < end:   # normal (during the same day)
                if start <= hour < end: return details["rate"]/one_year_discount
            else:
                if hour >= start or hour < end: return details["rate"]/one_year_discount

    return None

def apply_plan_to_pivot(pivot, plan, no_discount=False):
    '''Applies a plan to a pivot table of electricity consumption'''
    HH_cost_pivot = pivot.copy()
    
    if no_discount == False:
        for time in HH_cost_pivot.columns:
            rate=link_time_to_rate(time, plan)
            if rate: HH_cost_pivot[time] *= rate
    else:
        for time in HH_cost_pivot.columns:
            rate=link_time_to_rate(time, plan, no_discount=True)
            if rate: HH_cost_pivot[time] *= rate
    
    HH_cost_pivot["Total Cost"] = HH_cost_pivot.sum(axis=1)
    return HH_cost_pivot

def convert_to_pivot(consumption_df):
    '''Converts a SINGLE-COLUMN consumption dataframe to a pivot table'''
    column_name = consumption_df.columns[0]
    
    HDF_Pivot = consumption_df.pivot_table(
        values=column_name,
        index=[consumption_df.index.month,consumption_df.index.day],
        columns=consumption_df.index.strftime('%H:%M') # removing seconds from column names
    )
    HDF_Pivot.index.set_names(['Month', 'Day'], inplace=True)
    return HDF_Pivot

def calculate_costs(consumption_pivot, Elec_plans, grid_export=0.0, urban=True, microgen_rate=0.195,):
    '''calculates the annual cost of each electricity plan. microgen rate of 0.195 €/kWh assumed'''
    plan_options = []
    for plan in Elec_plans:
        #annual cost for 1st year
        cost_pivot_1st_year_discount = apply_plan_to_pivot(consumption_pivot, plan)
        annual_cost_1st_year_discount = round(cost_pivot_1st_year_discount["Total Cost"].sum(), 2)
        annual_cost_1st_year_discount -= grid_export*microgen_rate
        
        #annual cost for subsequent years
        cost_pivot_no_discount = apply_plan_to_pivot(consumption_pivot, plan, no_discount=True)
        annual_cost_no_discount = round(cost_pivot_no_discount["Total Cost"].sum(), 2)
        annual_cost_no_discount -= grid_export*microgen_rate
        
        #adding extra costs:
        extra_cost = 0.0
        standing_charge = plan.extras["urban"] if urban else plan.extras["rural"]
        extra_cost += standing_charge
        welcome_bonus = plan.extras.get("welcome_bonus", 0)
        extra_cost -= welcome_bonus
        PSO_levy = plan.extras["PSO_levy"]
        extra_cost += PSO_levy
        
        annual_cost_1st_year_discount += extra_cost
        annual_cost_no_discount += PSO_levy #only annual extra cost
        
        plan_options.append({
            "provider": plan.provider,
            "plan_name": plan.plan_name,
            "annual_cost_1st_year_discount": annual_cost_1st_year_discount,
            "annual_cost_no_discount": annual_cost_no_discount,
            "website_link": plan.website_link,
            "grid_export": grid_export,
            "grid_export_profit": grid_export*microgen_rate,
        })
    # Sorting the plan options by annual cost
    plan_options.sort(key=lambda x: x["annual_cost_1st_year_discount"])
    return plan_options

def parse_contents(contents, filename, date):
    '''Converts 64 bit encoded file to a dataframe'''
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            df = df.head(10)    
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),
        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

def get_solar_data(county_name, start_date, end_date, tilt, azimuth):
    '''Returns solar data for a given county and time range'''
    if county_name not in irish_counties:
        raise ValueError(f"County '{county_name}' not found. Please check the name.")

    # Get latitude, longitude, and altitude from the dictionary
    county_info = irish_counties[county_name]
    latitude = county_info["latitude"]
    longitude = county_info["longitude"]
    altitude = county_info.get("altitude", 0)  # Default altitude to 0 if not provided

    # Define timezone
    timezone = "Europe/Dublin"

    # Create a Location object
    site = Location(latitude, longitude, timezone, altitude)

    # Define the time range
    times = pd.date_range(
        start=start_date,
        end= pd.to_datetime(end_date) - datetime.timedelta(minutes=30),
        freq=datetime.timedelta(minutes=30),
        tz=timezone
    )

    # Get solar position and clear-sky data
    solar_position = site.get_solarposition(times)
    clearsky = site.get_clearsky(times, model="ineichen")

    # Calculate Plane-of-Array (POA) irradiance
    poa = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    )
    # Extract only the relevant columns: GHI, DNI, DHI, POA Irradiance
    results = pd.DataFrame({
        'GHI': clearsky['ghi'],  # Global Horizontal Irradiance
        'DNI': clearsky['dni'],  # Direct Normal Irradiance
        'DHI': clearsky['dhi'],  # Diffuse Horizontal Irradiance
        'POA Irradiance (W/m²)': poa['poa_global']
    }, index=times)
    
    results.index = results.index.tz_convert(None)

    return results

def calculate_solar_power_output(solar_data_df, pv_capacity, performance_ratio):
    '''Calculate the power and energy output in kW and kWh'''
    
    performance_ratio = performance_ratio / 100  # Convert percentage to decimal  
    
    power_output = solar_data_df['POA Irradiance (W/m²)'] * pv_capacity * performance_ratio / 1000
    energy_output = power_output * 0.5  # 0.5 hours (30 minutes) interval
    
    # compiling to single dataframe
    array_output = pd.DataFrame({
        "Power Output (kW)": power_output,
        "Energy Output (kWh)": energy_output
    }, index=solar_data_df.index)
    
    return array_output

def combine_consumption_and_solar(consumption_df, solar_power_output_df):
    hdf_and_pv = pd.DataFrame({
    "Consumption [kWh]": consumption_df["Read Value"],
    "Solar Generation [kWh]": solar_power_output_df["Energy Output (kWh)"]
    })
    consumption_with_solar = pd.DataFrame(
        data= (hdf_and_pv["Consumption [kWh]"] - hdf_and_pv["Solar Generation [kWh]"]),
        index=hdf_and_pv.index,
        columns=["Grid Import"],    
    )
    
    return consumption_with_solar

def calculate_total_overconsumption(consumption_with_solar_df):
    '''Calculates the amount of overconsumption in kWh'''
    if consumption_with_solar_df is not None:
        if not consumption_with_solar_df.empty:
            overconsumption = 0.0
            for value in consumption_with_solar_df["Grid Import"]:
                if value < 0:
                    overconsumption += value
            return overconsumption

    else: return 0.0
  
def remove_overconsumption(consumption_with_solar_df):
    '''Removes overconsumption from the consumption data'''
    if consumption_with_solar_df is not None:
        if not consumption_with_solar_df.empty:
            consumption_with_solar_df.loc[:,"Grid Import"] = consumption_with_solar_df["Grid Import"].apply(lambda x: max(x, 0))
    return consumption_with_solar_df


def format_elec_cost_output(plan_options):
    '''Formats the electricity plan options for display'''
    # Create table header
    table_header = html.Tr([
        html.Th("Provider", style={"text-align": "left", "font-weight": "bold"}),
        html.Th("Plan Name", style={"text-align": "left", "font-weight": "bold"}),
        html.Th("1st Year Cost (€)", style={"text-align": "left", "font-weight": "bold"}),
        html.Th("No Discount Cost (€)", style={"text-align": "left", "font-weight": "bold"}),
        html.Th("Website Link", style={"text-align": "left", "font-weight": "bold"})
    ])

    # Create table rows for the first 3 plans
    table_rows = [
        html.Tr([
            html.Td(result['provider']),
            html.Td(result['plan_name']),
            html.Td(f"€{result['annual_cost_1st_year_discount']:,.2f}"),
            html.Td(f"€{result['annual_cost_no_discount']:,.2f}"),
            html.Td(html.A("Link", href=result['website_link'], target="_blank", style={"color": "blue", "text-decoration": "underline"}))
        ]) for result in plan_options[:3]  # Only show the first 3 plans
    ]

    # Return the table and the grid export sentence
    return html.Div([
        html.Table(
            [table_header] + table_rows,
            style={"width": "100%", "border-collapse": "collapse", "margin-bottom": "20px"}
        ),
        html.Div([
            f"{plan_options[0]['grid_export']:,.2f} kWh were exported to the grid, reducing costs by €{plan_options[0]['grid_export_profit']:,.2f}"
        ])
    ])


#! GHG-related functions:
#!!!! Integrate your own eirgrid intensity data using the Extract_GHG_intensity.py file #!!!!

GHG_intensities_2023 = pd.read_csv(r"C:\Users\__path__\eirgrid_ghg_intensity_2023.csv")
GHG_intensities_2024 = pd.read_csv(r"C:\Users\__path__\eirgrid_ghg_intensity_2024.csv")

def average_ghg_intensity(ghg_intensity_df):
    """
    Averages the quarter-hourly GHG intensity data into half-hourly data.
    Input: ghg_intensity_df: DataFrame with quarter-hourly GHG intensity data.
    Returns: half_hourly_ghg_intensity: DataFrame with half-hourly GHG intensity data.
    """
    # Ensure the 'DateTime' column is in datetime format
    ghg_intensity_df['DateTime'] = pd.to_datetime(ghg_intensity_df['DateTime'])
    ghg_intensity_df.set_index('DateTime', inplace=True)
    # Resample to half-hourly intervals by averaging
    half_hourly_ghg_intensity = ghg_intensity_df.resample('30min').mean()
    
    return half_hourly_ghg_intensity

GHG_intensities_2023 = average_ghg_intensity(GHG_intensities_2023)
GHG_intensities_2024 = average_ghg_intensity(GHG_intensities_2024)

def CO2_Balance(grid_import, Carbon_intensity, grid_export=None):
    if grid_export is None:
        E_CO2 = (grid_import*Carbon_intensity)
    else:
        E_CO2 = ((grid_import - grid_export)*Carbon_intensity)
    return E_CO2

def calculate_ghg_emissions(import_export_df, ghg_intensity_df):
    '''Calculates the GHG emissions from grid-electricity consumption
    Input:  - consumption_df: DataFrame with ONLY electricity consumption data, column name "Grid Import".
            - ghg_intensity_df: DataFrame with half-hourly GHG intensity data.
    Output: half_hourly_ghg_emissions: DataFrame with half-hourly GHG emissions data.
    '''
    
    if "Export" in import_export_df.columns:
        gCO2_emissions = CO2_Balance(import_export_df['Grid Import'],
                                            ghg_intensity_df['CO2 Intensity (gCO2/kWh)'],
                                            import_export_df['Export'])
    else:
        gCO2_emissions = CO2_Balance(import_export_df['Grid Import'],
                                            ghg_intensity_df['CO2 Intensity (gCO2/kWh)'],
                                            None)
    
    half_hourly_ghg_emissions = pd.DataFrame(
        index=import_export_df.index,
        data={'gCO2 emissions': gCO2_emissions}
    )
    return half_hourly_ghg_emissions
    

#! Battery-related functions:

# Max charge/discharge for battery capacities. Assumptions taken from kilowatt.ie
charge_discharge_rates = {
    "1": 1.0,  # 1 kW
    "2": 2.0,  # 2 kW
    "3": 3.0, "4": 3.0, "5": 3.0, "6": 3.0,  # 3 kW
    "7": 4.0, "8": 4.0, "9": 4.0, "10": 4.0,  # 4 kW
    "11": 5.0, "12": 5.0, "13": 5.0, "14": 5.0, "15": 5.0, "16": 5.0, # 5 kW
}

def get_charge_discharge_rate(battery_capacity):
    '''Returns the charge/discharge rate for a given battery capacity in usable format'''

    return charge_discharge_rates.get(str(battery_capacity), 4.0)  # Default to 4.0 kW if not found

# Battery charge/discharge functions. Created for simplicity and readability
def batterychargeable(SOC_kWh, charge_threshold_kWh):
    '''Determines whether the battery can be charged'''
    if SOC_kWh < charge_threshold_kWh:
        return True
    else: return False

def batterydischargeable(SOC_kWh, discharge_threshold_kWh):
    '''Determines whether the battery can be discharged'''
    if SOC_kWh > discharge_threshold_kWh:
        return True
    else: return False
    
def det_charge_rate(start, end, charge_threshold_kWh, discharge_threshold_kWh, SOC_kWh, efficiency=0.85):
    '''Determines the minimum charge rate given number of hours available'''

    #determine number of hours available for charging
    if start < end: charge_hours = end - start
    else: charge_hours = 24 - start + end

    charge_half_hours = charge_hours * 2.0 # convert to HH
    # determine amount of kWhs to charge:
    charge_amount_kWh = min(charge_threshold_kWh - SOC_kWh, charge_threshold_kWh - discharge_threshold_kWh)
    # determine minimum charge rate:
    min_charge_rate = (charge_amount_kWh/efficiency) / charge_half_hours
    
    return min_charge_rate

def get_rate(hour, elec_rates):
    '''extracts the rate for a given hour from the electricity rates dictionary'''
    peak_start, peak_end = elec_rates["peak"]["hours"]
    day_start, day_end = elec_rates["day"]["hours"]
    boost_start, boost_end = elec_rates["EV"]["hours"]
    night_start, night_end = elec_rates["night"]["hours"]

    if peak_start <= hour < peak_end:
        return elec_rates["peak"]["rate"]
    elif day_start <= hour < day_end:
        return elec_rates["day"]["rate"]
    elif boost_start <= hour < boost_end:
        return elec_rates["EV"]["rate"]
    elif (night_start <= hour <= 0) or (0 <= hour <= night_end):
        return elec_rates["night"]["rate"]
    else:
        return elec_rates["day"]["rate"]  # Default to day rate if no match

def G_2_B(SOC_kWh, charge_threshold_kWh, max_charge_discharge_rate, slow_charge_on, slow_charge_rate, 
          battery_efficiency=0.9):
    '''During night hours, the battery is charged. This function returns additional grid consumption and cost'''
    
    if slow_charge_on == True: charge_rate = slow_charge_rate
    else: charge_rate = max_charge_discharge_rate
    if batterychargeable(SOC_kWh, charge_threshold_kWh) == True:
        charge_amount_kWh = min(charge_rate*battery_efficiency,
                                charge_threshold_kWh - SOC_kWh)
    else: charge_amount_kWh = 0.0
    
    return charge_amount_kWh

def G_2_H(consumption, B2H, S2H):
    '''Grid to home. Tests for other sources of consumption change then returns new consumption and cost'''

    G2H = consumption
    if B2H: G2H -= B2H
    if S2H: G2H -= min(S2H, consumption)
    
    return G2H

def S_2_H(consumption, solar_gen, max_inverter_ac_power):
    '''Returns by how much solar generation should be used to reduce consumption'''
    S2H = min(solar_gen,
              consumption,
              max_inverter_ac_power*0.5)
    
    return S2H
    
def B_2_H(consumption, SOC_kWh, S2H, discharge_threshold_kWh, max_charge_discharge_rate,
          battery_efficiency=0.9):
    '''Uses battery charge to reduce peak consumption. returns by how much the battery should be discharged'''
    
    if batterydischargeable(SOC_kWh, discharge_threshold_kWh) == True:
        discharge_amount_kWh = min(max_charge_discharge_rate*battery_efficiency,
                                   SOC_kWh - discharge_threshold_kWh,
                                   consumption,
                                   consumption-S2H  # ensure battery doesn't discharge more than needed
                                   )
    else: discharge_amount_kWh = 0.0
    
    return discharge_amount_kWh



def S_2_B(consumption, solar_gen, SOC_kWh, charge_threshold_kWh, max_charge_discharge_rate, 
          efficiency=0.9):
    '''Returns by how much the battery should be charged from excess solar'''    
    
    if batterychargeable(SOC_kWh, charge_threshold_kWh) == True:
        charge_amount_kWh = min(max_charge_discharge_rate*efficiency,
                                abs(consumption-solar_gen)*efficiency,
                                charge_threshold_kWh - SOC_kWh
                                )
    else: charge_amount_kWh = 0.0
    
    return charge_amount_kWh

def S_2_G(consumption, solar_gen, max_inverter_ac_power, 
          battery_full=True, S2B=0.0):
    '''Returns by how much excess solar should be exported to the grid'''
    
    if battery_full == True:
        if consumption-solar_gen < 0: 
            S2G = min(
                abs(consumption-solar_gen),
                max_inverter_ac_power*0.5
            )
        else: S2G = 0.0
    if battery_full == False:
        S2G = min(
            abs(consumption-(solar_gen-S2B)),
            max_inverter_ac_power*0.5
            )
        
    return S2G

def battery_algorithm(consumption_df, solar_generation_df,
                        battery_capacity, charge_threshold, discharge_threshold,
                        elec_rates, 
                        battery_efficiency=0.9, 
                        microgen_on=True,
                        morning_discharge_on=True,
                        ):
    '''Battery charge/discharge algorithm.
    Functions on a pre-set times to charge and discharge, (also charges using excess solar):
    - Night hours: charge (slow charge rate set to ensure full charge at minimum charge speed)
    - Morning hours: discharge
    - Excess solar: charge battery or export to grid
    - Peak hours: discharge
    '''

    #handling charge/discharge thresholds and setting initial values
    charge_threshold_percentage = charge_threshold/100.0
    discharge_threshold_percentage = discharge_threshold/100.0
    SOC_percentage = discharge_threshold_percentage # state-of-charge of the battery. taken as minimum for now
    SOC_kWh = SOC_percentage*battery_capacity
    next_SOC_kWh = SOC_kWh

    #initialising arrays to store energy flows
    battery_SOC = []
    discharge_to_home = []
    charge_from_grid = []; charge_from_solar = []
    solar_to_home = []; solar_to_grid = []
    grid_to_home = []; grid_imports = []
    
    if not solar_generation_df.empty: consumption_df.loc[:,"Solar"] = solar_generation_df["Energy Output (kWh)"]
    else: consumption_df.loc[:,"Solar"] = 0.0
    
    charge_discharge_rate = get_charge_discharge_rate(battery_capacity)
    # converting percentages into kWh values
    charge_threshold_kWh = charge_threshold_percentage*battery_capacity
    discharge_threshold_kWh = discharge_threshold_percentage*battery_capacity
    
    max_charge_discharge_rate = 0.5 * charge_discharge_rate
    max_inverter_ac_power = 5.0  # 5 kW taken as max - refers to rate at which solar can transfer energy
    morning_discharge_done_flag = False
    Slow_Charge_Rate_set_flag = False
    
    night_charge_rate = det_charge_rate(0, 5, charge_threshold_kWh, discharge_threshold_kWh, SOC_kWh)
    
    if microgen_on: microgen_on = True
    else: microgen_on = False
    
    for index, row in consumption_df.iterrows():
        hour = index.hour
        consumption = row['Read Value']
        solar_gen = row['Solar']
        
        overconsumption = (consumption - solar_gen) if (consumption - solar_gen < 0) else 0
        
        if elec_rates is not None:
            night_start = elec_rates["night"]["hours"][0]; night_end = elec_rates["night"]["hours"][1]
            boost_start = elec_rates["EV"]["hours"][0]; boost_end = elec_rates["EV"]["hours"][1]
            peak_start = elec_rates["peak"]["hours"][0]; peak_end = elec_rates["peak"]["hours"][1]
        else: night_start = 23; night_end = 8; peak_start = 17; peak_end = 19; boost_end= 5
        
        SOC_kWh = next_SOC_kWh
        G2B = 0
        S2B = 0
        B2H = 0
        S2H = 0
        G2H = 0
        S2G = 0
        Grid_import = 0
        
        # Charging at night
        if morning_discharge_on == True:
            if hour == 0: Slow_Charge_Rate_set_flag = False
            if hour == night_start:
                if Slow_Charge_Rate_set_flag == False : 
                    night_charge_rate = det_charge_rate(night_start, boost_end, charge_threshold_kWh, discharge_threshold_kWh, SOC_kWh)
                    Slow_Charge_Rate_set_flag = True
            
            if hour >= night_start or hour < boost_end:  # Night hours
                if B2H == 0 and S2B == 0:   # ensures no other flows
                    G2B = G_2_B(SOC_kWh, charge_threshold_kWh, max_charge_discharge_rate, 
                                slow_charge_on=True, slow_charge_rate=night_charge_rate)                
        else: 
            G2B = 0.0    
        
        #S2H
        if solar_gen > 0.1:
            S2H = S_2_H(consumption, solar_gen, max_inverter_ac_power)

            #S2B
            if overconsumption < -0.1:
                if (hour < peak_start or hour >= peak_end): # if not in peak hours
                    S2B= S_2_B(consumption, solar_gen, SOC_kWh, charge_threshold_kWh, max_inverter_ac_power)
                    if microgen_on == True:
                        if S2B == 0:
                            S2G = S_2_G(consumption, solar_gen, max_inverter_ac_power)
                        else:
                            S2G = S_2_G(consumption, solar_gen, max_inverter_ac_power, battery_full=False, S2B=S2B)
                    else: S2G = 0.0
                else: S2B = 0.0    

        #B2H
        if SOC_kWh > discharge_threshold_kWh and S2B == 0 and G2B == 0: # if battery is not empty or charging
            if hour >= boost_end and morning_discharge_done_flag == False: # discharge in morning hours
                B2H = B_2_H(consumption, SOC_kWh, S2H, discharge_threshold_kWh, max_charge_discharge_rate)
                if SOC_kWh == discharge_threshold_kWh: morning_discharge_done_flag = True # ends morning discharge if completed
                
            elif hour >= peak_start and hour < night_start: # ensures peak hours are covered
                B2H = B_2_H(consumption, SOC_kWh, S2H, discharge_threshold_kWh, max_charge_discharge_rate)
                
        #G2H
        G2H = G_2_H(consumption, B2H, S2H)
        if G2H < 0: G2H = 0.0
        Grid_import = G2H + G2B
            
        next_SOC_kWh = SOC_kWh + G2B - B2H + S2B
        if next_SOC_kWh > charge_threshold_kWh:
            next_SOC_kWh = charge_threshold_kWh
        elif next_SOC_kWh < discharge_threshold_kWh:
            next_SOC_kWh = discharge_threshold_kWh
            
        # Appending energy flows and costs to lists
        battery_SOC.append(SOC_kWh)
        charge_from_grid.append(G2B)
        discharge_to_home.append(B2H)
        charge_from_solar.append(S2B)
        solar_to_home.append(S2H)
        grid_to_home.append(G2H)
        solar_to_grid.append(S2G)
        grid_imports.append(Grid_import)

    # Adding relevant values to consumption dataframe
    consumption_df.loc[:, 'G2B'] = charge_from_grid
    consumption_df.loc[:, 'B2H'] = discharge_to_home
    consumption_df.loc[:, 'SOC'] = battery_SOC
    consumption_df.loc[:, 'S2B'] = charge_from_solar
    consumption_df.loc[:, 'S2H'] = solar_to_home
    consumption_df.loc[:, 'G2H'] = grid_to_home
    consumption_df.loc[:, 'S2G'] = solar_to_grid
    consumption_df.loc[:, 'Grid Import'] = grid_imports

    consumption_df = consumption_df[['Read Value', 'Solar' ,'SOC', 'G2B','B2H', 'S2H', 'S2B', 'G2H', 'S2G', 'Grid Import']]
    return consumption_df

def battery_algorithm_no_solar(consumption_df, 
                               battery_capacity, charge_threshold, discharge_threshold, battery_efficiency=0.85):
    '''Battery charge/discharge algorithm for no solar input.
    returns df with Read Value, SOC, G2B, B2H, G2H, Grid Import columns
    CURRENTLY WORKS BUT ALGORITHM IS TOO SIMPLE TO PRODUCE MEANINGFUL RESULTS
    '''
    
    #handling charge/discharge thresholds and setting initial values
    charge_threshold_percentage = charge_threshold/100.0
    discharge_threshold_percentage = discharge_threshold/100.0
    SOC_percentage = discharge_threshold_percentage # state-of-charge of the battery. taken as minimum for now
    SOC_kWh = SOC_percentage*battery_capacity
    next_SOC_kWh = SOC_kWh

    #initialising arrays to store energy flows
    battery_SOC = []
    discharge_to_home = []
    charge_from_grid = []
    grid_to_home = []
    grid_imports = []
    
    charge_discharge_rate = get_charge_discharge_rate(battery_capacity)
    # converting percentages into kWh values
    charge_threshold_kWh = charge_threshold_percentage*battery_capacity
    discharge_threshold_kWh = discharge_threshold_percentage*battery_capacity
    
    max_charge_discharge_rate = 0.5 * charge_discharge_rate
    morning_discharge_done_flag = False
    peak_discharge_done_flag = True
    night_charge_rate_set_flag = False
    day_charge_rate_set_flag = False
    
    night_slow_charge_rate = det_charge_rate(0, 5, charge_threshold_kWh, discharge_threshold_kWh, SOC_kWh)

    night_start = 23; night_end = 8; peak_start = 17; peak_end = 19; boost_end= 5
    
    for index, row in consumption_df.iterrows():
        
        hour = index.hour
        consumption = row['Read Value']

        SOC_kWh = next_SOC_kWh
        G2B = 0
        B2H = 0
        G2H = 0
        Grid_import = 0
        
        # Charging at night

        if hour == night_start:
            if night_charge_rate_set_flag == False : 
                night_slow_charge_rate = det_charge_rate(hour, boost_end, charge_threshold_kWh, discharge_threshold_kWh, SOC_kWh)
                night_charge_rate_set_flag = True
        
        if (hour >= night_start or hour < boost_end) and B2H == 0:  # Night hours
            
            G2B = G_2_B(SOC_kWh, charge_threshold_kWh, max_charge_discharge_rate, 
                        slow_charge_on=True, slow_charge_rate=night_slow_charge_rate)
        
        #B2H - morning discharge
        if hour == boost_end: morning_discharge_done_flag = False
        if hour> boost_end:
            
            if SOC_kWh > discharge_threshold_kWh and morning_discharge_done_flag == False and G2B == 0:
                
                B2H = B_2_H(consumption, SOC_kWh, S2H=0.0, discharge_threshold_kWh=discharge_threshold_kWh, max_charge_discharge_rate=max_charge_discharge_rate)
        
        if hour >= 12 and SOC_kWh == discharge_threshold_kWh and morning_discharge_done_flag == False:
            morning_discharge_done_flag = True
            
        # Charging before peak hours
        if morning_discharge_done_flag == True and day_charge_rate_set_flag == False:
            day_charge_rate = det_charge_rate(hour, 15, charge_threshold_kWh, discharge_threshold_kWh, SOC_kWh)
            day_charge_rate_set_flag = True
        
        if day_charge_rate_set_flag == True and hour <= 15 and hour >= 12 and B2H == 0:
            G2B = G_2_B(SOC_kWh, charge_threshold_kWh, max_charge_discharge_rate, 
                        slow_charge_on=True, slow_charge_rate=day_charge_rate)
            
        # Discharging during peak hours (and after)
        if hour == peak_start: peak_discharge_done_flag = False
        if hour >= peak_start and peak_discharge_done_flag == False:
            if SOC_kWh > discharge_threshold_kWh and peak_discharge_done_flag == False and G2B == 0:
                B2H = B_2_H(consumption, SOC_kWh, S2H=0.0, discharge_threshold_kWh=discharge_threshold_kWh, max_charge_discharge_rate=max_charge_discharge_rate)
        if hour == night_start:
            peak_discharge_done_flag = True
        #G2H
        G2H = G_2_H(consumption, B2H, S2H=0.0)
        Grid_import = G2H + G2B
            
        next_SOC_kWh = SOC_kWh + G2B - B2H
        if next_SOC_kWh > charge_threshold_kWh:
            next_SOC_kWh = charge_threshold_kWh
        elif next_SOC_kWh < discharge_threshold_kWh:
            next_SOC_kWh = discharge_threshold_kWh
        
        
        # Appending energy flows and costs to lists
        battery_SOC.append(SOC_kWh)
        charge_from_grid.append(G2B)
        discharge_to_home.append(B2H)
        grid_to_home.append(G2H)
        grid_imports.append(Grid_import)


    # Adding relevant values to consumption dataframe
    consumption_df.loc[:, 'G2B'] = charge_from_grid
    consumption_df.loc[:, 'B2H'] = discharge_to_home
    consumption_df.loc[:, 'SOC'] = battery_SOC
    consumption_df.loc[:, 'G2H'] = grid_to_home
    consumption_df.loc[:, 'Grid Import'] = grid_imports

    consumption_df = consumption_df[['Read Value', 'SOC', 'G2B','B2H', 'G2H', 'Grid Import']]

    return consumption_df


def Solar_only_algorithm(consumption_df, solar_generation_df, microgen_on=True):
    '''Algorithm for solar-only setup.
    returns df with Read Value, S2H, S2G, and Grid Import columns
    '''
    grid_imports = []
    solar_to_home = []
    solar_to_grid = []
    # print(solar_generation_df.head(10))
    consumption_df.loc[:,"Solar"] = solar_generation_df["Energy Output (kWh)"]
    
    if microgen_on: microgen_on = True
    else: microgen_on = False
    
    for index, row in consumption_df.iterrows():
        consumption = row['Read Value']
        solar_gen = row['Solar']

        S2H = 0
        S2G = 0
        Grid_import = 0
        
        if solar_gen > 0.1:
            S2H = S_2_H(consumption, solar_gen, max_inverter_ac_power=5.0)
            if microgen_on == True:
                S2G = S_2_G(consumption, solar_gen, max_inverter_ac_power=5.0)
            else: S2G = 0.0
        Grid_import = G_2_H(consumption, S2H=S2H, B2H=None)
        
        solar_to_home.append(S2H)
        solar_to_grid.append(S2G)
        grid_imports.append(Grid_import)
        
    consumption_df.loc[:, 'Solar'] = solar_generation_df["Energy Output (kWh)"]    
    consumption_df.loc[:, 'S2H'] = solar_to_home
    consumption_df.loc[:, 'S2G'] = solar_to_grid
    consumption_df.loc[:, 'Grid Import'] = grid_imports
    
    consumption_df = consumption_df[['Read Value', 'Solar', 'S2H', 'S2G', 'Grid Import']]
    return consumption_df

def reformat_retrofit_algorithm_output(battery_output_df):
    if 'Grid Import' not in battery_output_df.columns:
        battery_output_df.loc[:, 'Grid Import'] = battery_output_df['Read Value']
    battery_output_df = battery_output_df[['Grid Import']]
    battery_output_df = battery_output_df.rename(columns={'Grid Import': 'Grid Imports [kWh]'})

    return battery_output_df

def extract_rates(elec_plans, plan_name):
    for plan in elec_plans:
        if plan.plan_name == plan_name:
            return plan.rates
    return None

def calculate_kWh_and_GHG_totals(consumption_df):
    '''Calculates the total consumption and CO2 emissions for the current setup
    Input: requires a DataFrame with grid consumption data in the last column
    '''
    total_consumption = 0; total_emissions = 0
    if consumption_df is not None:
        if not consumption_df.empty:
            import_export_df = pd.DataFrame(
                index=consumption_df.index,
                data={'Grid Import': consumption_df.iloc[:, -1]}
            )
            if "S2G" in consumption_df.columns:
                import_export_df["Export"] = consumption_df["S2G"]
                
            
            total_consumption = import_export_df["Grid Import"].sum()
            total_emissions = calculate_ghg_emissions(import_export_df, GHG_intensities_2023)['gCO2 emissions'].sum()
            
            total_emissions = total_emissions / 1000.0  # Convert to kgCO2

    return total_consumption, total_emissions

# CAPEX stuff
def grant_calculator(solar_size_kWp, battery_size_kWh):
    '''Calculates the grant amount for solar PV and battery installations
    According to the SEAI (https://www.seai.ie/grants/home-energy-grants/individual-grants/solar-electricity-grant)
    '''

    grant_amount = 0
    #Solar:
    for i in range(1, int(solar_size_kWp)):
        if  i <= 2:
            grant_amount += 700
        else:
            grant_amount += 200
            
    grant_amount = min(grant_amount, 1800)  #caps out at €1800
    
    Battery_grant = False   #potentially use this to propose a battery grant?
    #Battery:
    if Battery_grant == True:
        return ''
    
    return grant_amount
    

def retrofit_capex(solar_size_kWp, battery_size_kWh, apply_grant=False):
    '''calculates capital cost of installing solar and battery'''
    
    if solar_size_kWp == 0.0: solar_cost = 0.0
    else: 
        solar_price_per_kWp = 1750 # €/kWp #! assumption from https://switcher.ie/solar-panels/#:~:text=Typically%2C%20a%20domestic%20solar%20PV,%E2%82%AC2%2C000%20per%20kW%20installed.
        solar_cost = solar_size_kWp * solar_price_per_kWp
    
    
    if battery_size_kWh == 0.0: battery_cost = 0.0
    else: 
        # print(battery_size_kWh)
        battery_cost = 1100 + 400*battery_size_kWh
    
    if apply_grant==True:
        grant_amount = grant_calculator(solar_size_kWp, battery_size_kWh)
    else: grant_amount = 0.0
    
    total_capex = solar_cost + battery_cost - grant_amount
    
    return solar_cost, battery_cost, total_capex



def cost_of_retrofit(solar_size_kWp, battery_size_kWh, years_of_installation, N_battery_cycles_per_annum=(2*365), Apply_Grant=True):
    '''calculates capital cost of installing solar and battery'''
    if not years_of_installation: years_of_installation = 1
    
    # CAPEX
    solar_price_per_kWp = 1750 # €/kWp #! assumption from https://switcher.ie/solar-panels/#:~:text=Typically%2C%20a%20domestic%20solar%20PV,%E2%82%AC2%2C000%20per%20kW%20installed.
    solar_cost = solar_size_kWp * solar_price_per_kWp
    battery_cost = 1100 + 400*battery_size_kWh
    
    # Grant
    if Apply_Grant == True:
        grant_amount = grant_calculator(solar_size_kWp, battery_size_kWh)
    else: grant_amount = 0
    
    total_capex = solar_cost + battery_cost - grant_amount
    
    # OPEX
    total_opex = 0
    #! until battery cycles are accessible, assume 2
    if not N_battery_cycles_per_annum: N_battery_cycles_per_annum = 2*365
    
    battery_years_before_replacement = -(-8000 // N_battery_cycles_per_annum)  # Round up using integer division  #! 8000 cycles before replacement (https://www.sunsave.energy/solar-panels-advice/batteries/lifespan) they also suggest typical warranty lasting 10 years
    total_opex += battery_cost * int(years_of_installation / battery_years_before_replacement)
    
    
    # battery_maintenance_per_year = #! decide whether to add this
    
    
    return total_capex+total_opex

battery_component_kgCO2_per_kWh = {
    'Module': 100.0, 'Inverter': 50.0, 'Rest': 60.0
}

def component_ghg_emissions(solar_size_kWp, battery_size_kWh):
    '''Calculates the total GHG emissions of the components'''
    
    #SOLAR:
    solar_total_CTG = 1440*solar_size_kWp   #https://www.renewableenergyhub.co.uk/main/solar-panels/solar-panels-carbon-analysis
    
    battery_total_CTG = sum(battery_component_kgCO2_per_kWh.values())*battery_size_kWh
    
    return solar_total_CTG, battery_total_CTG

def payback_period(original_hdf, new_hdf,
                   plan, capex):
    '''Calculates the payback period in years and months'''
    # Calculate the annual savings
    original_costs = calculate_costs(convert_to_pivot(original_hdf), [plan], urban=True)[0]
    new_costs = calculate_costs(convert_to_pivot(new_hdf), [plan], urban=True)[0]
    
    annual_savings = original_costs["annual_cost_1st_year_discount"] - new_costs["annual_cost_1st_year_discount"]
    
    # Calculate the payback period
    if annual_savings <= 0:
        return float('inf')  # If there are no savings, return infinity
    
    payback_years = capex / annual_savings
    
    # Convert to years and months
    years = int(payback_years)
    months = int((payback_years - years) * 12)
    
    return years, months

def ghg_payback_period(original_hdf, new_hdf, GHG_capex):
    '''Calculates the GHG payback period in years and months'''
    # Calculate the annual GHG savings
    original_emissions = calculate_kWh_and_GHG_totals(original_hdf)[1]
    new_emissions = calculate_kWh_and_GHG_totals(new_hdf)[1]
    
    annual_ghg_savings = original_emissions - new_emissions
    
    # Calculate the GHG payback period
    if annual_ghg_savings <= 0:
        return float('inf')  # If there are no savings, return infinity
    
    payback_years = GHG_capex / annual_ghg_savings
    
    # Convert to years and months
    years = int(payback_years)
    months = int((payback_years - years) * 12)
    
    return years, months


# Plotting

import plotly.express as px

def split_consumption_by_endpoint(post_retrofit_df):
    totals = {}
    if 'SOC' in post_retrofit_df.columns and 'Solar' in post_retrofit_df.columns: # Battery and Solar
        excluded_columns = ["Solar", "SOC"]
    elif 'Solar' in post_retrofit_df.columns and 'SOC' not in post_retrofit_df.columns: # Solar Only
        excluded_columns = ["Solar"]
        # post_retrofit_df.rename(columns={"Grid Import": "G2H"}, inplace=True)
    elif 'SOC' in post_retrofit_df.columns and 'Solar' not in post_retrofit_df.columns: # Battery Only
        excluded_columns = ["SOC"]
    post_retrofit_df = post_retrofit_df.drop(columns=excluded_columns)
    for column in post_retrofit_df.columns:
        totals[column] = round(post_retrofit_df[column].sum(), 2)
    totals_df = pd.DataFrame(list(totals.items()), columns=['Endpoint', 'Total'])
    return totals_df


import plotly.graph_objects as go

def plot_average_load_profile(consumption_df, solar_size=0, battery_size=0, ghg_intensity_df=GHG_intensities_2023, elec_rates=None, season='Summer'):
    '''Plots the average load profile for the given consumption DataFrame'''
    
    consumption_df = consumption_df.copy()
    # if season == 'Summer': season = 'Summer'; elif season == 'Winter': season = 'Winter'

    # Add GHG intensity to the consumption DataFrame if available
    if ghg_intensity_df is not None:
        consumption_df = consumption_df.join(ghg_intensity_df, how='left')
    
    # Filter by season if specified
    if season == 'Summer':
        consumption_df = consumption_df[(consumption_df.index.month >= 4) & (consumption_df.index.month <= 9)]
    elif season == 'Winter':
        consumption_df = consumption_df[(consumption_df.index.month <= 3) | (consumption_df.index.month >= 10)]
    elif season == 'Full Year':
        consumption_df = consumption_df
    
    # Ensure only numeric columns are used for resampling
    numeric_columns = consumption_df.select_dtypes(include=['number'])
    half_hourly_data = numeric_columns.resample('30min').mean()

    # Add hour and minute columns
    half_hourly_data['Hour'] = half_hourly_data.index.hour
    half_hourly_data['Minute'] = half_hourly_data.index.minute

    # Group by hour and minute, then calculate the average for all numeric columns
    avg_profile = (
        half_hourly_data.groupby(['Hour', 'Minute'])
        .mean()
        .reset_index()
    )
    avg_profile['Time'] = avg_profile['Hour'] + avg_profile['Minute'] / 60.0

    # Create the base plot
    fig = go.Figure()

    # Plot "Read Value" (Original Import)
    if 'Read Value' in avg_profile.columns:
        fig.add_trace(go.Scatter(
            x=avg_profile['Time'],
            y=avg_profile['Read Value'],
            mode='lines',
            name='Original Import',
            line=dict(color='blue'),
            yaxis='y1'
        ))

    # Plot "Grid Import" (Retrofit Import)
    if 'Grid Import' in avg_profile.columns:
        fig.add_trace(go.Scatter(
            x=avg_profile['Time'],
            y=avg_profile['Grid Import'],
            mode='lines',
            name='Retrofit Import',
            line=dict(color='green'),
            yaxis='y1'
        ))

    # Plot "Solar" (Solar Generation)
    if 'Solar' in avg_profile.columns:
        fig.add_trace(go.Scatter(
            x=avg_profile['Time'],
            y=avg_profile['Solar'],
            mode='lines',
            name='Solar Generation',
            line=dict(color='orange'),
            yaxis='y1',
            visible='legendonly'  # Make this line initially hidden
        ))

    # Plot "SOC" (Battery State of Charge)
    if 'SOC' in avg_profile.columns:
        avg_profile['SOC Percentage'] = (avg_profile['SOC'] / battery_size) * 100
        fig.add_trace(go.Scatter(
            x=avg_profile['Time'],
            y=avg_profile['SOC Percentage'],
            mode='lines',
            name='Battery SOC [%]',
            line=dict(color='purple', dash='dot'),
            yaxis='y4',
            visible='legendonly'  # Make this line initially hidden
        ))

    # Plot "CO2 Intensity (gCO2/kWh)" (GHG Intensity)
    if 'CO2 Intensity (gCO2/kWh)' in avg_profile.columns:
        fig.add_trace(go.Scatter(
            x=avg_profile['Time'],
            y=avg_profile['CO2 Intensity (gCO2/kWh)'],
            mode='lines',
            name='Average Grid Intensity',
            line=dict(color='red', dash='dot'),
            yaxis='y2'
        ))

    # Plot electricity rates if provided
    if elec_rates is None:
        elec_rates = extract_rates(Elec_plans, "EV Smart Electricity Discount with €50 Welcome Bonus")
    if elec_rates is not None:
        hours = list(range(24))
        rates = [get_rate(hour, elec_rates) for hour in hours]
        fig.add_trace(go.Scatter(
            x=hours,
            y=rates,
            mode='lines+markers',
            name='Electricity Rates',
            line=dict(color='purple', dash='dot'),
            yaxis='y3',
            visible='legendonly'  # Make this line initially hidden
        ))
    # Update layout to make lines toggleable and add secondary y-axes
    fig.update_layout(
        title=f'Avg {season} Load Profile (Solar = <b>{solar_size} kW</b> & Battery = <b>{battery_size} kWh<b>)',
        xaxis=dict(
            title='Hour of the Day',
            range=[0, 24],  # Hide x-axis before x=0
            showgrid=True
        ),
        yaxis=dict(
            title='Average Consumption [kWh]',
            side='left'
        ),
         yaxis2=dict(
            title='Grid Intensity [kgCO2/kWh]',
            overlaying='y',
            side='right',
            range=[
                avg_profile['CO2 Intensity (gCO2/kWh)'].min()-5, 
                avg_profile['CO2 Intensity (gCO2/kWh)'].max()+5
            ] if 'CO2 Intensity (gCO2/kWh)' in avg_profile.columns else None,  # Check if column exists
            showgrid=False,
        ),
       yaxis3=dict(
            title='Electricity Rates [€/kWh]',
            overlaying='y',
            side='left',
            showgrid=False,
            visible=False  # Make the third axis invisible
        ),
        yaxis4=dict(
            title='Battery SOC [%]',
            overlaying='y',
            side='right',
            showgrid=False,
            visible=False  # Make the fourth axis invisible
        ),
        legend=dict(
            title='Legend',
            orientation='h',
            yanchor='top',
            y=-0.3,  # Position below the x-axis title
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified'
    )
    return fig


def reformat_dash_hdf_store(df_data):
    '''Reformats the averaged dataframe for storage in the dash store'''
    if isinstance(df_data, list) and all(isinstance(i, dict) for i in df_data):
        formatted_df = pd.DataFrame(df_data)
        if 'Datetime' in formatted_df.columns:
            formatted_df.set_index('Datetime', inplace=True)
        elif 'DateTime' in formatted_df.columns:
            formatted_df.set_index('DateTime', inplace=True)
            formatted_df.rename(columns={'index': 'Datetime'}, inplace=True)
            # Set the datetime index
        else:
            raise ValueError("Datetime column is missing in the input data")

        formatted_df.index = pd.to_datetime(formatted_df.index)
            
        return formatted_df
    else:
        raise ValueError("input df is not in the expected format")

#default styles:

default_box_style = {'border': '1px solid #ccc', 'padding': '10px', 'padding-bottom': '5px', 'margin-top': '10px', 'margin-bottom': '5px',
                                'border-radius': '10px', 'box-shadow': '2px 2px 10px rgba(0, 0, 0, 0.1)',
                                'background-color': '#f9f9f9', 'width': '50%', 'text-align': 'left',
                                'margin-left': 'auto', 'margin-right': 'auto', 'font-size': '12px','width': '90%',
                         }

default_button_style = {'width': '100%', 'margin-top': '5px', 'border-radius': '10px'}

default_additional_input_style = {'width': '33%', 'font-size': '12px', 'height': '20px'}

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.FLATLY]
import dash_bootstrap_templates as dbt
dbt.load_figure_template('flatly')

##############################################################################################################! Layout

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div(
    style={'display': 'flex', 'flex-direction': 'row'},
    children=[
        html.Div(
            style={'flex': '0.2',
                   'padding': '10px',
                   'backgroundColor': '#f0f0f0',
                   'border-right': '1px solid #ccc'},
            children=[
                html.H2("Base Setup", style={'textAlign': 'center'}),
                html.Br(),
                dcc.RadioItems(
                    list({"Urban", "Rural"}), "Urban",
                    id='Urban-Rural',
                ),
                html.Br(),
                html.Label("County"),
                dcc.Dropdown(
                    id='county-dropdown',
                    value='Cork', options=county_dropdown_options,
                    style={'display': 'block', 'margin-top': '5px', 'fontSize': '14px'}
                ),
                html.Br(),
                html.Label("Upload your HDF file:"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '75%',
                        'lineHeight': 'auto',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin-right': 'auto',
                        'margin-left': 'auto'
                    },
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
                html.Span(style={'display': 'block', 'height': '5px'}),
                html.Button("Process HDF File", id='process-hdf-button', n_clicks=0, style={'width': '100%'}),
                dcc.Store(id='reformatted-hdf-df-store'),
                html.Br(),
                html.Button("Calculate totals", id='calculate-button-0', n_clicks=0, style={'width': '100%'}),
                html.Div(id='consumption-and-GHG-totals-0',
                         style=default_box_style,
                ),
                dcc.Store(id='consumption-and-GHG-totals-0-store'),
                html.Div(
                    id='cheapest-grid-price',
                    style=default_box_style
                ),
                dcc.Store(id='cheapest-price-store-0'),
                html.Label("Number of years to simulate:"),
                dcc.Input(
                    id='N-years-0',
                    type='number',
                    value=1,
                    min=1, max=10, step=1,
                    style={'width': 'auto', 'height': 'auto', 'margin': 'auto'} 
                ),
                html.Div(
                    id='cost-N-years-0',
                    style=default_box_style                    
                ),
            ],
        ),
        #! SETUP 1
        html.Div(
            style={'flex': '1', 'padding': '10px', 'border-right': '1px solid #ccc'},
            children=[
                html.H2("SETUP 1", style={'textAlign': 'center'}),
                html.Br(),
                html.Label("Solar PV"),
                dcc.Slider(
                    0, 10, step=0.5, marks=None, value=0, 
                    tooltip={"placement": "bottom", "always_visible": False},
                    id='Solar-Size-1'
                ),
                html.Div([
                    html.Div([
                        html.Label("Tilt [º]", style={'display': 'block', 'font-size': '12px'}),
                        dcc.Input(
                            id='tilt-input-1', type='number', placeholder='Enter tilt',
                            value=30, style=default_additional_input_style
                        )
                    ], style={'flex': '1', 'padding': '5px'}),
                    html.Div([
                        html.Label("Azimuth [º]", style={'display': 'block', 'font-size': '12px'}),
                        dcc.Input(
                            id='azimuth-input-1', type='number', placeholder='Enter azimuth',
                            value=180, style=default_additional_input_style
                        )
                    ], style={'flex': '1', 'padding': '5px'}),
                    html.Div([
                        html.Label("Performance Ratio [%]", style={'display': 'block', 'font-size': '12px'}),
                        dcc.Input(
                            id='performance-ratio-input-1', type='number', placeholder='Enter Performance Ratio',
                            value=70, style=default_additional_input_style
                        )
                    ], style={'flex': '1', 'padding': '5px'})
                ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}),
                
                html.Br(),
                html.Label("Battery Size"),
                dcc.Slider(
                    0, 16, step=1, marks=None, value=0, 
                    tooltip={"placement": "bottom", "always_visible": False},
                    id='Battery-Size-1',
                ),
                html.Div([
                    html.Div([
                        html.Label("Charge Threshold [%]", style={'display': 'block', 'font-size': '12px'}),
                        dcc.Input(
                            id='charge-threshold-input-1', type='number', placeholder='Enter charge threshold',
                            value=90, style=default_additional_input_style
                        )
                    ], style={'flex': '1', 'padding': '5px'}),
                    html.Div([
                        html.Label("Discharge Threshold [%]", style={'display': 'block', 'font-size': '12px'}),
                        dcc.Input(
                            id='discharge-threshold-input-1', type='number', placeholder='Enter discharge threshold',
                            value=10, style=default_additional_input_style
                        )
                    ], style={'flex': '1', 'padding': '5px'})
                ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}),
                html.Br(),
                
                # OUTPUTS
                html.H4("EMBEDDED COST/CARBON", style={'textAlign': 'center'}),
                dcc.Checklist(["Apply Grant"], id='apply-grant-1', style={'textAlign': 'center'}),
                html.Div(
                    children=[
                        html.Div(id='capex-1'),
                    ], style=default_box_style,
                ),
                dcc.Store(id='capex-store-1', data=None),
                dcc.Store(id='ghg-capex-store-1', data=None),
                html.Br(),
                html.Button("Calculate", id='calculate-button-1', n_clicks=0, style={'width': '100%'}),
                html.Button("Display", id='display-button-1', n_clicks=0, style={'width': '100%'}),
                html.Br(),
                html.Br(),
                html.H4("ELECTRICITY PLAN OPTIONS:", style={'textAlign': 'center'}),
                dcc.Checklist(["Apply Microgen"], id='apply-microgen-1', value=["Apply Microgen"], style={'textAlign': 'center'}),
                html.Div(
                    children=[
                        html.Div(id='calculate-costs-1'),
                        dcc.Store(id='cheapest-price-store-1'),
                    ], style=default_box_style,
                ),
                dcc.Store(id='full-retrofit-output-store-1'),
                html.Br(),
                html.H3("COMPARISON TO BASE CASE", style={'textAlign': 'center'}),
                html.Div(id='comparison-to-base-case-1',
                         style=default_box_style 
                ),
                html.Br(),
                
                html.Div(
                    dbc.Accordion(
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'align-items': 'center'},
                                    children=[
                                        html.Label("Number of years to simulate:"),
                                        dbc.Input(id='N-years-simulation-1', type='number', value=1, min=1, max=10, step=1, style={'width': '25%'}),
                                        html.Div(id='cost-N-years-1', style=default_box_style),
                                    ],                                    
                                ),
                                html.Span(style={'height': '10px'}),
                                html.Label("Payback Period:"),
                                html.Div(
                                    id='payback-period-1',
                                    style=default_box_style,
                                )
                            ],
                            title="Cost Breakdown",
                            # start_collapsed=True,
                        ),
                    ),
                ),
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    style={'display': 'flex', 'justify-content': 'flex-start', 'align-items': 'center', 'margin-bottom': '10px'},
                                    children=[
                                        dbc.Button("Show Graphs", id='show-graphs-1', n_clicks=0, style={'margin-right': '10px'}),
                                        dbc.RadioItems(
                                            id='season-toggle-1',
                                            options=[
                                                {'label': 'Full Year', 'value': 'Full Year'},
                                                {'label': 'Summer', 'value': 'Summer'},
                                                {'label': 'Winter', 'value': 'Winter'}
                                            ],
                                            value='Full Year',
                                            inline=True
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id='load-profile-1',
                                    style={'width': '100%'}
                                    # style=default_box_style
                                ),
                                dcc.Graph(
                                    id='consumption-pie-graph-1',
                                    style={'width': '100%'}
                                ),
                                ],
                                title="Consumption Breakdown",
                            ),
                        ],
                    ),

            ],
        ),
        #! SETUP 2
        html.Div(
            style={'flex': '1', 'padding': '10px'},
            children=[
                html.H2("SETUP 2", style={'textAlign': 'center'}),
                html.Br(),
                html.Label("Solar PV"),
                dcc.Slider(
                    0, 10, step=0.5, marks=None, value=0,
                    tooltip={"placement": "bottom", "always_visible": False},
                    id='Solar-Size-2'
                ),
                ## additional solar inputs:
                html.Div([
                    html.Div([
                        html.Label("Tilt [º]", style={'display': 'block', 'font-size': '12px'}),
                        dcc.Input(
                            id='tilt-input-2', type='number', placeholder='Enter tilt',
                            value=30, style=default_additional_input_style
                        )
                    ], style={'flex': '1', 'padding': '5px'}),
                    html.Div([
                        html.Label("Azimuth [º]", style={'display': 'block', 'font-size': '12px'}),
                        dcc.Input(
                            id='azimuth-input-2', type='number', placeholder='Enter azimuth',
                            value=180, style=default_additional_input_style
                        )
                    ], style={'flex': '1', 'padding': '5px'}),
                    html.Div([
                        html.Label("Perforance Ratio [%]", style={'display': 'block', 'font-size': '12px'}),
                        dcc.Input(
                            id='performance-ratio-input-2', type='number', placeholder='Enter Performance Ratio',
                            value=70, style=default_additional_input_style
                        )
                    ], style={'flex': '1', 'padding': '5px'})
                ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}),

                html.Br(),
                html.Label("Battery Size"),
                dcc.Slider(
                    0, 16, step=0.5, marks=None, value=0,
                    tooltip={"placement": "bottom", "always_visible": False},
                    id='Battery-Size-2'
                ),
                html.Div([
                    html.Div([
                        html.Label("Charge Threshold [%]", style={'display': 'block', 'font-size': '12px'}),
                        dcc.Input(
                            id='charge-threshold-input-2', type='number', placeholder='Enter charge threshold',
                            value=90, style=default_additional_input_style
                        )
                    ], style={'flex': '1', 'padding': '5px'}),
                    html.Div([
                        html.Label("Discharge Threshold [%]", style={'display': 'block', 'font-size': '12px'}),
                        dcc.Input(
                            id='discharge-threshold-input-2', type='number', placeholder='Enter discharge threshold',
                            value=10, style=default_additional_input_style
                        )
                    ], style={'flex': '1', 'padding': '5px'})
                ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}),
                html.Br(),
                
                # OUTPUTS
                html.H4("EMBEDDED COST/CARBON", style={'textAlign': 'center'}),
                dcc.Checklist(["Apply Grant"], id='apply-grant-2', style={'textAlign': 'center'}),
                html.Div(
                    children=[
                        html.Div(id='capex-2'),
                    ], style=default_box_style,
                ),
                dcc.Store(id='capex-store-2', data=None),
                dcc.Store(id='ghg-capex-store-2', data=None),
                html.Br(),
                html.Button("Calculate", id='calculate-button-2', n_clicks=0, style={'width': '100%'}),
                html.Button("Display", id='display-button-2', n_clicks=0, style={'width': '100%'}),
                html.Br(),
                html.Br(),
                html.H4("ELECTRICITY PLAN OPTIONS:", style={'textAlign': 'center'}),
                dcc.Checklist(["Apply Microgen"], id='apply-microgen-2', value=["Apply Microgen"], style={'textAlign': 'center'}),
                html.Div(
                    children=[
                        html.Div(id='calculate-costs-2'),
                        dcc.Store(id='cheapest-price-store-2'),
                    ], style=default_box_style,
                ),
                dcc.Store(id='full-retrofit-output-store-2'),
                html.Br(),
                html.H3("COMPARISON TO BASE CASE", style={'textAlign': 'center'}),
                html.Div(id='comparison-to-base-case-2',
                         style=default_box_style 
                ),
                html.Br(),

                html.Div(
                    dbc.Accordion(
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'align-items': 'center'},
                                    children=[
                                        html.Label("Number of years to simulate:"),
                                        dbc.Input(id='N-years-simulation-2', type='number', value=1, min=1, max=10, step=1, style={'width': '25%'}),
                                        html.Div(id='cost-N-years-2', style=default_box_style),
                                    ],                                    
                                ),
                                html.Span(style={'height': '10px'}),
                                html.Label("Payback Period:"),
                                html.Div(
                                    id='payback-period-2',
                                    style=default_box_style,
                                )
                            ],
                            title="Cost Breakdown",
                            # start_collapsed=True,
                        ),
                    ),
                ),
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    style={'display': 'flex', 'justify-content': 'flex-start', 'align-items': 'center', 'margin-bottom': '10px'},
                                    children=[
                                        dbc.Button("Show Graphs", id='show-graphs-2', n_clicks=0, style={'margin-right': '10px'}),
                                        dbc.RadioItems(
                                            id='season-toggle-2',
                                            options=[
                                                {'label': 'Full Year', 'value': 'Full Year'},
                                                {'label': 'Summer', 'value': 'Summer'},
                                                {'label': 'Winter', 'value': 'Winter'}
                                            ],
                                            value='Full Year',
                                            inline=True
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id='load-profile-2',
                                    style={'width': '100%'}
                                    # style=default_box_style
                                ),
                                dcc.Graph(
                                    id='consumption-pie-graph-2',
                                    style={'width': '100%'}
                                ),
                                ],
                                title="Consumption Breakdown",
                            ),
                        ],
                    ),
            ],
        ),
    ],
)




# Store the processed HDF data as a state
@app.callback(
    Output('reformatted-hdf-df-store', 'data'),
    Input('process-hdf-button', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def store_processed_hdf(n_clicks, list_of_contents, list_of_names, list_of_dates):
    if n_clicks == 0: return ''
    if n_clicks > 0:
        if list_of_contents is not None:
            for c, n, d in zip(list_of_contents, list_of_names, list_of_dates):
                content_type, content_string = c.split(',')
                decoded = base64.b64decode(content_string)
                try:
                    # Extracting csv from the decoded content
                    hdf_df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), index_col="Read Date and End Time")
                    # Reformatting HDF file
                    hdf_df = reformat_hdf(hdf_df)
                    # Averaging the data to a single year (2023)
                    single_year_hdf = process_to_single_year(hdf_df)
                    # Include the datetime index as a column
                    single_year_hdf.reset_index(inplace=True)
                    
                    return single_year_hdf.to_dict('records')
                
                except Exception as e:
                    print(f'There was an error processing this file: {e}')
                    return None
                
    else: return None
    

    
@app.callback(
    [Output('consumption-and-GHG-totals-0', 'children'),
     Output('consumption-and-GHG-totals-0-store', 'data')],
    Input('calculate-button-0', 'n_clicks'),
    State('reformatted-hdf-df-store', 'data'),
)
def display_consumption_and_GHG_totals_0(n_clicks, input_hdf_data):
    '''Calculates the total consumption and CO2 emissions for the current setup'''
    if n_clicks is None or n_clicks == 0:
        return '', None
    
    if input_hdf_data is not None:
        try:
            input_hdf = reformat_dash_hdf_store(input_hdf_data)
            total_consumption, total_emissions = calculate_kWh_and_GHG_totals(input_hdf)
            
            return html.Div([
                html.P(f"Total Consumption (Grid Import): {total_consumption:,.2f} kWh"),
                html.P(f"Total CO2 Emissions: {total_emissions:,.2f} kgCO2")
            ]), [total_consumption, total_emissions]
        
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)])
    
    return '', None

@app.callback(
    [Output('cheapest-grid-price', 'children'),
     Output('cheapest-price-store-0', 'data')],
    Input('calculate-button-0', 'n_clicks'),
    State('reformatted-hdf-df-store', 'data')
)
def display_cheapest_price(n_clicks, input_hdf_data):
    '''Calculates and displays the cheapest electricity price'''
    if n_clicks is None or n_clicks == 0:
        return '', None
    
    if input_hdf_data is not None:
        try:
            input_hdf = reformat_dash_hdf_store(input_hdf_data)
            consumption_pivot = convert_to_pivot(input_hdf)
            plan_options = calculate_costs(consumption_pivot, Elec_plans)
            
            cheapest_plan = plan_options[0]

            price_1st_year = cheapest_plan['annual_cost_1st_year_discount']
            price_no_discount = cheapest_plan['annual_cost_no_discount']
            
            return html.Div([
                html.Label(f"Cheapest elec plan price"),
                html.P(f"1st year: €{price_1st_year:,.2f}"),
                html.P(f"subsquent years: €{price_no_discount:,.2f}")
            ]), price_no_discount 
            # [price_1st_year, price_no_discount]
        
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)]), None
    
    return '', None

@app.callback(
    Output('cost-N-years-0', 'children'),
    Input('N-years-0', 'value'),
    State('reformatted-hdf-df-store', 'data')
)
def calculate_cost_N_years_0(N_years, input_hdf_data):
    '''Calculates the cost after N years using the 1st year price and then adds N-1 times the no discount annual cost'''
    if N_years is None or N_years <= 0:
        return ''
    
    if input_hdf_data is not None:
        try:
            input_hdf = reformat_dash_hdf_store(input_hdf_data)
            consumption_pivot = convert_to_pivot(input_hdf)
            plan_options = calculate_costs(consumption_pivot, Elec_plans)
            
            # Get the cheapest plan
            cheapest_plan = plan_options[0]
            first_year_cost = cheapest_plan['annual_cost_1st_year_discount']
            subsequent_year_cost = cheapest_plan['annual_cost_no_discount']
            
            total_cost = first_year_cost + (N_years - 1) * subsequent_year_cost
            
            return html.Div([
                html.P(f"{N_years}-year cost: €{total_cost:,.2f}")
            ])
        
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)])
    
    return ''

############################################################################################################! SETUP 1 CALLBACKS
    

@app.callback(
    [Output('capex-1', 'children'),
     Output('capex-store-1', 'data'),
     Output('ghg-capex-store-1', 'data')],
    Input('Solar-Size-1', 'value'),
    Input('Battery-Size-1', 'value'),
    Input('apply-grant-1', 'value')
)
def display_capex_1(solar_size, battery_size, apply_grant):
    '''Calculates the total capital expenditure for the current setup'''
    if solar_size == 0: solar_size = 0.0
    if battery_size == 0: battery_size = 0.0

    if apply_grant: apply_grant = True
    else: apply_grant = False
    
    solar_capex, battery_capex, total_capex = retrofit_capex(solar_size, battery_size, apply_grant)
    # total_capex = solar_capex + battery_capex
    
    CTG_by_component = component_ghg_emissions(solar_size, battery_size)
    solar_CTG, battery_CTG = CTG_by_component
    total_CTG = sum(CTG_by_component)
    
    # Table header and rows
    table_header = html.Tr([
        html.Th("Component", style={'font-size': '16px', 'text-transform': 'uppercase', 'text-align': 'left'}),
        html.Th("CAPEX [€]", style={'font-size': '16px', 'text-transform': 'uppercase', 'text-align': 'left'}),
        html.Th("GHG EMISSIONS [kgCO2]", style={'font-size': '16px', 'text-align': 'left'})
    ])
    table_rows = [
        html.Tr([
            html.Td("Solar", style={'font-size': '14px'}),
            html.Td(f"€{solar_capex:,.2f}", style={'font-size': '14px'}),
            html.Td(f"{solar_CTG:,.2f}", style={'font-size': '14px'})
        ]),
        html.Tr([
            html.Td("Battery", style={'font-size': '14px'}),
            html.Td(f"€{battery_capex:,.2f}", style={'font-size': '14px'}),
            html.Td(f"{battery_CTG:,.2f}", style={'font-size': '14px'})
        ]),
        html.Tr([
            html.Td("Total", style={'font-size': '14px', 'font-weight': 'bold'}),
            html.Td(f"€{total_capex:,.2f}", style={'font-size': '14px', 'font-weight': 'bold'}),
            html.Td(f"{total_CTG:,.2f}", style={'font-size': '14px', 'font-weight': 'bold'})
        ])
    ]
    
    # Return the table
    return html.Div([
        html.Table(
            [table_header] + table_rows,
            style={'width': '100%', 'border-collapse': 'collapse', 'text-align': 'left'}
        )
    ]), total_capex, CTG_by_component
    

@app.callback(
    Output('full-retrofit-output-store-1', 'data'),
    Input('calculate-button-1', 'n_clicks'),
    Input('apply-microgen-1', 'value'),
    State('reformatted-hdf-df-store', 'data'),
    State('Solar-Size-1', 'value'),
    State('tilt-input-1', 'value'),
    State('azimuth-input-1', 'value'),
    State('performance-ratio-input-1', 'value'),
    State('county-dropdown', 'value'),
    State('Battery-Size-1', 'value'),
    State('charge-threshold-input-1', 'value'),
    State('discharge-threshold-input-1', 'value'),
)
def calculate_outputs_1(n_clicks, microgen_on, input_hdf_data,
                          solar_capacity=0.0, tilt=30.0, azimuth=180.0, performance_ratio=70.0, county_choice='Cork',
                          battery_size=0.0, charge_threshold=90.0, discharge_threshold=10.0):
    '''Calculates the new consumption data with solar PV system and battery'''
    if n_clicks is None or n_clicks == 0:
        return ''
    
    if input_hdf_data is not None:
        try:
            input_hdf = reformat_dash_hdf_store(input_hdf_data)
            # # microgen option
            if microgen_on: microgen_on=True
            else: microgen_on=False
            
            # microgen_on = True
            
            solar_data = get_solar_data(county_choice, input_hdf.index[0], input_hdf.index[-1], tilt, azimuth)
            if solar_capacity != 0:
                solar_power_output = calculate_solar_power_output(solar_data, solar_capacity, performance_ratio)
            else: 
                solar_power_output = calculate_solar_power_output(solar_data, 0, 0)
            
            if battery_size == 0 and solar_capacity != 0:   # only solar
                # print("Solar only")
                full_retrofit_output = Solar_only_algorithm(input_hdf, solar_power_output, microgen_on=microgen_on)
                
            elif battery_size != 0 and solar_capacity == 0: # only battery
                # print("Battery only")
                full_retrofit_output = battery_algorithm_no_solar(input_hdf,
                                                                 battery_size, charge_threshold, discharge_threshold)
            elif battery_size != 0 and solar_capacity != 0: # solar and battery
                # print("Solar and Battery")
                full_retrofit_output = battery_algorithm(input_hdf, solar_power_output,
                                                    battery_size, charge_threshold, discharge_threshold, elec_rates=None, microgen_on=microgen_on)
            else: 
                # print("No solar or battery")
                full_retrofit_output = input_hdf
                
            # Ensure we are working with a copy to avoid SettingWithCopyWarning
            full_retrofit_output = full_retrofit_output.copy()
            # print(full_retrofit_output.head(10))
            
            # Include the datetime index as a column
            full_retrofit_output.reset_index(inplace=True)
            full_retrofit_output.rename(columns={'index': 'DateTime'}, inplace=True)

            return full_retrofit_output.to_dict('records')
            
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)])

    return ''

@app.callback(
    [Output('calculate-costs-1', 'children'),
     Output('cheapest-price-store-1', 'data')],
    Input('display-button-1', 'n_clicks'),
    State('full-retrofit-output-store-1', 'data'),
    State('Solar-Size-1', 'value'),
    State('Battery-Size-1', 'value')
)
def calculate_elec_costs_1(n_clicks, new_output_data, solar_capacity_kWh, battery_capacity_kWh=0):
    '''Calculates the costs of electricity plans with new solar PV system and new battery'''
    if n_clicks is None or n_clicks == 0: 
        return '', None
    
    if new_output_data is not None:
        try:
            # reformatting solar/battery output
            new_output = reformat_dash_hdf_store(new_output_data)
            # print(new_output.sum())
            
            if solar_capacity_kWh > 0.1: grid_export = new_output['S2G'].sum()
            else: grid_export = 0.0

            formatted_output = reformat_retrofit_algorithm_output(new_output)
            formatted_output_pivot = convert_to_pivot(formatted_output)

            plan_options = calculate_costs(formatted_output_pivot, Elec_plans, grid_export=grid_export)

            cheapest_plan = min(plan_options, key=lambda x: x['annual_cost_no_discount'])
            cheapest_cost = cheapest_plan['annual_cost_no_discount']
            cheapest_provider = cheapest_plan['provider']
            cheapest_plan_name = cheapest_plan['plan_name']
            
            
            # Return results as an unordered list
            return format_elec_cost_output(plan_options), cheapest_cost
        
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)]), None

    return '', None

@app.callback(
    Output('comparison-to-base-case-1', 'children'),
    Input('display-button-1', 'n_clicks'),
    State('cheapest-price-store-0', 'data'),
    State('cheapest-price-store-1', 'data'),
    State('full-retrofit-output-store-1', 'data'),
    State('consumption-and-GHG-totals-0-store', 'data')
)
def display_consumption_and_GHG_totals_1(n_clicks, annual_cost_0, annual_cost_1, full_battery_output, base_consumption_and_GHG):
    '''Calculates the total consumption and CO2 emissions for the current setup'''
    if n_clicks is None or n_clicks == 0: 
        return ''
    # print("annual_cost_1", annual_cost_1)
    
    if full_battery_output is not None:
        try:
            # Reformat and calculate totals
            post_retrofit_df = reformat_dash_hdf_store(full_battery_output)
            total_consumption, total_emissions = calculate_kWh_and_GHG_totals(post_retrofit_df)
            
            # Comparison with base consumption
            base_consumption, base_emissions = base_consumption_and_GHG[0], base_consumption_and_GHG[1]
            delta_grid_import_kWh = total_consumption - base_consumption
            delta_emissions_kgCO2 = total_emissions - base_emissions
            percentage_change_consumption = (delta_grid_import_kWh / base_consumption) * 100
            percentage_change_emissions = (delta_emissions_kgCO2 / base_emissions) * 100
            
            # Cheapest price comparison
            delta_annual_cost = annual_cost_1 - annual_cost_0
            percentage_change_cost = (delta_annual_cost / annual_cost_0) * 100
            
            # Format as a table
            table_header = html.Tr([
                html.Th("Metric", style={"text-align": "left", "font-weight": "bold"}),
                html.Th("Value", style={"text-align": "left", "font-weight": "bold"}),
                html.Th("Change", style={"text-align": "left", "font-weight": "bold"})
            ])
            
            table_rows = [
                html.Tr([
                    html.Td("Total Grid Import (kWh)"),
                    html.Td(f"{total_consumption:,.2f} kWh"),
                    html.Td(
                        f"{delta_grid_import_kWh:,.2f} kWh ({percentage_change_consumption:,.2f}%)",
                        style={"color": "green" if delta_grid_import_kWh < 0 else "red"}
                    )
                ]),
                html.Tr([
                    html.Td("Total CO2 Emissions [kgCO2]"),
                    html.Td(f"{total_emissions:,.2f} kgCO2"),
                    html.Td(
                        f"{delta_emissions_kgCO2:,.2f} kgCO2 ({percentage_change_emissions:,.2f}%)",
                        style={"color": "green" if delta_emissions_kgCO2 < 0 else "red"}
                    )
                ]),
                html.Tr([
                    html.Td("Cheapest Annual Cost (€)"),
                    html.Td(f"€{annual_cost_1:,.2f}"),
                    html.Td(
                        f"€{delta_annual_cost:,.2f} ({percentage_change_cost:,.2f}%)",
                        style={"color": "green" if delta_annual_cost < 0 else "red"}
                    )
                ])
            ]
            
            return html.Div([
                html.Table(
                    [table_header] + table_rows,
                    style={"width": "100%", "border-collapse": "collapse", "margin-bottom": "20px"}
                )
            ])
        
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)])
    
    return ''

@app.callback(
    Output('consumption-pie-graph-1', 'figure'),
    Input('show-graphs-1', 'n_clicks'),
    Input('season-toggle-1', 'value'),
    State('full-retrofit-output-store-1', 'data')
)
def display_consumption_pie_1(n_clicks, season, full_retrofit_output):
    '''Displays a pie chart of the consumption split by endpoint and calculates the Self-Consumption Ratio (SCR)'''
    if n_clicks is None or n_clicks == 0:
        return {}
    if full_retrofit_output is not None:
        try:
            post_retrofit_df = reformat_dash_hdf_store(full_retrofit_output)
            
            # Filter by season if specified
            if season == 'Summer':
                post_retrofit_df = post_retrofit_df[(post_retrofit_df.index.month >= 4) & (post_retrofit_df.index.month <= 9)]
            elif season == 'Winter':
                post_retrofit_df = post_retrofit_df[(post_retrofit_df.index.month <= 3) | (post_retrofit_df.index.month >= 10)]
            elif season == 'Full Year':
                post_retrofit_df = post_retrofit_df
            
            # root_node = pd.DataFrame([{'Endpoint': 'Total Consumption', 'Total': post_retrofit_df['Read Value'].sum(), 'Parent': ''}])
            totals_df = split_consumption_by_endpoint(post_retrofit_df)
            totals_df['Parent'] = 'Total Consumption'
            # consumption_endpoint_totals_df = pd.concat([root_node, totals_df])
            
            # Note: since all battery discharge goes to home (B2H), the non-grid B2H = S2B
            if 'S2B' in totals_df['Endpoint'].values and 'S2H' in totals_df['Endpoint'].values:
                total_consumption = totals_df.loc[totals_df['Endpoint'].isin(['S2H', 'S2B', 'Grid Import']), 'Total'].sum()
                pie_chart_data = totals_df[totals_df['Endpoint'].isin(['S2B', 'Grid Import', 'S2H'])]
            elif 'S2H' in totals_df['Endpoint'].values and 'S2B' not in totals_df['Endpoint'].values:
                pie_chart_data = totals_df[totals_df['Endpoint'].isin(['Grid Import', 'S2H'])]
                total_consumption = totals_df.loc[totals_df['Endpoint'].isin(['S2H', 'Grid Import']), 'Total'].sum()
            else:
                pie_chart_data = totals_df[totals_df['Endpoint'].isin(['Grid Import'])]
                total_consumption = totals_df.loc[totals_df['Endpoint'].isin(['Read Value']), 'Total'].sum()

            # Calculate Self-Consumption Ratio (SCR)
            Grid_Import_total = totals_df.loc[totals_df['Endpoint'] == 'Grid Import', 'Total'].sum()
            SCR_percentage = 100*(1 - (Grid_Import_total / total_consumption)) if total_consumption > 0 else 0
            
            fig = px.pie(
                pie_chart_data,
                names='Endpoint',
                values='Total',
                title=f'Consumption proportion by source (Grid, Solar). Self-Consumption Ratio (SCR): {SCR_percentage:.2f} %',
            )
            
            
            return fig
        
        except Exception as e:
             return {
                'data': [],
                'layout': {
                    'title': f'There was an error processing this file: {e}'
                }
             }
    
    return {}


@app.callback(
    Output('load-profile-1', 'figure'),
    Input('show-graphs-1', 'n_clicks'),
    Input('season-toggle-1', 'value'),
    State('Solar-Size-1', 'value'),
    State('Battery-Size-1', 'value'),
    State('full-retrofit-output-store-1', 'data')
)
def display_load_profile_1(n_clicks, season, solar_capacity, battery_capacity, full_retrofit_output):
    '''Displays a line chart of the average seasonal load profile with a toggle for new and original data'''
    if n_clicks is None or n_clicks == 0:
        return {}
    
    if full_retrofit_output is not None:
        try:
            post_retrofit_df = reformat_dash_hdf_store(full_retrofit_output)
            return plot_average_load_profile(post_retrofit_df, 
                                             solar_capacity,
                                             battery_capacity,
                                             elec_rates=extract_rates(Elec_plans, "EV Smart Electricity Discount with €50 Welcome Bonus"),
                                             season=season
                                             )

        except Exception as e:
            return {
                'data': [],
                'layout': {
                    'title': f'There was an error processing this file: {e}'
                }
            }

    return {}

@app.callback(
    Output('cost-N-years-1', 'children'),
    Input('N-years-simulation-1', 'value'),
    State('full-retrofit-output-store-1', 'data')
)
def calculate_cost_N_years_1(N_years, full_retrofit_output):
    '''Calculates the cost after N years using the 1st year price and then adds N-1 times the no discount annual cost'''
    if N_years is None or N_years <= 0:
        return ''
    
    if full_retrofit_output is not None:
        try:
            input_hdf = reformat_dash_hdf_store(full_retrofit_output)
            retrofit_hdf = input_hdf[["Grid Import"]]
            consumption_pivot = convert_to_pivot(retrofit_hdf)
            plan_options = calculate_costs(consumption_pivot, Elec_plans)
            # print(plan_options[0])
            
            # Get the cheapest plan
            cheapest_plan = plan_options[0]
            first_year_cost = cheapest_plan['annual_cost_1st_year_discount']
            subsequent_year_cost = cheapest_plan['annual_cost_no_discount']
            
            total_cost = first_year_cost + (N_years - 1) * subsequent_year_cost
            
            return html.Div([
                html.P(f"{N_years}-year cost: €{total_cost:,.2f}")
            ])
        
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)])
    
    return ''

@app.callback(
    Output('payback-period-1', 'children'),
    Input('display-button-1', 'n_clicks'),
    State('reformatted-hdf-df-store', 'data'),
    State('full-retrofit-output-store-1', 'data'),
    State('capex-store-1', 'data'),
    State('ghg-capex-store-1', 'data')
)
def display_payback_period_1(n_clicks, original_input_data, full_retrofit_output, capex, GHG_capex):
    '''Calculates and displays the payback period for the current setup (in terms of cost and GHG emissions)'''
    if n_clicks is None or n_clicks == 0:
        return ''
    if original_input_data is not None and full_retrofit_output is not None and capex is not None:
        try:
            original_hdf = reformat_dash_hdf_store(original_input_data)
            retrofit_hdf = reformat_dash_hdf_store(full_retrofit_output)
            retrofit_hdf = retrofit_hdf[["Grid Import"]]
            
            plan = Elec_plans[0]  # Assuming you want to use the first plan for payback calculation
            
            # CAPEX Payback Period
            capex_years, capex_months = payback_period(original_hdf, retrofit_hdf, plan, capex)
            
            # Solar CAPEX Payback Period
            solar_capex = capex * (GHG_capex[0] / sum(GHG_capex))  # Proportional CAPEX for solar
            solar_years, solar_months = payback_period(original_hdf, retrofit_hdf, plan, solar_capex)
            
            # Battery CAPEX Payback Period
            battery_capex = capex * (GHG_capex[1] / sum(GHG_capex))  # Proportional CAPEX for battery
            battery_years, battery_months = payback_period(original_hdf, retrofit_hdf, plan, battery_capex)
            
            # GHG Payback Period
            total_GHG_capex = sum(GHG_capex)
            solar_ghg_years, solar_ghg_months = ghg_payback_period(original_hdf, retrofit_hdf, GHG_capex[0])
            battery_ghg_years, battery_ghg_months = ghg_payback_period(original_hdf, retrofit_hdf, GHG_capex[1])
            total_ghg_years, total_ghg_months = ghg_payback_period(original_hdf, retrofit_hdf, total_GHG_capex)
            
            # Format as a table
            table_header = html.Tr(
                [html.Th("COMPONENT"), html.Th("CAPEX PAYBACK"), html.Th("CARBON PAYBACK")],
                style={'font-size': '16px', 'font-weight': 'bold', 'text-transform': 'uppercase', 'border-bottom': '2px solid black'}
            )
            table_rows = [
                html.Tr([html.Td("Solar"), html.Td(f"{solar_years} years, {solar_months} months"), html.Td(f"{solar_ghg_years} years, {solar_ghg_months} months")]),
                html.Tr([html.Td("Battery"), html.Td(f"{battery_years} years, {battery_months} months"), html.Td(f"{battery_ghg_years} years, {battery_ghg_months} months")]),
                html.Tr(
                    [html.Td("Total"), html.Td(f"{capex_years} years, {capex_months} months"), html.Td(f"{total_ghg_years} years, {total_ghg_months} months")],
                    style={'font-weight': 'bold'}
                )
            ]
            
            return html.Div([
                html.Table([table_header] + table_rows, style={'width': '100%', 'border-collapse': 'collapse', 'text-align': 'left'}),
            ])
        
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)])
    
    return ''


############################################################################################################! SETUP 2 CALLBACKS

    

@app.callback(
    [Output('capex-2', 'children'),
     Output('capex-store-2', 'data'),
     Output('ghg-capex-store-2', 'data')],
    Input('Solar-Size-2', 'value'),
    Input('Battery-Size-2', 'value'),
    Input('apply-grant-2', 'value')
)
def display_capex_2(solar_size, battery_size, apply_grant):
    '''Calculates the total capital expenditure for the current setup'''
    # print(f"Slider values - Solar Size: {solar_size}, Battery Size: {battery_size}")
    if solar_size == 0: solar_size = 0.0
    if battery_size == 0: battery_size = 0.0

    if apply_grant: apply_grant=True
    else: apply_grant=False
    
    solar_capex, battery_capex, total_capex = retrofit_capex(solar_size, battery_size, apply_grant)
    # total_capex = solar_capex + battery_capex
    
    CTG_by_component = component_ghg_emissions(solar_size, battery_size)
    solar_CTG, battery_CTG = CTG_by_component
    total_CTG = sum(CTG_by_component)
    
    # Table header and rows
    table_header = html.Tr([
        html.Th("Component", style={'font-size': '16px', 'text-transform': 'uppercase', 'text-align': 'left'}),
        html.Th("CAPEX [€]", style={'font-size': '16px', 'text-transform': 'uppercase', 'text-align': 'left'}),
        html.Th("GHG EMISSIONS [kgCO2]", style={'font-size': '16px', 'text-align': 'left'})
    ])
    table_rows = [
        html.Tr([
            html.Td("Solar", style={'font-size': '14px'}),
            html.Td(f"€{solar_capex:,.2f}", style={'font-size': '14px'}),
            html.Td(f"{solar_CTG:,.2f}", style={'font-size': '14px'})
        ]),
        html.Tr([
            html.Td("Battery", style={'font-size': '14px'}),
            html.Td(f"€{battery_capex:,.2f}", style={'font-size': '14px'}),
            html.Td(f"{battery_CTG:,.2f}", style={'font-size': '14px'})
        ]),
        html.Tr([
            html.Td("Total", style={'font-size': '14px', 'font-weight': 'bold'}),
            html.Td(f"€{total_capex:,.2f}", style={'font-size': '14px', 'font-weight': 'bold'}),
            html.Td(f"{total_CTG:,.2f}", style={'font-size': '14px', 'font-weight': 'bold'})
        ])
    ]
    
    # Return the table
    return html.Div([
        html.Table(
            [table_header] + table_rows,
            style={'width': '100%', 'border-collapse': 'collapse', 'text-align': 'left'}
        )
    ]), total_capex, CTG_by_component

@app.callback(
    Output('full-retrofit-output-store-2', 'data'),
    Input('calculate-button-2', 'n_clicks'),
    Input('apply-microgen-2', 'value'),
    State('reformatted-hdf-df-store', 'data'),
    State('Solar-Size-2', 'value'),
    State('tilt-input-2', 'value'),
    State('azimuth-input-2', 'value'),
    State('performance-ratio-input-2', 'value'),
    State('county-dropdown', 'value'),
    State('Battery-Size-2', 'value'),
    State('charge-threshold-input-2', 'value'),
    State('discharge-threshold-input-2', 'value'),
)
def calculate_outputs_2(n_clicks, microgen_on, input_hdf_data,
                          solar_capacity=0.0, tilt=30.0, azimuth=180.0, performance_ratio=70.0, county_choice='Cork',
                          battery_size=0.0, charge_threshold=90.0, discharge_threshold=10.0):
    '''Calculates the new consumption data with solar PV system and battery'''
    if n_clicks is None or n_clicks == 0:
        return ''
    
    if input_hdf_data is not None:
        try:
            input_hdf = reformat_dash_hdf_store(input_hdf_data)
            # # microgen option
            if microgen_on: microgen_on=True
            else: microgen_on=False
            # microgen_on = True
            
            solar_data = get_solar_data(county_choice, input_hdf.index[0], input_hdf.index[-1], tilt, azimuth)
            if solar_capacity != 0:
                solar_power_output = calculate_solar_power_output(solar_data, solar_capacity, performance_ratio)
            else: 
                solar_power_output = calculate_solar_power_output(solar_data, 0, 0)
            
            if battery_size == 0 and solar_capacity != 0:   # only solar
                # print("Solar only")
                full_retrofit_output = Solar_only_algorithm(input_hdf, solar_power_output, microgen_on=microgen_on)
                
            elif battery_size != 0 and solar_capacity == 0: # only battery
                # print("Battery only")
                full_retrofit_output = battery_algorithm_no_solar(input_hdf,
                                                                 battery_size, charge_threshold, discharge_threshold)
            elif battery_size != 0 and solar_capacity != 0: # solar and battery
                # print("Solar and Battery")
                full_retrofit_output = battery_algorithm(input_hdf, solar_power_output,
                                                    battery_size, charge_threshold, discharge_threshold, elec_rates=None, microgen_on=microgen_on)
            else: 
                # print("No solar or battery")
                full_retrofit_output = input_hdf
                
            # Ensure we are working with a copy to avoid SettingWithCopyWarning
            full_retrofit_output = full_retrofit_output.copy()
            # print(full_retrofit_output.head(10))
            
            # Include the datetime index as a column
            full_retrofit_output.reset_index(inplace=True)
            full_retrofit_output.rename(columns={'index': 'DateTime'}, inplace=True)

            return full_retrofit_output.to_dict('records')
            
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)])

    return ''


@app.callback(
    [Output('calculate-costs-2', 'children'),
     Output('cheapest-price-store-2', 'data')],
    Input('display-button-2', 'n_clicks'),
    State('full-retrofit-output-store-2', 'data'),
    State('Solar-Size-2', 'value'),
    State('Battery-Size-2', 'value')
)
def calculate_elec_costs_2(n_clicks, new_output_data, solar_capacity_kWh, battery_capacity_kWh=0):
    '''Calculates the costs of electricity plans with new solar PV system and new battery'''
    if n_clicks is None or n_clicks == 0: 
        return '', None
    
    if new_output_data is not None:
        try:
            # reformatting solar/battery output
            new_output = reformat_dash_hdf_store(new_output_data)
            if solar_capacity_kWh > 0.1: grid_export = new_output['S2G'].sum()
            else: grid_export = 0.0

            formatted_output = reformat_retrofit_algorithm_output(new_output)
            formatted_output_pivot = convert_to_pivot(formatted_output)

            plan_options = calculate_costs(formatted_output_pivot, Elec_plans, grid_export=grid_export)

            cheapest_plan = min(plan_options, key=lambda x: x['annual_cost_no_discount'])
            cheapest_cost = cheapest_plan['annual_cost_no_discount']
            
            # Return results as an unordered list
            return format_elec_cost_output(plan_options), cheapest_cost
        
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)])

    return '', None


@app.callback(
    Output('comparison-to-base-case-2', 'children'),
    Input('display-button-2', 'n_clicks'),
    State('cheapest-price-store-0', 'data'),
    State('cheapest-price-store-2', 'data'),
    State('full-retrofit-output-store-2', 'data'),
    State('consumption-and-GHG-totals-0-store', 'data')
)
def display_consumption_and_GHG_totals_1(n_clicks, annual_cost_0, annual_cost_1, full_battery_output, base_consumption_and_GHG):
    '''Calculates the total consumption and CO2 emissions for the current setup'''
    if n_clicks is None or n_clicks == 0: return ''
    # print("Stored Full Battery Output:", full_battery_output)  # Debugging statement
    if full_battery_output is not None:
        try:
            # Reformat and calculate totals
            post_retrofit_df = reformat_dash_hdf_store(full_battery_output)
            total_consumption, total_emissions = calculate_kWh_and_GHG_totals(post_retrofit_df)
            
            # Comparison with base consumption
            base_consumption, base_emissions = base_consumption_and_GHG[0], base_consumption_and_GHG[1]
            delta_grid_import_kWh = total_consumption - base_consumption
            delta_emissions_kgCO2 = total_emissions - base_emissions
            percentage_change_consumption = (delta_grid_import_kWh / base_consumption) * 100
            percentage_change_emissions = (delta_emissions_kgCO2 / base_emissions) * 100
            
            # Cheapest price comparison
            delta_annual_cost = annual_cost_1 - annual_cost_0
            percentage_change_cost = (delta_annual_cost / annual_cost_0) * 100
            
            # Format as a table
            table_header = html.Tr([
                html.Th("Metric", style={"text-align": "left", "font-weight": "bold"}),
                html.Th("Value", style={"text-align": "left", "font-weight": "bold"}),
                html.Th("Change", style={"text-align": "left", "font-weight": "bold"})
            ])
            
            table_rows = [
                html.Tr([
                    html.Td("Total Grid Import (kWh)"),
                    html.Td(f"{total_consumption:,.2f} kWh"),
                    html.Td(
                        f"{delta_grid_import_kWh:,.2f} kWh ({percentage_change_consumption:,.2f}%)",
                        style={"color": "green" if delta_grid_import_kWh < 0 else "red"}
                    )
                ]),
                html.Tr([
                    html.Td("Total CO2 Emissions [kgCO2]"),
                    html.Td(f"{total_emissions:,.2f} kgCO2"),
                    html.Td(
                        f"{delta_emissions_kgCO2:,.2f} kgCO2 ({percentage_change_emissions:,.2f}%)",
                        style={"color": "green" if delta_emissions_kgCO2 < 0 else "red"}
                    )
                ]),
                html.Tr([
                    html.Td("Cheapest Annual Cost (€)"),
                    html.Td(f"€{annual_cost_1:,.2f}"),
                    html.Td(
                        f"€{delta_annual_cost:,.2f} ({percentage_change_cost:,.2f}%)",
                        style={"color": "green" if delta_annual_cost < 0 else "red"}
                    )
                ])
            ]
            
            return html.Div([
                html.Table(
                    [table_header] + table_rows,
                    style={"width": "100%", "border-collapse": "collapse", "margin-bottom": "20px"}
                )
            ])

        
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)])
    
    return ''

@app.callback(
    Output('consumption-pie-graph-2', 'figure'),
    Input('show-graphs-2', 'n_clicks'),
    Input('season-toggle-2', 'value'),
    State('full-retrofit-output-store-2', 'data')
)

def display_consumption_pie_1(n_clicks, season, full_retrofit_output):
    '''Displays a pie chart of the consumption split by endpoint'''
    if n_clicks is None or n_clicks == 0:
        return {}
    if full_retrofit_output is not None:
        try:
            post_retrofit_df = reformat_dash_hdf_store(full_retrofit_output)
            
            # Filter by season if specified
            if season == 'Summer':
                post_retrofit_df = post_retrofit_df[(post_retrofit_df.index.month >= 4) & (post_retrofit_df.index.month <= 9)]
            elif season == 'Winter':
                post_retrofit_df = post_retrofit_df[(post_retrofit_df.index.month <= 3) | (post_retrofit_df.index.month >= 10)]
            elif season == 'Full Year':
                post_retrofit_df = post_retrofit_df
            
            # root_node = pd.DataFrame([{'Endpoint': 'Total Consumption', 'Total': post_retrofit_df['Read Value'].sum(), 'Parent': ''}])
            totals_df = split_consumption_by_endpoint(post_retrofit_df)
            totals_df['Parent'] = 'Total Consumption'
            # consumption_endpoint_totals_df = pd.concat([root_node, totals_df])
            
            # Note: since all battery discharge goes to home (B2H), the non-grid B2H = S2B
            if 'S2B' in totals_df['Endpoint'].values and 'S2H' in totals_df['Endpoint'].values:
                total_consumption = totals_df.loc[totals_df['Endpoint'].isin(['S2H', 'S2B', 'Grid Import']), 'Total'].sum()
                pie_chart_data = totals_df[totals_df['Endpoint'].isin(['S2B', 'Grid Import', 'S2H'])]
            elif 'S2H' in totals_df['Endpoint'].values and 'S2B' not in totals_df['Endpoint'].values:
                pie_chart_data = totals_df[totals_df['Endpoint'].isin(['Grid Import', 'S2H'])]
                total_consumption = totals_df.loc[totals_df['Endpoint'].isin(['S2H', 'Grid Import']), 'Total'].sum()
            else:
                pie_chart_data = totals_df[totals_df['Endpoint'].isin(['Grid Import'])]
                total_consumption = totals_df.loc[totals_df['Endpoint'].isin(['Read Value']), 'Total'].sum()

            # Calculate Self-Consumption Ratio (SCR)
            Grid_Import_total = totals_df.loc[totals_df['Endpoint'] == 'Grid Import', 'Total'].sum()
            SCR_percentage = 100*(1 - (Grid_Import_total / total_consumption)) if total_consumption > 0 else 0
            
            fig = px.pie(
                pie_chart_data,
                names='Endpoint',
                values='Total',
                title=f'Consumption proportion by source (Grid, Solar). Self-Consumption Ratio (SCR): {SCR_percentage:.2f} %',
            )
            return fig
        
        except Exception as e:
             return {
                'data': [],
                'layout': {
                    'title': f'There was an error processing this file: {e}'
                }
             }
    
    return {}

@app.callback(
    Output('load-profile-2', 'figure'),
    Input('show-graphs-2', 'n_clicks'),
    Input('season-toggle-2', 'value'),
    State('Solar-Size-2', 'value'),
    State('Battery-Size-2', 'value'),
    # Input('load-profile-toggle-2', 'value'),
    State('full-retrofit-output-store-2', 'data')
)
def display_load_profile_1(n_clicks, season, solar_capacity, battery_capacity, full_retrofit_output):
    '''Displays a line chart of the average seasonal load profile with a toggle for new and original data'''
    if n_clicks is None or n_clicks == 0:
        return {}
    
    if full_retrofit_output is not None:
        try:
            post_retrofit_df = reformat_dash_hdf_store(full_retrofit_output)
            return plot_average_load_profile(post_retrofit_df, 
                                             solar_capacity,
                                             battery_capacity,
                                             elec_rates=extract_rates(Elec_plans, "EV Smart Electricity Discount with €50 Welcome Bonus"),
                                             season=season
                                             )

        except Exception as e:
            return {
                'data': [],
                'layout': {
                    'title': f'There was an error processing this file: {e}'
                }
            }

    return {}

@app.callback(
    Output('cost-N-years-2', 'children'),
    Input('N-years-simulation-2', 'value'),
    State('full-retrofit-output-store-2', 'data')
)
def calculate_cost_N_years_2(N_years, full_retrofit_output):
    '''Calculates the cost after N years using the 1st year price and then adds N-1 times the no discount annual cost'''
    if N_years is None or N_years <= 0:
        return ''
    
    if full_retrofit_output is not None:
        try:
            input_hdf = reformat_dash_hdf_store(full_retrofit_output)
            retrofit_hdf = input_hdf[["Grid Import"]]
            consumption_pivot = convert_to_pivot(retrofit_hdf)
            plan_options = calculate_costs(consumption_pivot, Elec_plans)
            # print(plan_options[0])
            
            # Get the cheapest plan
            cheapest_plan = plan_options[0]
            first_year_cost = cheapest_plan['annual_cost_1st_year_discount']
            subsequent_year_cost = cheapest_plan['annual_cost_no_discount']
            
            total_cost = first_year_cost + (N_years - 1) * subsequent_year_cost
            
            return html.Div([
                html.P(f"{N_years}-year cost: €{total_cost:,.2f}")
            ])
        
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)])
    
    return ''

@app.callback(
    Output('payback-period-2', 'children'),
    Input('display-button-2', 'n_clicks'),
    State('reformatted-hdf-df-store', 'data'),
    State('full-retrofit-output-store-2', 'data'),
    State('capex-store-2', 'data'),
    State('ghg-capex-store-2', 'data')
)
def display_payback_period_1(n_clicks, original_input_data, full_retrofit_output, capex, GHG_capex):
    '''Calculates and displays the payback period for the current setup (in terms of cost and GHG emissions)'''
    if n_clicks is None or n_clicks == 0:
        return ''
    if original_input_data is not None and full_retrofit_output is not None and capex is not None:
        try:
            original_hdf = reformat_dash_hdf_store(original_input_data)
            retrofit_hdf = reformat_dash_hdf_store(full_retrofit_output)
            retrofit_hdf = retrofit_hdf[["Grid Import"]]
            
            plan = Elec_plans[0]  # Assuming you want to use the first plan for payback calculation
            
            # CAPEX Payback Period
            capex_years, capex_months = payback_period(original_hdf, retrofit_hdf, plan, capex)
            
            # Solar CAPEX Payback Period
            solar_capex = capex * (GHG_capex[0] / sum(GHG_capex))  # Proportional CAPEX for solar
            solar_years, solar_months = payback_period(original_hdf, retrofit_hdf, plan, solar_capex)
            
            # Battery CAPEX Payback Period
            battery_capex = capex * (GHG_capex[1] / sum(GHG_capex))  # Proportional CAPEX for battery
            battery_years, battery_months = payback_period(original_hdf, retrofit_hdf, plan, battery_capex)
            
            # GHG Payback Period
            total_GHG_capex = sum(GHG_capex)
            solar_ghg_years, solar_ghg_months = ghg_payback_period(original_hdf, retrofit_hdf, GHG_capex[0])
            battery_ghg_years, battery_ghg_months = ghg_payback_period(original_hdf, retrofit_hdf, GHG_capex[1])
            total_ghg_years, total_ghg_months = ghg_payback_period(original_hdf, retrofit_hdf, total_GHG_capex)
            
            # Format as a table
            table_header = html.Tr(
                [html.Th("COMPONENT"), html.Th("CAPEX PAYBACK"), html.Th("CARBON PAYBACK")],
                style={'font-size': '16px', 'font-weight': 'bold', 'text-transform': 'uppercase', 'border-bottom': '2px solid black'}
            )
            table_rows = [
                html.Tr([html.Td("Solar"), html.Td(f"{solar_years} years, {solar_months} months"), html.Td(f"{solar_ghg_years} years, {solar_ghg_months} months")]),
                html.Tr([html.Td("Battery"), html.Td(f"{battery_years} years, {battery_months} months"), html.Td(f"{battery_ghg_years} years, {battery_ghg_months} months")]),
                html.Tr(
                    [html.Td("Total"), html.Td(f"{capex_years} years, {capex_months} months"), html.Td(f"{total_ghg_years} years, {total_ghg_months} months")],
                    style={'font-weight': 'bold'}
                )
            ]
            
            return html.Div([
                html.Table([table_header] + table_rows, style={'width': '100%', 'border-collapse': 'collapse', 'text-align': 'left'}),
            ])
        
        except Exception as e:
            return html.Div(['There was an error processing this file: {}'.format(e)])
    
    return ''


if __name__ == '__main__':
    app.run(debug=True)