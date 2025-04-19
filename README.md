# BRC - Battery Retrofit Comparison

**BRC** is an open-source Dash web application that simulates the impact of adding rooftop solar PV (0–10 kWp) and a solar battery system (0–15 kWh) to a household electricity profile in Ireland. It was created as an Energy Engineering final year university project at University College Cork.

Inspired by [Dominic O'Gallaghoire's kilowatt.ie](https://kilowatt.ie/)

---

## Inputs

### Base Case
- Urban/Rural selector (default: Urban)  
  Affects standing charge in the electricity plan
- County (default: Cork)  
  Affects solar irradiance profile
- Consumption Profile (CSV upload)  
  Must include a `Read Value` column in kW
  Datetime column (named: `Read Date and End Time`) must have format: `%d-%m-%Y %H:%M`

### Setup 1 & Setup 2 (Retrofit Comparisons)
- **Solar PV Settings**:
  - Size (0–10 kWp)
  - Tilt (°), default: 30°
  - Azimuth (°), default: 180 (South-facing)
  - Performance Ratio (%), default: 70%
- **Battery Settings**:
  - Size (0–15 kWh)
  - Discharge Threshold (%), default: 10%
  - Charge Threshold (%), default: 90%

---

## Outputs

### Base Case
- Annual electricity costs (€)
- Annual energy consumption (kWh)
- Annual greenhouse gas emissions (kgCO₂)

### Retrofitted Setups
- Embedded Costs and Carbon:
  - Solar, Battery, and Total CAPEX (€)
  - Solar, Battery, and Total embedded carbon (kgCO₂)
- Cheapest electricity plan options (top 3)
- Comparison to base case for cost, consumption, and GHG emissions
- Payback Periods:
  - Financial (solar, battery, combined)
  - Carbon (solar, battery, combined)
- Load Profile Graphs (by season)
- Self-consumption ratios (S2H and S2B)

---

## Limitations & Assumptions

### Known Interface Issues
- "Comparison to Base Case" requires two clicks of the Display button to update correctly when parameters change.

### Financial and Carbon Estimates (Simplified)
- **Solar CAPEX**: 1750 €/kWp  
  **Carbon footprint**: 1440 kgCO₂/kWp  
  Source: Renewable Energy Hub
- **Battery CAPEX**: 1100 € + 400 €/kWh  
  **Carbon footprint**: 200 kgCO₂/kWh  
  (100 module, 50 inverter, 60 other)

### Microgeneration Export
- Assumes fixed export rate of 19.5c/kWh
- Exports not currently deducted from CO₂ calculations

### Price Calculation Method
- Based on a pivot table using 30-minute resolution
- A more efficient approach would be TOU period binning

### Battery Logic
- No degradation modeled
- 90% round-trip efficiency assumed

### Time-of-Use Tariff Windows
- Currently use default windows; do not reflect all supplier-specific TOU hours

---

## File Format Requirements

Uploaded CSVs should:
- Use a datetime index at 30-minute resolution
- Include a column named `Read Value` for energy usage
- Datetime column (named: `Read Date and End Time`) must have format: `%d-%m-%Y %H:%M`

---

## Project Status

While this release is finished for use in the project, it is completely open-source and available for anyone to use or improve.