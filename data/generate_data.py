"""
Data generation script for Agriculture & Climate Change Capstone Project.
Produces realistic, multi-variable agricultural and climatic data across multiple countries and crop types.
"""
import os
import numpy as np
import pandas as pd

def generate_agri_climate_dataset(n_samples=2500, random_state=42):
    np.random.seed(random_state)
    
    countries_regions = {
        'United States': 'North America',
        'Canada': 'North America',
        'Brazil': 'South America',
        'Argentina': 'South America',
        'India': 'South Asia',
        'China': 'East Asia',
        'Nigeria': 'Sub-Saharan Africa',
        'Kenya': 'Sub-Saharan Africa',
        'France': 'Europe',
        'Germany': 'Europe',
        'Australia': 'Oceania',
        'Indonesia': 'Southeast Asia'
    }
    
    crops = ['Wheat', 'Rice', 'Maize', 'Soybeans', 'Barley', 'Coffee']
    
    country_list = list(countries_regions.keys())
    
    records = []
    for _ in range(n_samples):
        country = np.random.choice(country_list)
        region = countries_regions[country]
        crop = np.random.choice(crops)
        year = np.random.randint(2000, 2026)
        
        # Climate trends (slight warming over years)
        warming_trend = (year - 2000) * 0.04
        
        # Base regional climates
        if region in ['Sub-Saharan Africa', 'South Asia', 'Southeast Asia']:
            base_temp = np.random.normal(26.5 + warming_trend, 2.5)
            base_rainfall = np.random.normal(1200, 350)
            irrigation_pct = np.random.uniform(15, 65)
            fertilizer = np.random.normal(95, 25)
        elif region in ['South America']:
            base_temp = np.random.normal(23.0 + warming_trend, 2.0)
            base_rainfall = np.random.normal(1400, 300)
            irrigation_pct = np.random.uniform(25, 75)
            fertilizer = np.random.normal(140, 35)
        elif region in ['North America', 'Europe']:
            base_temp = np.random.normal(13.5 + warming_trend, 3.0)
            base_rainfall = np.random.normal(850, 200)
            irrigation_pct = np.random.uniform(40, 90)
            fertilizer = np.random.normal(180, 40)
        else: # Oceania
            base_temp = np.random.normal(20.0 + warming_trend, 3.5)
            base_rainfall = np.random.normal(650, 220)
            irrigation_pct = np.random.uniform(30, 80)
            fertilizer = np.random.normal(130, 30)
            
        temp = max(5.0, base_temp)
        rainfall = max(150.0, base_rainfall)
        fertilizer = max(10.0, fertilizer)
        soil_ph = np.clip(np.random.normal(6.5, 0.6), 4.8, 8.4)
        extreme_weather_days = int(max(0, np.random.poisson(lam=12 + (year - 2000)*0.5 + (temp - 20)*0.8)))
        pesticide_kg = max(0.5, np.random.normal(3.2, 1.1))
        co2_emissions_mt = max(5.0, np.random.normal(45.0 + (year - 2000)*1.2, 12.0))
        
        # Calculate realistic crop yield with agronomic and climate non-linear interaction
        crop_base_yield = {
            'Wheat': 3.8,
            'Rice': 4.6,
            'Maize': 6.2,
            'Soybeans': 2.9,
            'Barley': 3.4,
            'Coffee': 1.8
        }[crop]
        
        # Temperature stress penalty (optimal ~ 18-24°C depending on crop)
        opt_temp = 20.0 if crop in ['Wheat', 'Barley'] else 26.0 if crop in ['Rice', 'Coffee'] else 23.0
        temp_penalty = -0.12 * ((temp - opt_temp) ** 2) / 10.0
        
        # Rainfall effect (optimal ~ 800-1400mm)
        rain_effect = 0.8 * (np.log(rainfall) / np.log(1000))
        
        # Fertilizer boost (diminishing returns)
        fert_effect = 1.2 * (np.sqrt(fertilizer) / 12.0)
        
        # Irrigation mitigation of extreme weather
        extreme_weather_penalty = -0.04 * extreme_weather_days * (1.0 - (irrigation_pct / 130.0))
        
        # Soil pH suitability (optimum ~ 6.5)
        ph_effect = -0.4 * abs(soil_ph - 6.5)
        
        noise = np.random.normal(0, 0.35)
        
        yield_tons_ha = crop_base_yield + temp_penalty + rain_effect + fert_effect + extreme_weather_penalty + ph_effect + noise
        yield_tons_ha = max(0.4, round(yield_tons_ha, 2))
        
        # Economic loss
        expected_yield = crop_base_yield * 1.2
        loss_pct = max(0.0, (expected_yield - yield_tons_ha) / expected_yield)
        economic_loss = round(loss_pct * (extreme_weather_days * 2.4 + np.random.uniform(5, 30)), 2)
        
        # Risk level classification
        if extreme_weather_days > 28 or temp > 29.5 or rainfall < 400:
            risk_level = 'Severe'
        elif extreme_weather_days > 18 or temp > 25.0:
            risk_level = 'High'
        elif extreme_weather_days > 10:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'
            
        records.append({
            'Year': year,
            'Country': country,
            'Region': region,
            'Crop_Type': crop,
            'Avg_Temperature_C': round(temp, 1),
            'Annual_Rainfall_mm': round(rainfall, 1),
            'Extreme_Weather_Days': extreme_weather_days,
            'Soil_pH': round(soil_ph, 2),
            'Fertilizer_Usage_kg_ha': round(fertilizer, 1),
            'Irrigation_Access_Pct': round(irrigation_pct, 1),
            'Pesticide_Usage_kg_ha': round(pesticide_kg, 2),
            'CO2_Emissions_MT': round(co2_emissions_mt, 1),
            'Yield_Tons_Per_Ha': yield_tons_ha,
            'Economic_Loss_Million_USD': economic_loss,
            'Climate_Risk_Level': risk_level
        })
        
    df = pd.DataFrame(records)
    
    # Introduce small realistic missingness for data cleaning demonstration in Capstone
    missing_mask = np.random.rand(*df[['Soil_pH', 'Fertilizer_Usage_kg_ha', 'Irrigation_Access_Pct']].shape) < 0.03
    df_missing_sub = df[['Soil_pH', 'Fertilizer_Usage_kg_ha', 'Irrigation_Access_Pct']].mask(missing_mask)
    df[['Soil_pH', 'Fertilizer_Usage_kg_ha', 'Irrigation_Access_Pct']] = df_missing_sub
    
    return df

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'agri_climate_data.csv')
    df = generate_agri_climate_dataset()
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} records saved to {csv_path}")
