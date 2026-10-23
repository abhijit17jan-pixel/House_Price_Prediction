"""
House Price Prediction System
Step 1: Synthetic Dataset Generation

Generates a realistic housing dataset with the attributes mentioned in the
capstone brief: area, location, number of rooms, age of property, and
available amenities.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)

N = 3000

cities = ["Prayagraj", "Lucknow", "Kanpur", "Varanasi", "Noida"]
city_base_price_per_sqft = {
    "Prayagraj": 3200,
    "Lucknow": 4200,
    "Kanpur": 3600,
    "Varanasi": 3400,
    "Noida": 6500,
}

locality_tiers = ["Low", "Medium", "High"]  # locality desirability within a city
tier_multiplier = {"Low": 0.85, "Medium": 1.0, "High": 1.35}

furnishing_types = ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]
furnishing_bonus = {"Unfurnished": 0, "Semi-Furnished": 120000, "Fully-Furnished": 300000}

property_types = ["Apartment", "Independent House", "Villa"]
property_type_multiplier = {"Apartment": 1.0, "Independent House": 1.08, "Villa": 1.35}

data = {
    "Area_sqft": np.random.randint(450, 4500, N),
    "Bedrooms": np.random.randint(1, 6, N),
    "Bathrooms": np.random.randint(1, 5, N),
    "Balconies": np.random.randint(0, 4, N),
    "Floor_No": np.random.randint(0, 25, N),
    "Total_Floors": np.random.randint(1, 30, N),
    "Age_of_Property": np.random.randint(0, 40, N),
    "City": np.random.choice(cities, N),
    "Locality_Tier": np.random.choice(locality_tiers, N, p=[0.3, 0.45, 0.25]),
    "Property_Type": np.random.choice(property_types, N, p=[0.55, 0.30, 0.15]),
    "Furnishing": np.random.choice(furnishing_types, N, p=[0.4, 0.4, 0.2]),
    "Parking_Spaces": np.random.randint(0, 4, N),
    "Nearby_Schools": np.random.randint(0, 10, N),
    "Nearby_Hospitals": np.random.randint(0, 8, N),
    "Public_Transport_Score": np.random.randint(1, 11, N),  # 1-10 accessibility score
    "Has_Garden": np.random.choice([0, 1], N, p=[0.6, 0.4]),
    "Has_Gym": np.random.choice([0, 1], N, p=[0.65, 0.35]),
    "Has_Swimming_Pool": np.random.choice([0, 1], N, p=[0.8, 0.2]),
    "Has_Security": np.random.choice([0, 1], N, p=[0.35, 0.65]),
    "Has_Power_Backup": np.random.choice([0, 1], N, p=[0.4, 0.6]),
    # 6 new features added per recommendation
    "Facing_Direction": np.random.choice(["East", "West", "North", "South", "North-East"], N, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
    "Water_Supply_Type": np.random.choice(["Municipal", "Borewell", "Both"], N, p=[0.4, 0.3, 0.3]),
    "Distance_to_Metro_km": np.round(np.random.exponential(scale=3.0, size=N).clip(0.2, 15.0), 1),
    "Gated_Community": np.random.choice([0, 1], N, p=[0.6, 0.4]),
    "Construction_Quality": np.random.choice(["Standard", "Premium", "Ultra-Luxury"], N, p=[0.6, 0.3, 0.1]),
}

df = pd.DataFrame(data)

# Compute Monthly Maintenance Charges based on features
base_maintenance = df["Area_sqft"] * 1.5
amenity_charge = (df["Has_Garden"] + df["Has_Gym"] + df["Has_Swimming_Pool"] + df["Has_Security"] + df["Has_Power_Backup"]) * 350
gated_charge = df["Gated_Community"] * 1200
type_charge = df["Property_Type"].map({"Apartment": 500, "Independent House": 0, "Villa": 2000})
df["Monthly_Maintenance_Charges"] = (base_maintenance + amenity_charge + gated_charge + type_charge) * np.random.uniform(0.9, 1.1, N)
df["Monthly_Maintenance_Charges"] = df["Monthly_Maintenance_Charges"].round(-1).astype(int)

# Ensure floor number never exceeds total floors
df["Floor_No"] = df.apply(lambda r: min(r["Floor_No"], r["Total_Floors"]), axis=1)

# ---------- Price generation (with realistic, non-trivial relationships) ----------
base_price_per_sqft = df["City"].map(city_base_price_per_sqft)
locality_mult = df["Locality_Tier"].map(tier_multiplier)
type_mult = df["Property_Type"].map(property_type_multiplier)

price = df["Area_sqft"] * base_price_per_sqft * locality_mult * type_mult

# Apply construction quality multiplier (Standard: 1.0, Premium: 1.15, Ultra-Luxury: 1.30)
quality_mult = df["Construction_Quality"].map({"Standard": 1.0, "Premium": 1.15, "Ultra-Luxury": 1.30})
price *= quality_mult

# Bedrooms/bathrooms add value
price += df["Bedrooms"] * 90_000
price += df["Bathrooms"] * 60_000
price += df["Balconies"] * 25_000
price += df["Parking_Spaces"] * 70_000

# Facing Direction bonus (East/North-East get premiums)
facing_bonus = df["Facing_Direction"].map({"East": 80_000, "North-East": 100_000, "North": 40_000, "West": 0, "South": -20_000})
price += facing_bonus

# Water Supply premium
water_bonus = df["Water_Supply_Type"].map({"Municipal": 60_000, "Borewell": 0, "Both": 120_000})
price += water_bonus

# Distance to Metro penalty (closer metro gives premium up to 300,000 INR)
metro_premium = np.maximum(300_000 - df["Distance_to_Metro_km"] * 25_000, 0)
price += metro_premium

# Gated community flat premium
price += df["Gated_Community"] * 150_000

# Monthly maintenance premium (capitalized value of maintenance standards)
price += df["Monthly_Maintenance_Charges"] * 120

# Higher floors in tall buildings often cost a bit more (view premium), ground floor houses no penalty
price += np.where(df["Total_Floors"] > 5, df["Floor_No"] * 4_000, 0)

# Depreciation with age
price -= df["Age_of_Property"] * 8_500

# Amenities
price += df["Nearby_Schools"] * 5_000
price += df["Nearby_Hospitals"] * 4_000
price += df["Public_Transport_Score"] * 9_000
price += df["Has_Garden"] * 80_000
price += df["Has_Gym"] * 60_000
price += df["Has_Swimming_Pool"] * 150_000
price += df["Has_Security"] * 50_000
price += df["Has_Power_Backup"] * 40_000

# Furnishing bonus
price += df["Furnishing"].map(furnishing_bonus)

# Add market noise
price *= np.random.normal(1.0, 0.06, N)

# Floor price at a sane minimum
price = np.maximum(price, 300_000)

df["Price"] = price.round(-3).astype(int)  # round to nearest thousand

# ---------- Inject a small amount of realism: missing values & duplicates ----------
for col in ["Bathrooms", "Balconies", "Nearby_Schools", "Public_Transport_Score", "Distance_to_Metro_km", "Facing_Direction", "Monthly_Maintenance_Charges"]:
    idx = np.random.choice(df.index, size=int(0.02 * N), replace=False)
    df.loc[idx, col] = np.nan

dup_rows = df.sample(15, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv(os.path.join(BASE_DIR, "Dataset", "house_data.csv"), index=False)
print("Dataset generated:", df.shape)
print(df.head())
