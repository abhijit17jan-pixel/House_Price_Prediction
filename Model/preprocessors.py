import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class HousePriceFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # The list of raw features expected from Flask app or data generation
        self.raw_cols = [
            "Area_sqft", "Bedrooms", "Bathrooms", "Balconies", "Floor_No", "Total_Floors",
            "Age_of_Property", "Parking_Spaces", "Nearby_Schools", "Nearby_Hospitals",
            "Public_Transport_Score", "Has_Garden", "Has_Gym", "Has_Swimming_Pool",
            "Has_Security", "Has_Power_Backup", "Distance_to_Metro_km", "Gated_Community",
            "Monthly_Maintenance_Charges", "City", "Locality_Tier", "Property_Type",
            "Furnishing", "Facing_Direction", "Water_Supply_Type", "Construction_Quality"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check input type and convert to DataFrame if necessary
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list) and all(isinstance(i, dict) for i in X):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.raw_cols)

        # Make copy to avoid modify warnings
        X = X.copy()

        # Type conversion & filling missing/null values if any (to handle type mismatches gracefully)
        for col in ["Area_sqft", "Distance_to_Metro_km", "Monthly_Maintenance_Charges"]:
            if col in X:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        for col in ["Bedrooms", "Bathrooms", "Balconies", "Floor_No", "Total_Floors", "Age_of_Property", 
                    "Parking_Spaces", "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Score", 
                    "Has_Garden", "Has_Gym", "Has_Swimming_Pool", "Has_Security", "Has_Power_Backup", "Gated_Community"]:
            if col in X:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)

        # 1. Feature Engineering
        X["Total_Rooms"] = X["Bedrooms"] + X["Bathrooms"]
        X["Is_New_Property"] = (X["Age_of_Property"] <= 5).astype(int)
        X["Amenity_Score"] = (
            X["Has_Garden"] + X["Has_Gym"] + X["Has_Swimming_Pool"] +
            X["Has_Security"] + X["Has_Power_Backup"]
        )
        X["Floor_Ratio"] = X["Floor_No"] / X["Total_Floors"].replace(0, 1)

        # Reorder features exactly as expected by the encoder and downstream steps
        # Category features should be at the end, and numeric features in the specified training order.
        ordered_cols = [
            "Area_sqft", "Bedrooms", "Bathrooms", "Balconies", "Floor_No", "Total_Floors",
            "Age_of_Property", "Parking_Spaces", "Nearby_Schools", "Nearby_Hospitals",
            "Public_Transport_Score", "Has_Garden", "Has_Gym", "Has_Swimming_Pool",
            "Has_Security", "Has_Power_Backup", "Total_Rooms", "Is_New_Property",
            "Amenity_Score", "Floor_Ratio",
            "Distance_to_Metro_km", "Gated_Community", "Monthly_Maintenance_Charges",
            "City", "Locality_Tier", "Property_Type", "Furnishing",
            "Facing_Direction", "Water_Supply_Type", "Construction_Quality"
        ]

        # Verify that all ordered columns are present, fill missing ones if needed
        for c in ordered_cols:
            if c not in X:
                X[c] = np.nan

        return X[ordered_cols]
