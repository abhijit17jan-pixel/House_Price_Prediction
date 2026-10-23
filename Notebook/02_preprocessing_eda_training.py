"""
House Price Prediction System
Step 2: Preprocessing + EDA + Feature Engineering + Model Training + Evaluation

Follows Phases 3-7 of the capstone methodology.
"""

import json
import os
import sys
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore # pyrefly: ignore[untyped-import]
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE, "Model"))
from preprocessors import HousePriceFeatureTransformer  # pyrefly: ignore[missing-import]


# ============================================================
# Phase 3: Data Preprocessing
# ============================================================
df = pd.read_csv(f"{BASE}/Dataset/house_data.csv")
print("Initial shape:", df.shape)

# Remove duplicates
before = df.shape[0]
df = df.drop_duplicates()
print(f"Removed {before - df.shape[0]} duplicate rows")

# Handle missing values (numeric -> median, categorical -> mode)
missing_before = df.isnull().sum().sum()
for col in df.select_dtypes(include='number').columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])
print(f"Filled {missing_before} missing values with column medians/modes")

# Outlier treatment on Price and Area using IQR capping
for col in ["Price", "Area_sqft"]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

# ============================================================
# Phase 4: Exploratory Data Analysis
# ============================================================
# os is now imported at the top

os.makedirs(f"{BASE}/Documentation/eda_plots", exist_ok=True)

plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="Price", kde=True, color="steelblue")
plt.title("Distribution of House Prices")
plt.xlabel("Price (INR)")
plt.tight_layout()
plt.savefig(f"{BASE}/Documentation/eda_plots/price_distribution.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 5))
sns.boxplot(x=df["Area_sqft"], color="lightgreen")
plt.title("Boxplot of Area (sqft)")
plt.tight_layout()
plt.savefig(f"{BASE}/Documentation/eda_plots/area_boxplot.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 5))
plt.scatter(df["Area_sqft"], df["Price"], alpha=0.3, s=10, color="darkorange")
plt.title("Area vs Price")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (INR)")
plt.tight_layout()
plt.savefig(f"{BASE}/Documentation/eda_plots/area_vs_price.png", dpi=120)
plt.close()

plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{BASE}/Documentation/eda_plots/correlation_heatmap.png", dpi=120)
plt.close()

print("EDA plots saved to Documentation/eda_plots/")

# ============================================================
# Phase 5: Feature Engineering & Splits
# ============================================================
df["Price_per_sqft"] = df["Price"] / df["Area_sqft"]  # for reference/analysis only, not used as input

# We define the raw features that we will pass into the pipeline.
raw_features = [
    "Area_sqft", "Bedrooms", "Bathrooms", "Balconies", "Floor_No", "Total_Floors",
    "Age_of_Property", "Parking_Spaces", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Score", "Has_Garden", "Has_Gym", "Has_Swimming_Pool",
    "Has_Security", "Has_Power_Backup", "Distance_to_Metro_km", "Gated_Community",
    "Monthly_Maintenance_Charges", "City", "Locality_Tier", "Property_Type",
    "Furnishing", "Facing_Direction", "Water_Supply_Type", "Construction_Quality"
]

categorical_cols = ["City", "Locality_Tier", "Property_Type", "Furnishing", "Facing_Direction", "Water_Supply_Type", "Construction_Quality"]

# Reconstruct label encoders for compatibility/metadata
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])  # pyrefly: ignore[unsupported-operation]
    label_encoders[col] = le

X = df[raw_features]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# Phase 6: Model Building (using scikit-learn Pipelines)
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
    ],
    remainder="passthrough"
)

lr_pipeline = Pipeline([
    ("feature_engineering", HousePriceFeatureTransformer()),
    ("preprocessing", preprocessor),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

rf_pipeline = Pipeline([
    ("feature_engineering", HousePriceFeatureTransformer()),
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1))
])

xgb_pipeline = Pipeline([
    ("feature_engineering", HousePriceFeatureTransformer()),
    ("preprocessing", preprocessor),
    ("regressor", XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.06, random_state=42, verbosity=0))
])

pipelines = {
    "Linear Regression": lr_pipeline,
    "Random Forest": rf_pipeline,
    "XGBoost": xgb_pipeline,
}

results = {}
trained_pipelines = {}

for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results[name] = {"MAE": round(mae, 2), "MSE": round(mse, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
    trained_pipelines[name] = pipeline
    print(f"{name}: MAE={mae:.0f} RMSE={rmse:.0f} R2={r2:.4f}")

# ============================================================
# Phase 7: Model Evaluation - pick the best model by R2
# ============================================================
best_model_name = max(results, key=lambda k: results[k]["R2"])
best_pipeline = trained_pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

# Comparative bar chart of R2 scores
plt.figure(figsize=(7, 5))
names = list(results.keys())
r2_scores = [results[n]["R2"] for n in names]
sns.barplot(x=names, y=r2_scores, hue=names, legend=False, palette="viridis")
plt.title("Model Comparison (R2 Score)")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.savefig(f"{BASE}/Documentation/eda_plots/model_comparison.png", dpi=120)
plt.close()

# ============================================================
# Save artifacts
# ============================================================
joblib.dump(best_pipeline, f"{BASE}/Model/best_model.pkl")

# Extract the StandardScaler for backward compatibility/reports
scaler = lr_pipeline.named_steps["scaler"]
joblib.dump(scaler, f"{BASE}/Model/scaler.pkl")
joblib.dump(label_encoders, f"{BASE}/Model/label_encoders.pkl")

# Save ordered list of features used internally by the pipeline for references
feature_cols = [
    "Area_sqft", "Bedrooms", "Bathrooms", "Balconies", "Floor_No", "Total_Floors",
    "Age_of_Property", "Parking_Spaces", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Score", "Has_Garden", "Has_Gym", "Has_Swimming_Pool",
    "Has_Security", "Has_Power_Backup", "Total_Rooms", "Is_New_Property",
    "Amenity_Score", "Floor_Ratio",
    "Distance_to_Metro_km", "Gated_Community", "Monthly_Maintenance_Charges",
    "City_enc", "Locality_Tier_enc", "Property_Type_enc", "Furnishing_enc",
    "Facing_Direction_enc", "Water_Supply_Type_enc", "Construction_Quality_enc",
]
joblib.dump(feature_cols, f"{BASE}/Model/feature_cols.pkl")

with open(f"{BASE}/Model/model_info.json", "w") as f:
    json.dump({
        "best_model": best_model_name,
        "uses_scaled_input": False, # Handled internally by the pipeline now
        "results": results,
        "feature_cols": feature_cols,
        "categorical_options": {c: list(label_encoders[c].classes_) for c in categorical_cols},
    }, f, indent=2)

df.to_csv(f"{BASE}/Dataset/house_data_cleaned.csv", index=False)

print("\nAll models and preprocessing pipelines saved to Model/")
print(json.dumps(results, indent=2))
