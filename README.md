# House Price Prediction System

**AIML Summer Internship 2026 — IIHMF, MNNIT Allahabad, Prayagraj**
Capstone Project · Project 2: House Price Prediction System

Predicts house prices based on area, location, number of rooms, age of
property, and available amenities, using a complete ML lifecycle from data
generation through Flask deployment.

## Folder Structure

```
HousePricePrediction/
│
├── Dataset/
│   ├── house_data.csv            # Raw generated dataset (3,015 rows)
│   └── house_data_cleaned.csv    # After preprocessing + feature engineering
│
├── Notebook/
│   ├── 01_generate_dataset.py         # Synthetic dataset generation
│   └── 02_preprocessing_eda_training.py  # Preprocessing, EDA, FE, training, evaluation
│
├── Model/
│   ├── best_model.pkl             # Best performing model (XGBoost)
│   ├── scaler.pkl                 # StandardScaler (used by Linear Regression)
│   ├── label_encoders.pkl         # LabelEncoders for categorical fields
│   ├── feature_cols.pkl           # Ordered list of model input features
│   └── model_info.json            # Metrics + dropdown options for the app
│
├── Flask_App/
│   ├── app.py                     # Flask application (Phase 8: Deployment) + auth
│   ├── users.db                   # SQLite user store (created automatically on first run)
│   ├── templates/
│   │   ├── home.html              # Public landing page
│   │   ├── login.html             # Login page
│   │   ├── register.html          # Registration page
│   │   └── dashboard.html         # Valuation form + live prediction (login required)
│   └── static/style.css, script.js
│
├── Documentation/
│   └── eda_plots/                 # Histograms, boxplots, scatter, heatmap, model comparison
│
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt

# 1. Regenerate the dataset (optional — house_data.csv already included)
python Notebook/01_generate_dataset.py

# 2. Re-run preprocessing + train models (optional — Model/ already populated)
python Notebook/02_preprocessing_eda_training.py

# 3. Launch the web app
cd Flask_App
python app.py
```

Then open **http://127.0.0.1:5000** in a browser. You'll land on the home
page — register a new account, log in, and you'll be taken to the
valuation dashboard. The `/dashboard` and `/predict` routes require login;
visiting them while signed out redirects to the login page.

## Methodology (per capstone guidelines)

| Phase | What was done |
|---|---|
| 1. Problem Understanding | Predict house price from structural, locational and amenity features |
| 2. Data Collection | Synthetic dataset generated with realistic, non-trivial price relationships (3,015 rows, 21 raw attributes) |
| 3. Data Preprocessing | Missing-value imputation (median), duplicate removal, IQR outlier capping on Price/Area |
| 4. EDA | Histogram, boxplot, scatter plot, correlation heatmap (see `Documentation/eda_plots/`) |
| 5. Feature Engineering | Total_Rooms, Is_New_Property, Amenity_Score, Floor_Ratio, label-encoded categoricals |
| 6. Model Building | Linear Regression, Random Forest, XGBoost |
| 7. Model Evaluation | MAE, MSE, RMSE, R² — see table below |
| 8. Deployment | Flask app with a form-based UI, live prediction via `/predict` |

## Model Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 29,22,876 | 39,68,958 | 0.6586 |
| Random Forest | 7,60,176 | 10,27,407 | 0.9771 |
| **XGBoost (deployed)** | **6,89,287** | **9,89,953** | **0.9788** |

XGBoost was selected as the deployed model based on the highest R² score
on the held-out test set (20% split).

## Notes

- The dataset is synthetically generated (not sourced from Kaggle) with
  deliberately realistic pricing logic — city-wise base rate per sqft,
  locality tier and property type multipliers, age-based depreciation,
  and amenity premiums — plus injected missing values and duplicates to
  mirror real-world data quality issues for the preprocessing phase.
- All three required regression models are trained and compared; the
  best one is deployed, but `Model/model_info.json` retains metrics for
  all three for the report and presentation.
