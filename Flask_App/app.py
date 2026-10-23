"""
House Price Prediction System
Phase 8: Model Deployment (Flask)

Serves a public landing page, login/registration, and — once authenticated —
a form-based valuation dashboard where a user enters property details and
receives a predicted price from the best-performing trained model (XGBoost).
"""

import json
import os
import sys
import sqlite3
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "Model")
DB_PATH = os.path.join(BASE_DIR, "users.db")

# Add MODEL_DIR to sys.path so preprocessors class can be unpickled by joblib
sys.path.append(MODEL_DIR)
from preprocessors import HousePriceFeatureTransformer  # pyrefly: ignore[missing-import]


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "capstone-house-price-dev-secret")

# ---- Load trained unified pipeline once at startup ----
model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))

with open(os.path.join(MODEL_DIR, "model_info.json")) as f:
    model_info = json.load(f)

BEST_MODEL_NAME = model_info["best_model"]
CATEGORICAL_OPTIONS = model_info["categorical_options"]
METRICS = model_info["results"]


# ============================================================
# User database (SQLite) — registration & login
# ============================================================
def get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    db.close()


init_db()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped





def build_raw_dataframe(form):
    """Turn raw form input into a DataFrame of raw features matching model expectations."""
    row = {
        "Area_sqft": float(form["area_sqft"]),
        "Bedrooms": int(form["bedrooms"]),
        "Bathrooms": int(form["bathrooms"]),
        "Balconies": int(form["balconies"]),
        "Floor_No": int(form["floor_no"]),
        "Total_Floors": int(form["total_floors"]),
        "Age_of_Property": int(form["age"]),
        "Parking_Spaces": int(form["parking"]),
        "Nearby_Schools": int(form["schools"]),
        "Nearby_Hospitals": int(form["hospitals"]),
        "Public_Transport_Score": int(form["transport_score"]),
        "Has_Garden": 1 if form.get("garden") == "on" else 0,
        "Has_Gym": 1 if form.get("gym") == "on" else 0,
        "Has_Swimming_Pool": 1 if form.get("pool") == "on" else 0,
        "Has_Security": 1 if form.get("security") == "on" else 0,
        "Has_Power_Backup": 1 if form.get("power_backup") == "on" else 0,
        "Distance_to_Metro_km": float(form["distance_to_metro"]),
        "Gated_Community": 1 if form.get("gated_community") == "on" else 0,
        "Monthly_Maintenance_Charges": float(form["maintenance_charges"]),
        "City": form["city"],
        "Locality_Tier": form["locality_tier"],
        "Property_Type": form["property_type"],
        "Furnishing": form["furnishing"],
        "Facing_Direction": form["facing_direction"],
        "Water_Supply_Type": form["water_supply_type"],
        "Construction_Quality": form["construction_quality"]
    }
    return pd.DataFrame([row])


@app.route("/")
def home():
    return render_template("home.html", best_model=BEST_MODEL_NAME, metrics=METRICS)


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        error = None
        if not name or not email or not password:
            error = "All fields are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters long."
        elif password != confirm_password:
            error = "Passwords do not match."

        if error is None:
            db = get_db()
            existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if existing:
                error = "An account with that email already exists."
            else:
                db.execute(
                    "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                    (name, email, generate_password_hash(password)),
                )
                db.commit()
                db.close()
                return redirect(url_for("login"))
            db.close()

        return render_template("register.html", error=error)

    return render_template("register.html", error=None)


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        db.close()

        if user is None or not check_password_hash(user["password_hash"], password):
            return render_template("login.html", error="Incorrect email or password.")

        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        next_path = request.args.get("next") or url_for("dashboard")
        return redirect(next_path)

    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        cities=CATEGORICAL_OPTIONS["City"],
        tiers=CATEGORICAL_OPTIONS["Locality_Tier"],
        ptypes=CATEGORICAL_OPTIONS["Property_Type"],
        furnishings=CATEGORICAL_OPTIONS["Furnishing"],
        facing_directions=CATEGORICAL_OPTIONS["Facing_Direction"],
        water_supply_types=CATEGORICAL_OPTIONS["Water_Supply_Type"],
        construction_qualities=CATEGORICAL_OPTIONS["Construction_Quality"],
        best_model=BEST_MODEL_NAME,
        metrics=METRICS,
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        # Backend validation: check if floor number exceeds total floors
        floor_no = int(request.form.get("floor_no", 0))
        total_floors = int(request.form.get("total_floors", 0))
        if floor_no > total_floors:
            return jsonify({
                "success": False,
                "error": "Floor number cannot exceed total floors in the building."
            }), 400

        X = build_raw_dataframe(request.form)
        prediction = float(model.predict(X)[0])
        prediction = max(prediction, 0)
        return jsonify({
            "success": True,
            "prediction": round(prediction),
            "prediction_formatted": f"₹ {prediction:,.0f}",
            "model_used": BEST_MODEL_NAME,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/metrics")
def metrics():
    return jsonify(METRICS)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
