"""
train_model.py
==============
Trains a supervised machine learning model (Random Forest) to predict 
anxiety levels based on raw wearable data AND stress levels.

Features:
- Stress (survey)
- HRV (RMSSD)
- Sleep (overall score, deep sleep)
- Steps (total daily steps)

Includes burnout risk notification logic: flagging a user when their 
predicted anxiety remains high for 7 consecutive days.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

ANXIETY_THRESHOLD = 65  # Threshold to consider anxiety "high" for burnout risk

# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def _to_utc(s: pd.Series) -> pd.Series:
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")

def load_daily_questions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = _to_utc(pd.to_datetime(df["timeStampStart"], unit="s").dt.normalize())
    
    # Keep stress as a feature, anxiety as the target
    df["stress"] = df["stress"].clip(0, 100)
    df["anxiety"] = df["anxiety"].clip(0, 100)
    return df.groupby("date", as_index=False)[["stress", "anxiety"]].mean()

def load_hrv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["date"] = _to_utc(df["timestamp"].dt.normalize())
    df = df[df["coverage"] >= 0.7]
    daily = df.groupby("date", as_index=False)["rmssd"].mean()
    return daily[["date", "rmssd"]]

def load_steps(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = _to_utc(df["timestamp"].dt.normalize())
    daily = df.groupby("date", as_index=False)["steps"].sum()
    return daily[["date", "steps"]]

def load_sleep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = _to_utc(df["timestamp"].dt.normalize())
    daily = df.groupby("date", as_index=False).agg(
        sleep_score=("overall_score", "mean"),
        deep_sleep_min=("deep_sleep_in_minutes", "mean"),
    )
    return daily[["date", "sleep_score", "deep_sleep_min"]]

# ──────────────────────────────────────────────────────────────────────────────
# Feature Extractor
# ──────────────────────────────────────────────────────────────────────────────

def process_user_data(uid: int, user_dir: Path) -> pd.DataFrame | None:
    files = {
        "q":     user_dir / "daily_questions.csv",
        "hrv":   user_dir / "hrv.csv",
        "steps": user_dir / "steps.csv",
        "sleep": user_dir / "sleep.csv",
    }
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        return None

    try:
        q  = load_daily_questions(files["q"])
        h  = load_hrv(files["hrv"])
        s  = load_steps(files["steps"])
        sl = load_sleep(files["sleep"])
    except Exception as exc:
        return None

    for df in (q, h, s, sl):
        df["date"] = _to_utc(df["date"])

    merged = q.merge(h, on="date", how="left")
    merged = merged.merge(s, on="date", how="left")
    merged = merged.merge(sl, on="date", how="left")
    
    # Handle missing values
    merged["sleep_score"] = merged["sleep_score"].ffill(limit=2).fillna(merged["sleep_score"].median())
    merged["deep_sleep_min"] = merged["deep_sleep_min"].ffill(limit=2).fillna(merged["deep_sleep_min"].median())
    merged["rmssd"]       = merged["rmssd"].interpolate(limit=2).fillna(merged["rmssd"].median())
    merged["steps"]       = merged["steps"].fillna(merged["steps"].median())

    # Drop any rows that still have NaNs
    merged = merged.dropna()

    merged["user_id"] = uid
    merged["month"] = merged["date"].dt.month
    
    return merged

# ──────────────────────────────────────────────────────────────────────────────
# Main Training Routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Anxiety Prediction Model")
    default_data_dir = Path(__file__).resolve().parent.parent / "SSAQS dataset"
    parser.add_argument("--data-dir", default=str(default_data_dir), help="Path to the 'SSAQS dataset' folder")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    user_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: int(d.name))
    
    print(f"Extracting data from {len(user_dirs)} participants...")
    
    all_data_frames = []
    for user_dir in user_dirs:
        uid = int(user_dir.name)
        df = process_user_data(uid, user_dir)
        if df is not None and not df.empty:
            all_data_frames.append(df)

    full_dataset = pd.concat(all_data_frames, ignore_index=True)
    print(f"Total labeled daily datapoints: {len(full_dataset)}")

    # Features and Target - Strictly using raw data
    feature_cols = [
        "stress",           
        "rmssd", 
        "steps",  
        "sleep_score", "deep_sleep_min"
    ]
    target = "anxiety"
    
    print("\nPreparing 70/30 Train/Test split (shuffled and stratified by month)...")
    
    # Stratify by month to ensure a balanced mix and avoid block consecutive months
    months = full_dataset['date'].dt.month
    
    X = full_dataset[feature_cols]
    y = full_dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.30, 
        random_state=42, 
        shuffle=True,
        stratify=months
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    # Normalize all features before feeding to the model
    print("\nNormalizing all features (including stress)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("Training Random Forest Regressor to predict Anxiety...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Classification Metrics (Threshold > 65)
    y_test_binary = y_test > ANXIETY_THRESHOLD
    predictions_binary = predictions > ANXIETY_THRESHOLD
    cm = confusion_matrix(y_test_binary, predictions_binary)
    test_accuracy = accuracy_score(y_test_binary, predictions_binary)

    # Train Classification Metrics
    train_predictions = model.predict(X_train_scaled)
    y_train_binary = y_train > ANXIETY_THRESHOLD
    train_predictions_binary = train_predictions > ANXIETY_THRESHOLD
    train_accuracy = accuracy_score(y_train_binary, train_predictions_binary)

    print("\n" + "="*40)
    print(" MODEL EVALUATION (Test Set)")
    print("="*40)
    print(f"Mean Absolute Error (MAE): {mae:.2f} (Scale: 0-100)")
    print(f"R-squared (R2) Score:      {r2:.4f}")
    
    print("\n" + "-"*40)
    print(" BINARY ACCURACY (Anxiety > 65)")
    print("-"*40)
    print(f"Train Accuracy:  {train_accuracy*100:.2f}%")
    print(f"Test Accuracy:   {test_accuracy*100:.2f}%")

    print("\n" + "-"*40)
    print(" BINARY CONFUSION MATRIX (Test Set)")
    print("-"*40)
    print(f"True Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")
    
    # Feature Importances
    print("\n" + "="*40)
    print(" FEATURE IMPORTANCES (Learned Weights)")
    print("="*40)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for idx in indices:
        print(f"{feature_cols[idx]:<20} : {importances[idx]:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Burnout Risk Notification Simulation
    # ---------------------------------------------------------
    # Simulate Burnout Notification (7 consecutive days > 65)
    # ---------------------------------------------------------
    # Since we did a random split, chronological testing doesn't work directly on y_test.
    # We will simulate the burnout logic on the ENTIRE dataset predictions to see who gets flagged.
    
    print("\n" + "="*40)
    print(" BURNOUT RISK NOTIFICATIONS (Simulated)")
    print("="*40)
    print("Goal: Notify user if predicted anxiety > 65 for 7 consecutive days.")
    
    df_sorted = full_dataset.sort_values(by=['user_id', 'date']).copy()
    X_all_scaled = scaler.transform(df_sorted[feature_cols])
    df_sorted['predicted_anxiety'] = model.predict(X_all_scaled)
    
    users_notified = set()
    for user_id, group in df_sorted.groupby('user_id'):
        consecutive_days = 0
        
        for _, row in group.iterrows():
            if row['predicted_anxiety'] > ANXIETY_THRESHOLD:
                consecutive_days += 1
                if consecutive_days == 7:
                    print(f" [ALERT] USER {user_id} flagged for burnout risk on: {row['date'].date()}")
                    users_notified.add(user_id)
                    break # Stop notifying same user over and over
            else:
                consecutive_days = 0
            
    print(f"\nTotal users notified: {len(users_notified)}")
    print("Done.")

if __name__ == "__main__":
    main()
