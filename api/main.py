from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import json
import sqlite3
from datetime import datetime

app = FastAPI(
    title="Procurement Fraud Detection API",
    description="Detects anomalous purchase orders using Isolation Forest with SHAP explainability",
    version="1.0.0"
)

MODEL_DIR = "models"
model        = joblib.load(f"{MODEL_DIR}/isolation_forest.joblib")
scaler       = joblib.load(f"{MODEL_DIR}/scaler.joblib")
explainer    = joblib.load(f"{MODEL_DIR}/shap_explainer.joblib")
le_dept      = joblib.load(f"{MODEL_DIR}/le_dept.joblib")
le_cat       = joblib.load(f"{MODEL_DIR}/le_cat.joblib")
feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.joblib")

with open(f"{MODEL_DIR}/metrics.json") as f:
    saved_metrics = json.load(f)
with open(f"{MODEL_DIR}/shap_importance.json") as f:
    shap_importance = json.load(f)

DB_PATH = "data/procurement.db"

DEPT_RISK = {
    'IT': 0.3, 'Finance': 0.5, 'HR': 0.1,
    'Operations': 0.2, 'Marketing': 0.3,
    'Legal': 0.4, 'Procurement': 0.6
}
CATEGORY_STATS = {
    'Software': 7750, 'Hardware': 25500, 'Consulting': 41000,
    'Office Supplies': 1025, 'Travel': 4100,
    'Maintenance': 5150, 'Training': 2750,
}

def assign_risk_level(score: float) -> str:
    if score < -0.15:   return 'Critical'
    elif score < -0.10: return 'High'
    elif score < -0.05: return 'Medium'
    else:               return 'Low'

def encode_safe(encoder, value: str, fallback: int = 0) -> int:
    try:    return int(encoder.transform([value])[0])
    except: return fallback

def get_shap_reasons(shap_vals, feature_names, top_n=3):
    indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    return [{
        "feature":      feature_names[i],
        "contribution": round(float(shap_vals[i]), 4),
        "direction":    "increases anomaly score" if shap_vals[i] > 0 else "decreases anomaly score"
    } for i in indices]

class OrderRequest(BaseModel):
    order_id:            str
    vendor:              str
    department:          str
    category:            str
    amount:              float
    approval_lag_hours:  int
    approved_on_weekend: int
    vendor_frequency:    Optional[int] = 10

class FeedbackRequest(BaseModel):
    order_id:      str
    label:         str
    reviewer_note: Optional[str] = ""

@app.get("/")
def root():
    return {
        "project": "Procurement Fraud Detection System",
        "author":  "Josephine Sherly P",
        "version": "1.0.0",
        "endpoints": ["/predict", "/anomalies", "/feedback", "/retrain", "/metrics", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "IsolationForest", "features": len(feature_cols)}

@app.get("/metrics")
def get_metrics():
    return {"model_metrics": saved_metrics, "shap_importance": shap_importance}

@app.post("/predict")
def predict(order: OrderRequest):
    try:
        category_avg   = CATEGORY_STATS.get(order.category, 10000)
        amount_ratio   = order.amount / category_avg
        vendor_rare    = 1 if order.vendor_frequency < 5 else 0
        dept_enc       = encode_safe(le_dept, order.department)
        cat_enc        = encode_safe(le_cat, order.category)
        dept_risk      = DEPT_RISK.get(order.department, 0.3)

        feature_vector = np.array([[
            order.amount, amount_ratio, order.vendor_frequency,
            vendor_rare, order.approval_lag_hours, order.approved_on_weekend,
            dept_risk, dept_enc, cat_enc
        ]])

        scaled        = scaler.transform(feature_vector)
        prediction    = model.predict(scaled)[0]
        anomaly_score = float(model.decision_function(scaled)[0])
        is_anomaly    = int(prediction == -1)
        risk_level    = assign_risk_level(anomaly_score)
        shap_vals     = explainer.shap_values(scaled)[0]
        shap_reasons  = get_shap_reasons(shap_vals, feature_cols)

        result = {
            "order_id":      order.order_id,
            "is_anomaly":    is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "risk_level":    risk_level,
            "shap_reasons":  shap_reasons,
            "timestamp":     datetime.now().isoformat()
        }

        conn = sqlite3.connect(DB_PATH)
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            order_id TEXT, is_anomaly INTEGER, anomaly_score REAL,
            risk_level TEXT, timestamp TEXT)""")
        conn.execute("INSERT INTO predictions VALUES (?,?,?,?,?)",
            (order.order_id, is_anomaly, anomaly_score, risk_level, result["timestamp"]))
        conn.commit()
        conn.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/anomalies")
def get_anomalies(limit: int = 50, risk_level: Optional[str] = None):
    try:
        conn  = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM predictions WHERE is_anomaly = 1"
        if risk_level:
            query += f" AND risk_level = '{risk_level}'"
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return {"total": len(df), "anomalies": df.to_dict(orient="records")}
    except Exception:
        return {"total": 0, "anomalies": [], "note": "No predictions yet"}

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    if feedback.label not in ["true_fraud", "false_positive"]:
        raise HTTPException(status_code=400, detail="label must be 'true_fraud' or 'false_positive'")
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
            order_id TEXT, label TEXT, reviewer_note TEXT, timestamp TEXT)""")
        conn.execute("INSERT INTO feedback VALUES (?,?,?,?)",
            (feedback.order_id, feedback.label,
             feedback.reviewer_note, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return {
            "status": "recorded", "order_id": feedback.order_id,
            "label": feedback.label,
            "message": "Feedback saved. Will be used in next retraining cycle."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
def retrain():
    try:
        conn        = sqlite3.connect(DB_PATH)
        feedback_df = pd.read_sql_query("SELECT * FROM feedback", conn)
        conn.close()
        count = len(feedback_df)
        return {
            "status": "retraining scheduled", "feedback_count": count,
            "message": f"{count} feedback records queued for next training cycle."
        }
    except Exception:
        return {"status": "no feedback yet", "message": "Submit feedback via /feedback first."}
