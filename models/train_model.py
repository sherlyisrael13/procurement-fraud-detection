import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import shap
import joblib
import json
import os

# Project  : Procurement Fraud Detection System with Explainable AI
# Author   : Josephine Sherly P

def engineer_features(df):
    df = df.copy()

    # Feature 1: amount vs category average (spending spike detection)
    category_avg = df.groupby('category')['amount'].transform('mean')
    df['amount_to_category_ratio'] = df['amount'] / category_avg

    # Feature 2: vendor frequency (rare vendors are suspicious)
    vendor_freq = df['vendor'].value_counts()
    df['vendor_frequency'] = df['vendor'].map(vendor_freq)
    df['vendor_is_rare'] = (df['vendor_frequency'] < 5).astype(int)

    # Feature 3: weekend approval flag
    df['approved_at'] = pd.to_datetime(df['approved_at'])
    df['approval_dow'] = df['approved_at'].dt.dayofweek
    df['approved_on_weekend'] = (df['approval_dow'] >= 5).astype(int)

    # Feature 4: encode categorical columns
    le_dept = LabelEncoder()
    le_cat  = LabelEncoder()
    df['dept_encoded'] = le_dept.fit_transform(df['department'])
    df['cat_encoded']  = le_cat.fit_transform(df['category'])

    # Save encoders for use in API later
    os.makedirs('models', exist_ok=True)
    joblib.dump(le_dept, 'models/le_dept.joblib')
    joblib.dump(le_cat,  'models/le_cat.joblib')

    feature_cols = [
        'amount',
        'amount_to_category_ratio',
        'vendor_frequency',
        'vendor_is_rare',
        'approval_lag_hours',
        'approved_on_weekend',
        'department_risk_score',
        'dept_encoded',
        'cat_encoded'
    ]

    return df, feature_cols

def assign_risk_level(score):
    # score is the raw anomaly score from Isolation Forest
    # more negative = more anomalous
    if score < -0.15:
        return 'Critical'
    elif score < -0.10:
        return 'High'
    elif score < -0.05:
        return 'Medium'
    else:
        return 'Low'

def train():
    print("Loading data...")
    df = pd.read_csv('data/procurement_data.csv')
    print(f"  Loaded {len(df)} orders")

    print("\nEngineering features...")
    df, feature_cols = engineer_features(df)
    print(f"  Features: {feature_cols}")

    X      = df[feature_cols].values
    y_true = df['is_anomaly'].values

    print("\nScaling features...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nTraining Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    print("\nEvaluating model...")
    raw_scores    = model.decision_function(X_scaled)
    preds         = model.predict(X_scaled)
    preds_binary  = (preds == -1).astype(int)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, preds_binary, target_names=['Normal', 'Anomaly']))

    precision = precision_score(y_true, preds_binary)
    recall    = recall_score(y_true, preds_binary)
    f1        = f1_score(y_true, preds_binary)

    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")

    print("\nAssigning risk levels...")
    df['anomaly_score'] = raw_scores
    df['risk_level']    = df['anomaly_score'].apply(assign_risk_level)
    df['predicted_anomaly'] = preds_binary

    risk_summary = df[df['predicted_anomaly']==1]['risk_level'].value_counts()
    print(f"\nRisk level breakdown (flagged orders):")
    print(risk_summary.to_string())

    print("\nGenerating SHAP values (this takes ~30 seconds)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Save mean absolute SHAP values per feature for global importance
    shap_importance = dict(zip(
        feature_cols,
        np.abs(shap_values).mean(axis=0).tolist()
    ))
    print("\nSHAP Feature Importance (contribution to anomaly score):")
    for feat, val in sorted(shap_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat:<35} {val:.4f}")

    print("\nSaving model artifacts...")
    joblib.dump(model,        'models/isolation_forest.joblib')
    joblib.dump(scaler,       'models/scaler.joblib')
    joblib.dump(feature_cols, 'models/feature_cols.joblib')
    joblib.dump(explainer,    'models/shap_explainer.joblib')

    with open('models/feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)

    with open('models/shap_importance.json', 'w') as f:
        json.dump(shap_importance, f)

    metrics = {
        'precision': round(precision, 3),
        'recall':    round(recall, 3),
        'f1':        round(f1, 3),
        'total_orders':   len(df),
        'total_flagged':  int(preds_binary.sum()),
        'risk_breakdown': risk_summary.to_dict()
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ All model artifacts saved to models/")
    print(f"   isolation_forest.joblib")
    print(f"   scaler.joblib")
    print(f"   le_dept.joblib + le_cat.joblib")
    print(f"   shap_explainer.joblib")
    print(f"   feature_cols.json")
    print(f"   shap_importance.json")
    print(f"   metrics.json")

    return df, model, scaler, feature_cols

if __name__ == '__main__':
    df, model, scaler, feature_cols = train()