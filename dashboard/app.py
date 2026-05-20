import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import joblib
import json

# Project  : Procurement Fraud Detection System with Explainable AI
# Author   : Josephine Sherly P

DB_PATH      = "data/procurement.db"
MODEL_DIR    = "models"

st.set_page_config(
    page_title="Procurement Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f5f6fa; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e8e8e8; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMetric"] { background: #ffffff; border-radius: 10px; padding: 1rem 1.2rem; border: 1px solid #eaeaea; }
    .stButton > button { background-color: #1a1a2e; color: white; border-radius: 8px; border: none; padding: 0.5rem 1.5rem; font-weight: 500; width: 100%; }
    .stButton > button:hover { background-color: #4361ee; color: white; }
    hr { border-color: #eaeaea; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .risk-critical { background: #fff0f0; border-left: 5px solid #d62828; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    .risk-high { background: #fff7f0; border-left: 5px solid #f77f00; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    .risk-medium { background: #fffbf0; border-left: 5px solid #f4c430; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    .risk-low { background: #f0fff4; border-left: 5px solid #2dc653; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    .risk-title { font-size: 18px; font-weight: 600; margin: 0 0 4px; color: #1a1a2e; }
    .risk-desc { font-size: 13px; color: #555; margin: 0; }
    .form-card { background: #ffffff; border-radius: 12px; padding: 1.5rem; border: 1px solid #eaeaea; margin-bottom: 1rem; }
    .footer { text-align: center; padding: 2rem 0 1rem; font-size: 12px; color: #aaa; border-top: 1px solid #eaeaea; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Load model directly ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model        = joblib.load(f"{MODEL_DIR}/isolation_forest.joblib")
    scaler       = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    explainer    = joblib.load(f"{MODEL_DIR}/shap_explainer.joblib")
    le_dept      = joblib.load(f"{MODEL_DIR}/le_dept.joblib")
    le_cat       = joblib.load(f"{MODEL_DIR}/le_cat.joblib")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.joblib")
    with open(f"{MODEL_DIR}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{MODEL_DIR}/shap_importance.json") as f:
        shap_imp = json.load(f)
    return model, scaler, explainer, le_dept, le_cat, feature_cols, metrics, shap_imp

model, scaler, explainer, le_dept, le_cat, feature_cols, saved_metrics, shap_importance = load_model()

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
RISK_COLORS = {
    'Critical': '#d62828', 'High': '#f77f00',
    'Medium': '#f4c430', 'Low': '#2dc653'
}

def assign_risk_level(score):
    if score < -0.15:   return 'Critical'
    elif score < -0.10: return 'High'
    elif score < -0.05: return 'Medium'
    else:               return 'Low'

def encode_safe(encoder, value, fallback=0):
    try:    return int(encoder.transform([value])[0])
    except: return fallback

def get_shap_reasons(shap_vals, feature_names, top_n=3):
    indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    return [{"feature": feature_names[i], "contribution": round(float(shap_vals[i]), 4)} for i in indices]

def predict_order(order_id, vendor, department, category, amount, lag, weekend, vendor_freq):
    cat_avg      = CATEGORY_STATS.get(category, 10000)
    amount_ratio = amount / cat_avg
    vendor_rare  = 1 if vendor_freq < 5 else 0
    dept_enc     = encode_safe(le_dept, department)
    cat_enc      = encode_safe(le_cat, category)
    dept_risk    = DEPT_RISK.get(department, 0.3)
    fv = np.array([[amount, amount_ratio, vendor_freq, vendor_rare, lag, weekend, dept_risk, dept_enc, cat_enc]])
    scaled        = scaler.transform(fv)
    prediction    = model.predict(scaled)[0]
    score         = float(model.decision_function(scaled)[0])
    is_anomaly    = int(prediction == -1)
    risk          = assign_risk_level(score)
    shap_vals     = explainer.shap_values(scaled)[0]
    shap_reasons  = get_shap_reasons(shap_vals, feature_cols)
    return is_anomaly, round(score, 4), risk, shap_reasons

def load_orders():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("SELECT * FROM orders", conn)
    conn.close()
    return df

def page_header(title, subtitle):
    st.markdown(f"""
    <div style='margin-bottom:1.5rem;'>
        <h1 style='font-size:24px; font-weight:700; color:#1a1a2e; margin:0;'>{title}</h1>
        <p style='font-size:14px; color:#888; margin:4px 0 0;'>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def risk_alert(risk, order_id, score):
    cls  = f"risk-{risk.lower()}"
    icons = {'Critical':'🔴','High':'🟠','Medium':'🟡','Low':'🟢'}
    desc  = {
        'Critical': 'Immediate review required. High probability of fraudulent activity.',
        'High':     'Elevated risk detected. Flag for senior auditor review.',
        'Medium':   'Moderate anomaly signals. Routine audit recommended.',
        'Low':      'Minor anomaly. Likely a false positive. Log and monitor.'
    }
    st.markdown(f"""
    <div class="{cls}">
        <div class="risk-title">{icons.get(risk,'')} {risk} Risk — {order_id}</div>
        <div class="risk-desc">{desc.get(risk,'')} &nbsp;|&nbsp; Anomaly Score: {score}</div>
    </div>""", unsafe_allow_html=True)

def footer():
    st.markdown("""
    <div class="footer">
        Procurement Fraud Detection System &nbsp;|&nbsp;
        Built by <strong>Josephine Sherly P</strong> &nbsp;|&nbsp;
        CSE (AIML) Final Year &nbsp;|&nbsp;
        <a href='https://github.com/sherlyisrael13' style='color:#4361ee;'>github.com/sherlyisrael13</a>
    </div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.5rem;'>
        <div style='font-size:20px; font-weight:700; color:#1a1a2e;'>🔍 FraudDetect</div>
        <div style='font-size:12px; color:#888; margin-top:2px;'>Enterprise Anomaly Investigation</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px; color:#2dc653; margin-bottom:1rem;'>● Model Loaded</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px; color:#aaa; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", [
        "📊 Dashboard Overview",
        "🔎 Predict New Order",
        "📈 Model Insights"
    ], label_visibility="collapsed")
    st.markdown("""
    <div style='margin-top:2rem; padding-top:1rem; border-top:1px solid #eee;'>
        <div style='font-size:12px; color:#aaa;'>Built by</div>
        <div style='font-size:13px; font-weight:600; color:#1a1a2e;'>Josephine Sherly P</div>
        <div style='font-size:12px; color:#aaa;'>CSE (AIML) — Final Year</div>
        <div style='font-size:12px; color:#4361ee; margin-top:4px;'>github: sherlyisrael13</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":
    page_header("Dashboard Overview", "Real-time procurement anomaly monitoring across all departments")
    df         = load_orders()
    total      = len(df)
    total_anom = int(df['is_anomaly'].sum())
    anom_pct   = round(100 * total_anom / total, 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orders",    f"{total:,}")
    col2.metric("Anomalies Found", f"{total_anom:,}", delta=f"{anom_pct}% rate", delta_color="inverse")
    col3.metric("Anomaly Rate",    f"{anom_pct}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Anomaly Distribution by Type")
        anom_df = df[df['is_anomaly']==1]['anomaly_reason'].value_counts().reset_index()
        anom_df.columns = ['Type','Count']
        fig = px.bar(anom_df, x='Type', y='Count', color='Count',
                     color_continuous_scale=[[0,'#fde8e8'],[1,'#d62828']], text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
                          height=320, margin=dict(t=20,b=40,l=20,r=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Orders by Department")
        dept_df = df.groupby('department').agg(Total=('order_id','count'), Anomalies=('is_anomaly','sum')).reset_index()
        fig2 = px.bar(dept_df, x='department', y=['Total','Anomalies'], barmode='group',
                      color_discrete_map={'Total':'#4361ee','Anomalies':'#d62828'})
        fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=320,
                           margin=dict(t=20,b=40,l=20,r=20),
                           legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Transaction Amount — Normal vs Anomalous")
    fig3 = px.histogram(df, x='amount', color=df['is_anomaly'].map({0:'Normal',1:'Anomaly'}),
                        nbins=60, barmode='overlay', opacity=0.75,
                        color_discrete_map={'Normal':'#4361ee','Anomaly':'#d62828'})
    fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=280,
                       margin=dict(t=10,b=40,l=20,r=20), legend=dict(title=''))
    st.plotly_chart(fig3, use_container_width=True)
    footer()

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Predict New Order":
    page_header("Predict New Order", "Submit a purchase order for real-time fraud analysis")

    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        order_id   = st.text_input("Order ID",    value="PO-NEW-001")
        vendor     = st.text_input("Vendor Name", value="Acme Consulting Ltd")
        department = st.selectbox("Department",   ['IT','Finance','HR','Operations','Marketing','Legal','Procurement'])
        category   = st.selectbox("Category",     ['Software','Hardware','Consulting','Office Supplies','Travel','Maintenance','Training'])
    with col2:
        amount      = st.number_input("Amount (₹)",                    min_value=0.0,  value=5000.0,  step=500.0)
        lag         = st.number_input("Approval Lag (hours)",           min_value=0,    value=24)
        weekend     = st.selectbox("Approved on Weekend?",              [0,1], format_func=lambda x: "Yes — Weekend" if x==1 else "No — Weekday")
        vendor_freq = st.number_input("Vendor Frequency (past orders)", min_value=1,    value=10)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Analyse Order"):
        with st.spinner("Running anomaly analysis..."):
            is_anomaly, score, risk, shap_reasons = predict_order(
                order_id, vendor, department, category,
                amount, int(lag), int(weekend), int(vendor_freq)
            )
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Anomaly Detected", "YES ⚠️" if is_anomaly else "NO ✅")
            r2.metric("Risk Level",        risk)
            r3.metric("Anomaly Score",     score)

            risk_alert(risk, order_id, score)

            st.markdown("#### SHAP Explanation — Feature Contributions to Anomaly Score")
            st.caption("Negative contribution = feature pushed the score toward anomalous. Larger absolute value = stronger influence.")

            shap_df = pd.DataFrame(shap_reasons)
            fig = go.Figure(go.Bar(
                x=shap_df['contribution'], y=shap_df['feature'], orientation='h',
                marker_color=['#d62828' if v < 0 else '#4361ee' for v in shap_df['contribution']],
                text=[f"{v:.4f}" for v in shap_df['contribution']], textposition='outside'
            ))
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                              height=260, margin=dict(t=10,b=20,l=20,r=60))
            st.plotly_chart(fig, use_container_width=True)
    footer()

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    page_header("Model Insights", "Isolation Forest performance metrics and global SHAP feature importance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision",    saved_metrics['precision'])
    m2.metric("Recall",       saved_metrics['recall'])
    m3.metric("F1 Score",     saved_metrics['f1'])
    m4.metric("Total Orders", f"{saved_metrics['total_orders']:,}")
    st.caption("Metrics computed on injected ground-truth anomaly labels. Unsupervised anomaly detection — no fraud labels used during training.")

    st.divider()
    st.markdown("#### Global SHAP Feature Importance")
    st.caption("Average absolute SHAP contribution per feature. Higher value = stronger influence on the anomaly score.")

    shap_df = pd.DataFrame(list(shap_importance.items()), columns=['Feature','Importance']).sort_values('Importance', ascending=True)
    fig = go.Figure(go.Bar(
        x=shap_df['Importance'], y=shap_df['Feature'], orientation='h',
        marker=dict(color=shap_df['Importance'], colorscale=[[0,'#e8eeff'],[1,'#4361ee']]),
        text=[f"{v:.4f}" for v in shap_df['Importance']], textposition='outside'
    ))
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                      height=380, margin=dict(t=10,b=20,l=20,r=80))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Feature Engineering Reference")
    st.dataframe(pd.DataFrame({
        'Feature':     ['amount','amount_to_category_ratio','vendor_frequency','vendor_is_rare',
                        'approval_lag_hours','approved_on_weekend','department_risk_score','dept_encoded','cat_encoded'],
        'Description': [
            'Raw transaction amount in ₹',
            'Amount ÷ category average — detects spending spikes',
            'How many times this vendor appears in the dataset',
            '1 if vendor appears fewer than 5 times (new/rare vendor)',
            'Hours between order creation and approval',
            '1 if approved on Saturday or Sunday',
            'Historical anomaly risk score of the department (0–1)',
            'Label-encoded department identifier',
            'Label-encoded category identifier'
        ]
    }), use_container_width=True, hide_index=True)
    footer()
