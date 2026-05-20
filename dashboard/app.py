import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json

# Project  : Procurement Fraud Detection System with Explainable AI
# Author   : Josephine Sherly P

API_URL = "http://127.0.0.1:8000"
DB_PATH = "data/procurement.db"

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
    .risk-critical { background: #fff0f0; border-left: 5px solid #d62828; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    .risk-high { background: #fff7f0; border-left: 5px solid #f77f00; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    .risk-medium { background: #fffbf0; border-left: 5px solid #f4c430; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    .risk-low { background: #f0fff4; border-left: 5px solid #2dc653; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    .risk-title { font-size: 18px; font-weight: 600; margin: 0 0 4px; color: #1a1a2e; }
    .risk-desc { font-size: 13px; color: #555; margin: 0; }
    .footer { text-align: center; padding: 2rem 0 1rem; font-size: 12px; color: #aaa; border-top: 1px solid #eaeaea; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.5rem;'>
        <div style='font-size:20px; font-weight:700; color:#1a1a2e;'>🔍 FraudDetect</div>
        <div style='font-size:12px; color:#888; margin-top:2px;'>Enterprise Anomaly Investigation</div>
    </div>""", unsafe_allow_html=True)

    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.markdown("<div style='font-size:12px; color:#2dc653; margin-bottom:1rem;'>● API Online</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:12px; color:#d62828; margin-bottom:1rem;'>● API Offline</div>", unsafe_allow_html=True)
    except Exception:
        st.markdown("<div style='font-size:12px; color:#d62828; margin-bottom:1rem;'>● API Offline</div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:11px; color:#aaa; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📊 Dashboard Overview",
        "🔎 Predict New Order",
        "📋 Flagged Orders",
        "📈 Model Insights"
    ], label_visibility="collapsed")

    st.markdown("""
    <div style='margin-top:2rem; padding-top:1rem; border-top:1px solid #eee;'>
        <div style='font-size:12px; color:#aaa;'>Built by</div>
        <div style='font-size:13px; font-weight:600; color:#1a1a2e;'>Josephine Sherly P</div>
        <div style='font-size:12px; color:#aaa;'>CSE (AIML) — Final Year</div>
        <div style='font-size:12px; color:#4361ee; margin-top:4px;'>github: sherlyisrael13</div>
    </div>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_all_orders():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("SELECT * FROM orders", conn)
    conn.close()
    return df

def load_predictions():
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

RISK_COLORS = {
    'Critical': '#d62828', 'High': '#f77f00',
    'Medium': '#f4c430',   'Low':  '#2dc653'
}

def page_header(title, subtitle):
    st.markdown(f"""
    <div style='margin-bottom:1.5rem;'>
        <h1 style='font-size:24px; font-weight:700; color:#1a1a2e; margin:0;'>{title}</h1>
        <p style='font-size:14px; color:#888; margin:4px 0 0;'>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def risk_alert(risk, order_id, score):
    cls   = f"risk-{risk.lower()}"
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

# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":
    page_header("Dashboard Overview", "Real-time procurement anomaly monitoring across all departments")

    df         = load_all_orders()
    pred_df    = load_predictions()
    total      = len(df)
    total_anom = int(df['is_anomaly'].sum())
    anom_pct   = round(100 * total_anom / total, 1)
    api_preds  = len(pred_df) if not pred_df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders",    f"{total:,}")
    col2.metric("Anomalies Found", f"{total_anom:,}", delta=f"{anom_pct}% rate", delta_color="inverse")
    col3.metric("Anomaly Rate",    f"{anom_pct}%")
    col4.metric("API Predictions", f"{api_preds:,}")

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
        dept_df = df.groupby('department').agg(
            Total=('order_id','count'), Anomalies=('is_anomaly','sum')).reset_index()
        fig2 = px.bar(dept_df, x='department', y=['Total','Anomalies'], barmode='group',
                      color_discrete_map={'Total':'#4361ee','Anomalies':'#d62828'})
        fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=320,
                           margin=dict(t=20,b=40,l=20,r=20),
                           legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Transaction Amount — Normal vs Anomalous")
    fig3 = px.histogram(df, x='amount',
                        color=df['is_anomaly'].map({0:'Normal',1:'Anomaly'}),
                        nbins=60, barmode='overlay', opacity=0.75,
                        color_discrete_map={'Normal':'#4361ee','Anomaly':'#d62828'})
    fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=280,
                       margin=dict(t=10,b=40,l=20,r=20), legend=dict(title=''))
    st.plotly_chart(fig3, use_container_width=True)
    footer()

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Predict New Order":
    page_header("Predict New Order", "Submit a purchase order for real-time fraud analysis")

    st.markdown('<div style="background:#ffffff; border-radius:12px; padding:1.5rem; border:1px solid #eaeaea; margin-bottom:1rem;">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        order_id   = st.text_input("Order ID",    value="PO-NEW-001")
        vendor     = st.text_input("Vendor Name", value="Acme Consulting Ltd")
        department = st.selectbox("Department",   ['IT','Finance','HR','Operations','Marketing','Legal','Procurement'])
        category   = st.selectbox("Category",     ['Software','Hardware','Consulting','Office Supplies','Travel','Maintenance','Training'])
    with col2:
        amount              = st.number_input("Amount (₹)",                    min_value=0.0,  value=5000.0,  step=500.0)
        approval_lag_hours  = st.number_input("Approval Lag (hours)",           min_value=0,    value=24)
        approved_on_weekend = st.selectbox("Approved on Weekend?",              [0,1], format_func=lambda x: "Yes — Weekend" if x==1 else "No — Weekday")
        vendor_frequency    = st.number_input("Vendor Frequency (past orders)", min_value=1,    value=10)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Analyse Order"):
        payload = {
            "order_id":            order_id,
            "vendor":              vendor,
            "department":          department,
            "category":            category,
            "amount":              amount,
            "approval_lag_hours":  int(approval_lag_hours),
            "approved_on_weekend": int(approved_on_weekend),
            "vendor_frequency":    int(vendor_frequency)
        }
        with st.spinner("Running anomaly analysis..."):
            try:
                res  = requests.post(f"{API_URL}/predict", json=payload)
                data = res.json()

                st.divider()
                r1, r2, r3 = st.columns(3)
                r1.metric("Anomaly Detected", "YES ⚠️" if data['is_anomaly'] else "NO ✅")
                r2.metric("Risk Level",        data['risk_level'])
                r3.metric("Anomaly Score",     data['anomaly_score'])

                risk_alert(data['risk_level'], data['order_id'], data['anomaly_score'])

                st.markdown("#### SHAP Explanation — Feature Contributions to Anomaly Score")
                st.caption("Negative contribution = feature pushed the score toward anomalous. Larger absolute value = stronger influence.")

                shap_df = pd.DataFrame(data['shap_reasons'])
                fig = go.Figure(go.Bar(
                    x=shap_df['contribution'], y=shap_df['feature'], orientation='h',
                    marker_color=['#d62828' if v < 0 else '#4361ee' for v in shap_df['contribution']],
                    text=[f"{v:.4f}" for v in shap_df['contribution']], textposition='outside'
                ))
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                  height=260, margin=dict(t=10,b=20,l=20,r=60))
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.markdown("#### Auditor Feedback")
                st.caption("Your assessment is stored and used to improve the model in future retraining cycles.")
                fb1, fb2 = st.columns(2)
                with fb1:
                    label = st.selectbox("Assessment", ["true_fraud","false_positive"],
                                         format_func=lambda x: "✅ True Fraud" if x=="true_fraud" else "❌ False Positive")
                with fb2:
                    note = st.text_input("Reviewer Note (optional)", placeholder="e.g. Known vendor, approved manually")

                if st.button("Submit Feedback"):
                    fb_res = requests.post(f"{API_URL}/feedback", json={
                        "order_id": order_id, "label": label, "reviewer_note": note
                    })
                    if fb_res.status_code == 200:
                        st.success("Feedback recorded successfully.")
                    else:
                        st.error("Feedback submission failed.")

            except Exception as e:
                st.error(f"API connection error: {e}. Make sure uvicorn is running.")
    footer()

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Flagged Orders":
    page_header("Flagged Orders", "All purchase orders flagged as anomalous by the detection model")

    risk_filter = st.selectbox("Filter by Risk Level", ["All","Critical","High","Medium","Low"])

    try:
        url = f"{API_URL}/anomalies?limit=200"
        if risk_filter != "All":
            url += f"&risk_level={risk_filter}"
        res  = requests.get(url)
        data = res.json()

        if data['total'] == 0:
            st.info("No predictions yet. Go to 'Predict New Order' to analyse orders first.")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Total Flagged", data['total'])
            m2.metric("Filter Active", risk_filter)

            pred_df = pd.DataFrame(data['anomalies'])
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(
                pred_df.style.map(
                    lambda v: f"color: {RISK_COLORS.get(v,'#333')}; font-weight:600"
                    if v in RISK_COLORS else '',
                    subset=['risk_level'] if 'risk_level' in pred_df.columns else []
                ),
                use_container_width=True, height=380
            )

            st.markdown("#### Risk Level Distribution")
            fig = px.pie(pred_df, names='risk_level', color='risk_level',
                         color_discrete_map=RISK_COLORS, hole=0.45)
            fig.update_layout(paper_bgcolor='white', height=320,
                              margin=dict(t=20,b=20,l=20,r=20),
                              legend=dict(orientation='h', yanchor='bottom', y=-0.2))
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load flagged orders: {e}")
    footer()

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    page_header("Model Insights", "Isolation Forest performance metrics and global SHAP feature importance")

    try:
        res      = requests.get(f"{API_URL}/metrics")
        data     = res.json()
        metrics  = data['model_metrics']
        shap_imp = data['shap_importance']

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision",    metrics['precision'])
        m2.metric("Recall",       metrics['recall'])
        m3.metric("F1 Score",     metrics['f1'])
        m4.metric("Total Orders", f"{metrics['total_orders']:,}")
        st.caption("Metrics computed on injected ground-truth anomaly labels. Unsupervised anomaly detection — no fraud labels used during training.")

        st.divider()
        st.markdown("#### Global SHAP Feature Importance")
        st.caption("Average absolute SHAP contribution per feature. Higher value = stronger influence on the anomaly score.")

        shap_df = pd.DataFrame(
            list(shap_imp.items()), columns=['Feature','Importance']
        ).sort_values('Importance', ascending=True)

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
                            'approval_lag_hours','approved_on_weekend','department_risk_score',
                            'dept_encoded','cat_encoded'],
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

    except Exception as e:
        st.error(f"Could not load metrics: {e}")
    footer()
