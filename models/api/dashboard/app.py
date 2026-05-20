import streamlit as st
import requests
import pandas as pd
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
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #1a1a2e; margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem; color: #555; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; text-align: center;
        border-left: 4px solid #4361ee;
    }
    .risk-critical { color: #d62828; font-weight: 700; }
    .risk-high     { color: #f77f00; font-weight: 700; }
    .risk-medium   { color: #fcbf49; font-weight: 700; }
    .risk-low      { color: #2dc653; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔍 Procurement Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enterprise Anomaly Investigation Platform — Josephine Sherly P</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Dashboard Overview",
    "🔎 Predict New Order",
    "📋 Flagged Orders",
    "📈 Model Insights"
])

# ── Helper functions ──────────────────────────────────────────────────────────
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

def risk_color(risk):
    colors = {
        'Critical': '#d62828',
        'High':     '#f77f00',
        'Medium':   '#fcbf49',
        'Low':      '#2dc653'
    }
    return colors.get(risk, '#888')

# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":
    df       = load_all_orders()
    pred_df  = load_predictions()

    total        = len(df)
    total_anom   = int(df['is_anomaly'].sum())
    anom_pct     = round(100 * total_anom / total, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders",     f"{total:,}")
    col2.metric("Anomalies Found",  f"{total_anom:,}")
    col3.metric("Anomaly Rate",     f"{anom_pct}%")
    col4.metric("API Predictions",  f"{len(pred_df):,}" if not pred_df.empty else "0")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Anomalies by Type")
        anom_df = df[df['is_anomaly']==1]['anomaly_reason'].value_counts().reset_index()
        anom_df.columns = ['Anomaly Type', 'Count']
        fig = px.bar(
            anom_df, x='Anomaly Type', y='Count',
            color='Count', color_continuous_scale='Reds',
            title="Distribution of Injected Anomaly Types"
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Orders by Department")
        dept_df = df.groupby('department').agg(
            total=('order_id','count'),
            anomalies=('is_anomaly','sum')
        ).reset_index()
        fig2 = px.bar(
            dept_df, x='department', y=['total','anomalies'],
            barmode='group', title="Total vs Anomalous Orders per Department",
            color_discrete_map={'total':'#4361ee','anomalies':'#d62828'}
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Amount Distribution — Normal vs Anomalous")
    fig3 = px.histogram(
        df, x='amount', color=df['is_anomaly'].map({0:'Normal',1:'Anomaly'}),
        nbins=60, barmode='overlay', opacity=0.7,
        color_discrete_map={'Normal':'#4361ee','Anomaly':'#d62828'},
        title="Transaction Amount Distribution"
    )
    fig3.update_layout(height=320)
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Predict New Order":
    st.subheader("Submit a Purchase Order for Fraud Analysis")
    st.info("Fill in the order details below. The model will return an anomaly score, risk level, and SHAP explanation.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            order_id   = st.text_input("Order ID",   value="PO-NEW-001")
            vendor     = st.text_input("Vendor Name", value="Acme Corp")
            department = st.selectbox("Department", ['IT','Finance','HR','Operations','Marketing','Legal','Procurement'])
            category   = st.selectbox("Category",   ['Software','Hardware','Consulting','Office Supplies','Travel','Maintenance','Training'])
        with col2:
            amount              = st.number_input("Amount (₹)", min_value=0.0, value=5000.0, step=100.0)
            approval_lag_hours  = st.number_input("Approval Lag (hours)", min_value=0, value=24)
            approved_on_weekend = st.selectbox("Approved on Weekend?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            vendor_frequency    = st.number_input("Vendor Frequency (past orders)", min_value=1, value=10)

        submitted = st.form_submit_button("🔍 Analyse Order", use_container_width=True)

    if submitted:
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
        with st.spinner("Analysing order..."):
            try:
                res  = requests.post(f"{API_URL}/predict", json=payload)
                data = res.json()

                st.divider()
                r1, r2, r3 = st.columns(3)
                r1.metric("Anomaly Detected", "YES ⚠️" if data['is_anomaly'] else "NO ✅")
                r2.metric("Risk Level",        data['risk_level'])
                r3.metric("Anomaly Score",     data['anomaly_score'])

                risk  = data['risk_level']
                color = risk_color(risk)
                st.markdown(f"### Risk Level: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)

                st.subheader("SHAP Explanation — Why was this flagged?")
                st.caption("Each bar shows how much a feature contributed to the anomaly score.")

                shap_df = pd.DataFrame(data['shap_reasons'])
                fig = px.bar(
                    shap_df, x='contribution', y='feature',
                    orientation='h', color='contribution',
                    color_continuous_scale='RdBu',
                    title="Feature Contributions to Anomaly Score"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Submit Auditor Feedback")
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    label = st.selectbox("Your Assessment", ["true_fraud", "false_positive"])
                with fb_col2:
                    note  = st.text_input("Reviewer Note (optional)")

                if st.button("Submit Feedback"):
                    fb_res = requests.post(f"{API_URL}/feedback", json={
                        "order_id": order_id, "label": label, "reviewer_note": note
                    })
                    if fb_res.status_code == 200:
                        st.success("Feedback recorded. Thank you.")
                    else:
                        st.error("Feedback failed.")

            except Exception as e:
                st.error(f"API error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Flagged Orders":
    st.subheader("All Flagged Anomalous Orders")

    risk_filter = st.selectbox("Filter by Risk Level", ["All", "Critical", "High", "Medium", "Low"])

    try:
        url    = f"{API_URL}/anomalies?limit=200"
        if risk_filter != "All":
            url += f"&risk_level={risk_filter}"
        res    = requests.get(url)
        data   = res.json()
        if data['total'] == 0:
            st.info("No predictions yet. Go to 'Predict New Order' to analyse some orders first.")
        else:
            st.metric("Total Flagged", data['total'])
            pred_df = pd.DataFrame(data['anomalies'])
            st.dataframe(pred_df, use_container_width=True, height=400)

            fig = px.pie(
                pred_df, names='risk_level',
                color='risk_level',
                color_discrete_map={
                    'Critical':'#d62828','High':'#f77f00',
                    'Medium':'#fcbf49','Low':'#2dc653'
                },
                title="Flagged Orders by Risk Level"
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load flagged orders: {e}")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.subheader("Model Performance & SHAP Feature Importance")

    try:
        res     = requests.get(f"{API_URL}/metrics")
        data    = res.json()
        metrics = data['model_metrics']
        shap_imp = data['shap_importance']

        st.subheader("Model Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", metrics['precision'])
        m2.metric("Recall",    metrics['recall'])
        m3.metric("F1 Score",  metrics['f1'])

        st.caption("Metrics based on injected anomaly labels in synthetic dataset.")

        st.divider()
        st.subheader("Global SHAP Feature Importance")
        st.caption("Average absolute SHAP contribution per feature toward the anomaly score across all orders.")

        shap_df = pd.DataFrame(
            list(shap_imp.items()), columns=['Feature','Importance']
        ).sort_values('Importance', ascending=True)

        fig = px.bar(
            shap_df, x='Importance', y='Feature',
            orientation='h',
            color='Importance', color_continuous_scale='Blues',
            title="SHAP Feature Importance (Global)"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Feature Engineering Reference")
        st.dataframe(pd.DataFrame({
            'Feature':     ['amount','amount_to_category_ratio','vendor_frequency',
                           'vendor_is_rare','approval_lag_hours','approved_on_weekend',
                           'department_risk_score','dept_encoded','cat_encoded'],
            'Description': [
                'Raw transaction amount',
                'Amount divided by category average — detects spending spikes',
                'How many times this vendor appears in dataset',
                '1 if vendor appears fewer than 5 times',
                'Hours between order creation and approval',
                '1 if approved on Saturday or Sunday',
                'Historical risk score of the department',
                'Label-encoded department',
                'Label-encoded category'
            ]
        }), use_container_width=True)

    except Exception as e:
        st.error(f"Could not load metrics: {e}")