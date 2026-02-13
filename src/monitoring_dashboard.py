import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="ML Pipeline Monitoring", page_icon="📈", layout="wide")

st.title("🛡️ ML Pipeline Observability Dashboard")
st.markdown("Monitor data quality, pipeline health, and model performance in real-time.")

# --- SIDEBAR: LOG SELECTION ---
st.sidebar.header("Pipeline Controls")
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# --- HELPER: Load Logs ---
def load_latest_log(step_name):
    path = f"data/logs/{step_name}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            return data[-1] if isinstance(data, list) else data
    return None

# --- TOP ROW: KPI CARDS ---
col1, col2, col3, col4 = st.columns(4)

prep_log = load_latest_log("step1_data_prep")
feat_log = load_latest_log("step2_features")
reg_log = load_latest_log("step3_regression")
eval_log = load_latest_log("step4_evaluation")

with col1:
    if prep_log:
        st.metric("Data Processed", f"{prep_log.get('final_rows', 0):,}", f"{prep_log.get('matches_found', 0):,} matches")
    else:
        st.metric("Data Processed", "No Data")

with col2:
    if reg_log:
        st.metric("Model RMSE", f"€{reg_log.get('rmse', 0):.2f}", f"R²: {reg_log.get('r2', 0):.2f}")
    else:
        st.metric("Model RMSE", "No Data")

with col3:
    clf_log = load_latest_log("step3_classification")
    if clf_log:
        st.metric("Pricing Accuracy", f"{clf_log.get('accuracy', 0)*100:.1f}%")
    else:
        st.metric("Pricing Accuracy", "No Data")

with col4:
    valid_logs = glob.glob("data/logs/*_validation.json")
    failures = 0
    for v in valid_logs:
        with open(v, "r") as f:
            d = json.load(f)
            if d[-1].get("status") == "failure":
                 failures += 1
    st.metric("Validation Alerts", failures, delta_color="inverse" if failures > 0 else "normal")

# --- MAIN CONTENT: TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Pipeline Health", "🧪 Data Quality (Pandera)", "📉 Model Drift (Evidently)"])

with tab1:
    st.header("Step-wise Execution Logs")
    
    steps = ["step1_data_prep", "step2_features", "step3_regression", "step4_evaluation"]
    for step in steps:
        log = load_latest_log(step)
        with st.expander(f"📌 {step.replace('_', ' ').title()}", expanded=True):
            if log:
                c1, c2 = st.columns([1, 3])
                c1.info(f"**Timestamp**: {log.get('timestamp')[:19]}")
                c2.json(log)
            else:
                st.warning(f"No logs found for {step}")

with tab2:
    st.header("Pandera Validation History")
    
    valid_files = glob.glob("data/logs/*_validation.json")
    if not valid_files:
        st.info("No validation results yet. Run the pipeline!")
    else:
        for vf in valid_files:
            name = os.path.basename(vf).replace("_validation.json", "")
            with open(vf, "r") as f:
                history = json.load(f)
            
            df_v = pd.DataFrame(history)
            st.subheader(f"🔍 {name.replace('_', ' ').title()}")
            
            # Show success rate pie chart
            success_count = (df_v['status'] == 'success').sum()
            fail_count = (df_v['status'] == 'failure').sum()
            
            fig = px.pie(values=[success_count, fail_count], names=['Success', 'Failure'], 
                         color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4, height=300)
            st.plotly_chart(fig)
            
            if fail_count > 0:
                st.error(f"Alert: {fail_count} validation failures detected in {name}!")
                st.dataframe(df_v[df_v['status'] == 'failure'])

with tab3:
    st.header("Evidently AI Reports")
    
    reports = glob.glob("data/reports/*.html")
    if not reports:
        st.info("No reports generated yet. Reports are created during training and evaluation.")
    else:
        selected_report = st.selectbox("Select Report", [os.path.basename(r) for r in reports])
        report_path = os.path.join("data/reports", selected_report)
        
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=1000, scrolling=True)

st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
