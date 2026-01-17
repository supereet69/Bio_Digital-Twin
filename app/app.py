import sys, os, time, random
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from graphviz import Digraph

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.data_loader import load_bio_leaching_data
from src.data_cleaning import preprocess_bio_leaching
from src.twin_model import fit_ph_model
from src.prediction_model import predict_future_ph_state
from src.validation import compute_validation_metrics
from src.recovery_model import predict_metal_recovery_dynamic
from src.optimization import optimize_operating_conditions

# ================= PAGE =================
st.set_page_config(layout="wide")

st.markdown("""
<style>
.panel {
    background: linear-gradient(180deg, #0b1220 0%, #020617 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: inset 0 6px 12px rgba(255,255,255,0.02),
                inset 0 -6px 12px rgba(0,0,0,0.4);
    border-left: 4px solid #ffffff;
}
.panel-header {
    border-bottom: 1px solid #334155;
    padding-bottom: 6px;
    margin-bottom: 10px;
    font-weight: 600;
}
.anomaly-panel { border-left: 4px solid #dc2626; }
.sys-panel { border-left: 4px solid #22c55e; }

/* ===== BUTTON STYLING ===== */
button[kind="primary"], button[kind="secondary"], div.stButton > button {
    background: transparent !important;
    border: 1.5px solid #64748b !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 0.6em 1em !important;
    transition: all 0.25s ease-in-out !important;
}

/* Hover Fill Effect */
div.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #22c55e) !important;
    border-color: transparent !important;
    box-shadow: 0 0 12px rgba(37,99,235,0.6);
    transform: translateY(-1px);
}

/* Click Effect */
div.stButton > button:active {
    background: linear-gradient(135deg, #1d4ed8, #16a34a) !important;
    box-shadow: 0 0 6px rgba(37,99,235,0.8);
    transform: translateY(0px);
}
</style>
""", unsafe_allow_html=True)


st.title("Bio-Leaching Digital Twin")
st.caption("Prediction • Fault Diagnosis • Optimization")

# ================= DATA =================
master_df, meta_df = load_bio_leaching_data()
clean_df = preprocess_bio_leaching(master_df)

# ================= FLOWSHEET =================
def render_bio_flowsheet(state="idle", fault=None):
    dot = Digraph(engine="dot")
    dot.attr(rankdir="LR", bgcolor="transparent")

    def color(unit):
        if state == "run":
            if fault == unit:
                return "#dc2626"
            return "#22c55e"
        return "#64748b"

    dot.attr("node", shape="box", style="rounded,filled", fontcolor="white",
             width="3.8", height="2.0")

    dot.node("A", "Feed", fillcolor=color("feed"))
    dot.node("B", "Bio-Reactor", fillcolor=color("reactor"))
    dot.node("C", "Sensors", fillcolor=color("sensor"))
    dot.node("D", "Digital Twin", fillcolor="#8b5cf6")

    dot.edge("A","B", color="white", penwidth="2")
    dot.edge("B","C", color="white", penwidth="2")
    dot.edge("C","D", color="white", penwidth="2")

    return dot

ERROR_LIBRARY = {
    "Biological Inhibition": ["🔴 Reduced microbial activity detected"],
    "Process Shock": ["🔴 Feed composition change detected"],
    "Sensor Fault": ["🔴 Calibration drift suspected"]
}

# =========================================================
# ================= LAYOUT ===============================
# =========================================================

row1_c1, row1_c2 = st.columns([1,2])

# ---------------- CONTROL PANEL ----------------
with row1_c1:
    st.markdown("<div class='panel sys-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header'>System Parameters</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        system_map = {"Bio-leaching":1,"Biological Control":2,"Abiotic Control":3}
        flask_id = system_map[st.selectbox("System", system_map.keys())]
        future_days = st.slider("Prediction Days",5,30,20)

    with c2:
        T = st.slider("Temperature (K)",290,330,303)
        PD = st.slider("Pulp Density",0.01,0.25,0.08)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel anomaly-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header'>Anomaly Settings</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        anomaly_type = st.selectbox("Anomaly",["None","Biological Inhibition","Process Shock","Sensor Fault"])
    with c4:
        anomaly_time = st.slider("Anomaly Time",0.0,float(future_days),future_days/2)

    anomaly_severity = st.slider("Severity",0.0,1.0,0.4)

    run_button = st.button("▶ Run Simulation", use_container_width=True)
    optimize_button = st.button("Optimize", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- GRAPHS ----------------
with row1_c2:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header'>Process Trends</div>", unsafe_allow_html=True)
    ph_plot = st.empty()
    rec_plot = st.empty()
    opt_plot = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ===== INITIAL PLACEHOLDER CHARTS =====

t_dummy = np.linspace(0, future_days, 30)

ph_dummy = pd.DataFrame({
    "Time": t_dummy,
    "pH": np.ones_like(t_dummy) * 7.5,
    "Type": ["Idle"] * len(t_dummy)
})

rec_dummy = pd.DataFrame({
    "Time": t_dummy,
    "Recovery": np.linspace(0, 5, len(t_dummy)),
    "Type": ["Idle"] * len(t_dummy)
})

ph_plot.altair_chart(
    alt.Chart(ph_dummy).mark_line(strokeDash=[6,4], opacity=0.4).encode(
        x=alt.X("Time:Q", title="Time (days)"),
        y=alt.Y("pH:Q", title="pH"),
        color=alt.value("#64748b")
    ).properties(height=260),
    use_container_width=True
)

rec_plot.altair_chart(
    (alt.Chart(rec_dummy).mark_line(strokeDash=[6,4], opacity=0.4).encode(
        x=alt.X("Time:Q", title="Time (days)"),
        y=alt.Y("Recovery:Q", title="Metal Recovery (%)"),
        color=alt.value("#64748b")
    ) + alt.Chart(rec_dummy).mark_circle(size=40, opacity=0.4).encode(
        x="Time:Q", y="Recovery:Q", color=alt.value("#64748b")
    )).properties(height=260),
    use_container_width=True
)

# ---------------- MIDDLE ----------------
row2_c1, row2_c2 = st.columns(2)

with row2_c1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header'>Plant Flowsheet</div>", unsafe_allow_html=True)
    flow_box = st.empty()
    flow_box.graphviz_chart(render_bio_flowsheet(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row2_c2:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header'>Live Event Log</div>", unsafe_allow_html=True)
    log_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- BOTTOM ----------------
st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.markdown("<div class='panel-header'>Results</div>", unsafe_allow_html=True)
k1,k2,k3,k4 = st.columns(4)
kpi1 = k1.empty(); kpi2 = k2.empty(); kpi3 = k3.empty(); kpi4 = k4.empty()
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# ================= RUN SIMULATION ========================
# =========================================================

if run_button:

    fault_map = {
        "Biological Inhibition":"reactor",
        "Process Shock":"feed",
        "Sensor Fault":"sensor"
    }

    flask_df = clean_df[clean_df["Flask_ID"]==flask_id]
    t_obs = flask_df["Time_days"].values
    pH_obs = flask_df["pH"].values
    OD = np.nanmean(flask_df["OD600"].values)

    params = fit_ph_model(t_obs, pH_obs)
    pH_fit = params[0] + (params[2]-params[0])*np.exp(-params[1]*t_obs)
    rmse, r2 = compute_validation_metrics(pH_obs, pH_fit)

    t_pred, pH_base = predict_future_ph_state(t_obs, pH_obs, params, OD, T, PD, future_days)
    pH_anom = pH_base.copy()

    idx = np.where(t_pred >= t_pred[0] + anomaly_time)[0]

    if anomaly_type!="None" and len(idx)>0:
        start = idx[0]
        if anomaly_type=="Biological Inhibition":
            pH_anom[start:] += anomaly_severity
        elif anomaly_type=="Process Shock":
            pH_anom[start:] += 1.2*anomaly_severity
        elif anomaly_type=="Sensor Fault":
            pH_anom += np.random.normal(0,0.15,len(pH_anom))

# ================= METAL RECOVERY BY SYSTEM =================

    if flask_id == 1:  # Bio-leaching (bacteria + metal)
        rec_base = predict_metal_recovery_dynamic(t_pred, pH_base, OD)
        rec_anom = predict_metal_recovery_dynamic(t_pred, pH_anom, OD)

    elif flask_id == 2:  # Biological control (no metal present)
        rec_base = np.zeros_like(t_pred)
        rec_anom = np.zeros_like(t_pred)

    elif flask_id == 3:  # Abiotic control (chemical leaching only)
        rec_base = 0.15 * predict_metal_recovery_dynamic(t_pred, pH_base, 0.05)
        rec_anom = 0.15 * predict_metal_recovery_dynamic(t_pred, pH_anom, 0.05)


    logs = []

    ph_rows = [{"Time":t_obs[i],"pH":pH_obs[i],"Type":"Observed"} for i in range(len(t_obs))]
    ph_b, ph_a, rb, ra = [], [], [], []

    for i in range(len(t_pred)):

        fault = None
        if anomaly_type!="None" and len(idx)>0 and i>=idx[0]:
            fault = fault_map.get(anomaly_type)
            logs.append(random.choice(ERROR_LIBRARY[anomaly_type]))
        else:
           if flask_id == 1:
                logs.append("🟢 Bio-leaching reactor stable")
            elif flask_id == 2:
                logs.append("🟢 Biological control – no metal present")
            else:
                logs.append("🟢 Abiotic chemical leaching active")


        flow_box.graphviz_chart(render_bio_flowsheet("run", fault), use_container_width=True)

        ph_b.append({"Time":t_pred[i],"pH":pH_base[i],"Type":"Baseline"})
        rb.append({"Time":t_pred[i],"Recovery":rec_base[i],"Type":"Baseline"})

        if anomaly_type!="None":
            ph_a.append({"Time":t_pred[i],"pH":pH_anom[i],"Type":"Anomaly"})
            ra.append({"Time":t_pred[i],"Recovery":rec_anom[i],"Type":"Anomaly"})

        ph_df = pd.DataFrame(ph_rows + ph_b + ph_a)
        rec_df = pd.DataFrame(rb + ra)

        ph_plot.altair_chart(
            alt.Chart(ph_df).mark_line().encode(x="Time",y="pH",color="Type").properties(height=260),
            use_container_width=True
        )

        rec_plot.altair_chart(
            (alt.Chart(rec_df).mark_line().encode(x="Time",y="Recovery",color="Type") +
             alt.Chart(rec_df).mark_circle(size=60).encode(x="Time",y="Recovery",color="Type")).properties(height=260),
            use_container_width=True
        )

        log_html = "<br>".join(logs[-12:])
        log_box.markdown(
            f"<div style='height:220px; overflow-y:auto; color:white'>{log_html}</div>",
            unsafe_allow_html=True
        )

        time.sleep(0.05)

    flow_box.graphviz_chart(render_bio_flowsheet(), use_container_width=True)

    kpi1.metric("RMSE", f"{rmse:.3f}")
    kpi2.metric("R²", f"{r2:.3f}")
    kpi3.metric("Final Recovery (%)", f"{rec_base[-1]:.2f}")
    kpi4.metric("Avg OD", f"{OD:.2f}")

# =========================================================
# ================= OPTIMIZATION ==========================
# =========================================================

if optimize_button:

    if flask_id != 1:
        st.warning("⚠ Optimization is only valid for Bio-leaching system (Flask 1). Control experiments do not represent metal recovery processes.")

        kpi1.metric("Optimal T (K)", "N/A")
        kpi2.metric("Optimal PD", "N/A")
        kpi3.metric("Max Recovery (%)", "N/A")
        kpi4.metric("Grid Points", "N/A")

    else:
        flask_df = clean_df[clean_df["Flask_ID"]==flask_id]
        t_obs = flask_df["Time_days"].values
        pH_obs = flask_df["pH"].values
        OD = np.nanmean(flask_df["OD600"].values)
        params = fit_ph_model(t_obs, pH_obs)

        T_vals = np.linspace(290,330,15)
        PD_vals = np.linspace(0.05,0.25,15)

        best_T,best_PD,best_rec,grid = optimize_operating_conditions(
            t_obs,pH_obs,params,OD,T_vals,PD_vals,future_days
        )

        df = pd.DataFrame(grid, columns=["T","PD","Recovery"])

        opt_plot.altair_chart(
            alt.Chart(df).mark_rect().encode(
                x="T:O", y="PD:O",
                color=alt.Color("Recovery:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=["T","PD","Recovery"]
            ).properties(height=300),
            use_container_width=True
        )

        kpi1.metric("Optimal T (K)", f"{best_T:.1f}")
        kpi2.metric("Optimal PD", f"{best_PD:.2f}")
        kpi3.metric("Max Recovery (%)", f"{best_rec:.2f}")
        kpi4.metric("Grid Points", f"{len(grid)}")
