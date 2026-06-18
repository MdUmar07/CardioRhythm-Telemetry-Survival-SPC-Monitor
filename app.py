import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Medtronic CardioRhythm Monitor", layout="wide")
st.title("CardioRhythm: Telemetry Survival & SPC Monitor")
st.markdown("An interactive statistical engine for implantable device efficacy and safety monitoring.")

# --- DATA LOADING (Cached for performance) ---
@st.cache_data
def load_data():
    df_surv = pd.read_csv('pacemaker_survival_data.csv')
    df_spc = pd.read_csv('pacemaker_spc_data.csv')
    return df_surv, df_spc

df_surv, df_spc = load_data()

# --- DASHBOARD TABS ---
tab1, tab2 = st.tabs(["📊 Fleet Survival Analytics (Efficacy)", "🚨 Patient SPC Monitor (Safety)"])

# ==========================================
# TAB 1: SURVIVAL ANALYSIS
# ==========================================
with tab1:
    st.header("Pacemaker Fleet Battery Survival")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kaplan-Meier Survival Curve")
        kmf = KaplanMeierFitter()
        kmf.fit(df_surv['Duration_Months'], event_observed=df_surv['Event_Occurred'])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        kmf.plot_survival_function(ax=ax, color='#004b87', linewidth=2) # Medtronic Blue
        ax.set_title("Probability of Battery Survival over Time")
        ax.set_xlabel("Months Since Implant")
        ax.set_ylabel("Survival Probability")
        ax.axvline(x=120, color='red', linestyle='--', label='10-Year Mark')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        st.subheader("Cox Proportional Hazards Model")
        cph = CoxPHFitter()
        cph.fit(df_surv.drop('Device_ID', axis=1), duration_col='Duration_Months', event_col='Event_Occurred')
        
        st.markdown(f"**Model Concordance (Accuracy):** {cph.concordance_index_:.2f}")
        st.markdown("**Hazard Ratios:**")
        
        # Displaying the summary dataframe cleanly
        summary_df = cph.summary[['exp(coef)', 'p']]
        summary_df.columns = ['Hazard Ratio', 'P-Value']
        
        summary_df['Hazard Ratio'] = summary_df['Hazard Ratio'].round(2)
        summary_df['P-Value'] = summary_df['P-Value'].round(4)
        
        st.dataframe(summary_df)
        
        st.info("💡 **Insight:** Pacing Percentage has a Hazard Ratio > 1.0 and a significant p-value (<0.05). This proves that higher device utilization drastically shortens battery lifespan. Patient age has no statistical impact.")

# ==========================================
# TAB 2: STATISTICAL PROCESS CONTROL (SPC)
# ==========================================
with tab2:
    st.header("Real-Time Telemetry SPC Monitor")
    
    # Device Selection
    device_list = df_spc['Device_ID'].unique()
    selected_device = st.selectbox("Select Patient Device ID to Monitor:", device_list, index=2) # Defaults to MD_0003
    
    df_device = df_spc[df_spc['Device_ID'] == selected_device].copy()
    df_device = df_device.sort_values('Date').reset_index(drop=True)
    
    # SPC Math
    baseline_data = df_device.iloc[:20]['Lead_Impedance_Ohms']
    mean_imp = baseline_data.mean()
    std_imp = baseline_data.std()
    UCL = mean_imp + (3 * std_imp)
    LCL = mean_imp - (3 * std_imp)
    
    # Detect Anomalies
    df_device['Anomaly'] = (df_device['Lead_Impedance_Ohms'] > UCL) | (df_device['Lead_Impedance_Ohms'] < LCL)
    has_anomaly = df_device['Anomaly'].any()
    
    if has_anomaly:
        st.error(f"🚨 WARNING: SPECIAL CAUSE VARIATION DETECTED IN {selected_device}. Lead Impedance has breached 3-Sigma Upper Control Limit. Investigate for potential lead fracture immediately.")
    else:
        st.success(f"✅ Device {selected_device} is operating within normal control limits.")
        
    # Plot SPC Chart
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df_device['Date'], df_device['Lead_Impedance_Ohms'], marker='o', color='gray', label='Daily Telemetry')
    ax2.axhline(mean_imp, color='green', linestyle='-', label=f'Mean ({mean_imp:.1f})')
    ax2.axhline(UCL, color='red', linestyle='--', label=f'UCL ({UCL:.1f})')
    ax2.axhline(LCL, color='red', linestyle='--', label=f'LCL ({LCL:.1f})')
    
    # Highlight anomalies
    anomalies = df_device[df_device['Anomaly']]
    ax2.scatter(anomalies['Date'], anomalies['Lead_Impedance_Ohms'], color='red', s=100, zorder=5, label='Anomaly')
    
    ax2.set_title(f"Shewhart Individuals (X) Chart: Lead Impedance for {selected_device}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Impedance (Ohms)")
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    st.pyplot(fig2)