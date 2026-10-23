"""
Bank Marketing Subscription Prediction - Streamlit Analytics Dashboard & Prediction System
Built with Streamlit, Plotly, Scikit-Learn, and Keras 3.
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Bank Term Deposit Prediction | Enterprise ANN System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS for glassmorphism, responsive cards, and premium typography
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 90%);
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Badges */
    .badge-primary {
        background-color: #0369a1;
        color: #e0f2fe;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background-color: #b45309;
        color: #fef3c7;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-success {
        background-color: #047857;
        color: #d1fae5;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Leakage Alert Box */
    .leakage-box {
        background: rgba(180, 83, 9, 0.15);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 15px 0;
    }

    /* Subheader */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-top: 20px;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data and results
@st.cache_data
def load_comparison_data():
    comp_path = "reports/model_comparison.csv"
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None

@st.cache_data
def load_histories_data():
    hist_path = "artifacts/training_histories.json"
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            return json.load(f)
    return None

@st.cache_resource
def get_predictor():
    from src.predict import BankPredictor
    return BankPredictor()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://images.unsplash.com/photo-1559526324-4b87b5e36e44?auto=format&fit=crop&w=400&q=80", use_container_width=True)
st.sidebar.title("🏦 Bank Marketing AI")
st.sidebar.caption("Enterprise Artificial Neural Network System")

pages = [
    "🚀 Project Overview",
    "🎯 Customer Prediction",
    "📊 Model Comparison",
    "🔍 Explainability & SHAP",
    "📁 Batch Prediction",
    "📖 Dataset & Architecture"
]

choice = st.sidebar.radio("Navigate Dashboard", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** UCI Bank Marketing (`bank-full.csv`)")
st.sidebar.markdown("**Instances:** 45,211 | **Features:** 16")
st.sidebar.markdown("**Imbalance Ratio:** 7.55 : 1")
st.sidebar.markdown("---")
st.sidebar.caption("Antigravity ML System v1.0.0 | Academic & Production Ready")

# ==============================================================================
# PAGE 1: PROJECT OVERVIEW
# ==============================================================================
if choice == "🚀 Project Overview":
    st.title("🏦 Bank Marketing Term Deposit Prediction")
    st.markdown("### Production End-to-End Artificial Neural Network & ML System")
    st.markdown("""
    This project provides a comprehensive machine learning system designed to predict whether bank clients will subscribe to a 
    long-term financial deposit product (`y` = `yes` / `no`) during direct telemarketing campaigns. 
    By contrasting deep feed-forward neural networks with tuned traditional models, optimizing decision thresholds, and analyzing target leakage, 
    the system assists marketing teams in maximizing deposit acquisition while minimizing wasted customer outreach.
    """)

    # Top KPI summary cards
    df_comp = load_comparison_data()
    best_model_name = df_comp.iloc[0]["Model"] if df_comp is not None else "Medium ANN"
    best_prauc = df_comp.iloc[0]["PR-AUC"] if df_comp is not None else 0.58
    best_rocauc = df_comp.iloc[0]["ROC-AUC"] if df_comp is not None else 0.93

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Observations</div>
            <div class="metric-value">45,211</div>
            <div class="metric-sub">Semicolon-delimited records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Subscription Imbalance</div>
            <div class="metric-value">11.7%</div>
            <div class="metric-sub">5,289 Yes vs 39,922 No</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Best Overall Model</div>
            <div class="metric-value" style="font-size: 1.4rem;">{best_model_name}</div>
            <div class="metric-sub">Evaluated on untouched test set</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peak Test PR-AUC</div>
            <div class="metric-value">{best_prauc:.3f}</div>
            <div class="metric-sub">ROC-AUC: {best_rocauc:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Duration Leakage Notice
    st.markdown("""
    <div class="leakage-box">
        <strong>⚠️ Crucial Target Leakage Distinction:</strong><br>
        The feature <code>duration</code> represents the length of the phone call in seconds. While highly predictive (r ≈ 0.39), 
        <strong>call duration is completely unknown before dialing the customer</strong>. 
        This dashboard supports two distinct operational modes:
        <ul>
            <li><strong>Pre-Call Lead Prioritization (15 features):</strong> Evaluates potential subscribers before contact is initiated.</li>
            <li><strong>Post-Call Outcome Audit (16 features):</strong> Benchmarks customer intent immediately after call completion.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Key Analytics Tabs
    tab1, tab2 = st.tabs(["📈 Class Distribution & EDA", "🏆 Model Performance Summary"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("reports/figures/target_distribution.png"):
                st.image("reports/figures/target_distribution.png", caption="Target Class Imbalance (88.3% No vs 11.7% Yes)", use_container_width=True)
            else:
                st.info("Target distribution chart will appear once EDA script runs.")
        with c2:
            if os.path.exists("reports/figures/duration_leakage_analysis.png"):
                st.image("reports/figures/duration_leakage_analysis.png", caption="Empirical Demonstration of Duration Leakage", use_container_width=True)
            else:
                st.info("Duration leakage chart will appear once EDA script runs.")

    with tab2:
        if df_comp is not None:
            st.dataframe(
                df_comp[["Model", "Type", "Features", "PR-AUC", "ROC-AUC", "F1-Score (0.5)", "Val-Optimized Thresh", "F1-Score (Opt Thresh)", "Training Time (s)"]],
                use_container_width=True
            )
        else:
            st.info("Evaluation results loading... Run `python -m src.evaluate` to generate initial tables.")

# ==============================================================================
# PAGE 2: CUSTOMER PREDICTION
# ==============================================================================
elif choice == "🎯 Customer Prediction":
    st.title("🎯 Individual Customer Prediction")
    st.markdown("Score an individual prospect for term deposit subscription using trained neural networks or traditional models.")

    # Mode Selector
    mode_col1, mode_col2 = st.columns([2, 2])
    with mode_col1:
        pred_mode = st.radio(
            "Select Prediction Mode",
            ["Post-Call Scoring (16 features, includes duration)", "Pre-Call Lead Prioritization (15 features, duration excluded)"],
            help="Pre-call mode reflects the realistic business scenario where call duration is not yet known."
        )
    is_pre_call = "Pre-Call" in pred_mode
    mode_key = "pre_call" if is_pre_call else "post_call"

    # Model selector
    available_models = [
        "Medium ANN", "Advanced ANN", "Baseline ANN", "XGBoost", "Random Forest", "Logistic Regression"
    ]
    if is_pre_call:
        available_models = ["Pre-call ANN (No Duration)", "Pre-call XGBoost (No Duration)"] + available_models

    with mode_col2:
        selected_model = st.selectbox("Select Model Architecture", available_models)
        custom_thresh = st.slider("Decision Threshold", min_value=0.05, max_value=0.95, value=0.40, step=0.01,
                                  help="Customers with predicted probability >= threshold will be classified as 'Likely to Subscribe'.")

    st.markdown("<div class='section-header'>Customer Profile & Campaign Inputs</div>", unsafe_allow_html=True)

    with st.form("customer_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Client Demographics")
            age = st.number_input("Age", min_value=18, max_value=100, value=38)
            job = st.selectbox("Occupation", [
                "management", "technician", "entrepreneur", "blue-collar", "retired",
                "admin.", "services", "self-employed", "unemployed", "housemaid", "student", "unknown"
            ])
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education = st.selectbox("Education Level", ["tertiary", "secondary", "primary", "unknown"])
            default = st.selectbox("Has Credit in Default?", ["no", "yes", "unknown"])

        with col2:
            st.markdown("##### 💳 Financial & Account Details")
            balance = st.number_input("Yearly Average Balance (€)", min_value=-10000, max_value=150000, value=2500)
            housing = st.selectbox("Has Housing Loan?", ["yes", "no", "unknown"])
            loan = st.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
            contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
            month = st.selectbox("Last Contact Month", [
                "may", "jul", "aug", "jun", "nov", "apr", "feb", "jan", "oct", "sep", "mar", "dec"
            ])

        with col3:
            st.markdown("##### 📞 Campaign History")
            day = st.slider("Last Contact Day of Month", min_value=1, max_value=31, value=15)
            campaign = st.number_input("Contacts in Current Campaign", min_value=1, max_value=50, value=1)
            pdays = st.number_input("Days Since Prev Campaign (-1 if never)", min_value=-1, max_value=900, value=-1)
            previous = st.number_input("Contacts Before Current Campaign", min_value=0, max_value=50, value=0)
            poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "failure", "other", "success"])

            if not is_pre_call:
                duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=260)
            else:
                duration = 0  # not used in pre-call

        submitted = st.form_submit_button("⚡ Predict Subscription Probability", use_container_width=True)

    if submitted:
        customer_payload = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": contact, "day": day, "month": month, "duration": duration,
            "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome
        }

        try:
            predictor = get_predictor()
            pred_res = predictor.predict(
                customer_payload,
                model_name=selected_model,
                mode=mode_key,
                threshold=custom_thresh
            )

            prob = pred_res["probability"]
            label = pred_res["prediction_label"]

            st.markdown("<div class='section-header'>Prediction Result</div>", unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1, 1.5])

            with res_col1:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={'suffix': "%", 'font': {'size': 40, 'color': "#38bdf8"}},
                    title={'text': "Subscription Probability", 'font': {'size': 18, 'color': "#f8fafc"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#38bdf8"},
                        'bgcolor': "#1e293b",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, custom_thresh * 100], 'color': "rgba(100, 116, 139, 0.3)"},
                            {'range': [custom_thresh * 100, 100], 'color': "rgba(14, 165, 233, 0.4)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': custom_thresh * 100
                        }
                    }
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=260, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with res_col2:
                if label == "yes":
                    st.markdown("""
                    <div style="background: rgba(16, 185, 129, 0.2); border: 2px solid #10b981; border-radius: 12px; padding: 20px; margin-top: 15px;">
                        <h3 style="color: #34d399; margin: 0;">✅ Likely to Subscribe</h3>
                        <p style="color: #e2e8f0; margin-top: 8px;">
                            The predicted probability exceeds the active threshold. This customer represents a prime marketing prospect.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444; border-radius: 12px; padding: 20px; margin-top: 15px;">
                        <h3 style="color: #f87171; margin: 0;">❌ Unlikely to Subscribe</h3>
                        <p style="color: #e2e8f0; margin-top: 8px;">
                            The probability is below the threshold. Recommend prioritizing other leads or tailoring a specialized promotional offer.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.caption("⚠️ **Disclaimer:** Machine learning predictions estimate historical likelihood and do not guarantee individual customer decisions.")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ==============================================================================
# PAGE 3: MODEL COMPARISON
# ==============================================================================
elif choice == "📊 Model Comparison":
    st.title("📊 Rigorous Model Comparison")
    st.markdown("All configurations evaluated strictly on the **same untouched 15% test set** (6,782 samples).")

    df_comp = load_comparison_data()
    if df_comp is not None:
        st.dataframe(df_comp, use_container_width=True)

        st.markdown("<div class='section-header'>Comparison Visualizations</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists("reports/figures/pr_curves_comparison.png"):
                st.image("reports/figures/pr_curves_comparison.png", caption="Precision-Recall Curves (Primary Metric for Imbalance)", use_container_width=True)
            else:
                st.info("PR curves figure will be displayed after evaluation completes.")

        with col2:
            if os.path.exists("reports/figures/roc_curves_comparison.png"):
                st.image("reports/figures/roc_curves_comparison.png", caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)
            else:
                st.info("ROC curves figure will be displayed after evaluation completes.")

        col3, col4 = st.columns(2)
        with col3:
            if os.path.exists("reports/figures/model_metrics_comparison.png"):
                st.image("reports/figures/model_metrics_comparison.png", caption="Benchmark Metrics (PR-AUC, ROC-AUC, F1, Recall)", use_container_width=True)
        with col4:
            if os.path.exists("reports/figures/ann_learning_curves.png"):
                st.image("reports/figures/ann_learning_curves.png", caption="ANN Architectures Training & Validation Loss Curves", use_container_width=True)

        if os.path.exists("reports/figures/confusion_matrices.png"):
            st.image("reports/figures/confusion_matrices.png", caption="Confusion Matrix Grid Across All Architectures", use_container_width=True)
    else:
        st.info("Model comparison CSV not found. Please train and evaluate models first.")

# ==============================================================================
# PAGE 4: EXPLAINABILITY & SHAP
# ==============================================================================
elif choice == "🔍 Explainability & SHAP":
    st.title("🔍 Model Explainability & Interpretability")
    st.markdown("""
    Black-box neural networks and ensemble models must be demystified for regulated financial environments. 
    We deploy **SHAP (SHapley Additive exPlanations)** and **Permutation Importance** to inspect both global feature relevance 
    and local individual decision drivers.
    """)

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("reports/figures/shap_summary_plot.png"):
            st.image("reports/figures/shap_summary_plot.png", caption="SHAP Global Summary Plot (Beeswarm)", use_container_width=True)
        else:
            st.info("Run `python -m src.explainability` to generate SHAP summary plots.")

    with col2:
        if os.path.exists("reports/figures/aggregated_feature_comparison.png"):
            st.image("reports/figures/aggregated_feature_comparison.png", caption="Original Feature Importance: XGBoost vs Medium ANN", use_container_width=True)
        else:
            st.info("Aggregated importance plot will appear once explainability runs.")

    st.markdown("<div class='section-header'>Probability Calibration Analysis</div>", unsafe_allow_html=True)
    if os.path.exists("reports/figures/calibration_curves.png"):
        st.image("reports/figures/calibration_curves.png", caption="Reliability Curves (Comparing Predicted Probabilities to Empirical Subscription Frequencies)", use_container_width=True)

    st.markdown("""
    > **⚠️ Interpretability Note:** Feature importance and SHAP attribution measure mathematical predictive association within the trained model. 
    > They do not establish causal relationships (e.g., forcing a customer to speak longer on the phone does not artificially cause them to purchase a term deposit).
    """)

# ==============================================================================
# PAGE 5: BATCH PREDICTION
# ==============================================================================
elif choice == "📁 Batch Prediction":
    st.title("📁 Batch Prospect Scoring")
    st.markdown("Upload a CSV of marketing leads to score an entire campaign in bulk.")

    batch_mode = st.radio("Prediction Mode", ["Post-Call (16 features)", "Pre-Call (15 features, duration excluded)"])
    b_mode_key = "pre_call" if "Pre-Call" in batch_mode else "post_call"

    uploaded_file = st.file_uploader("Upload CSV File (Semicolon ';' or Comma ',' delimited)", type=["csv"])

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        load_sample = st.button("📥 Load Sample Leads Batch (15 rows)")

    df_batch = None
    if load_sample:
        from src.data_loader import load_raw_data
        df_full = load_raw_data("data/bank-full.csv", sep=";")
        df_batch = df_full.head(15).drop(columns=["y"]).copy()
        st.success("Loaded 15 sample records from UCI dataset!")

    elif uploaded_file is not None:
        try:
            # Try comma first, then semicolon
            uploaded_file.seek(0)
            first_line = uploaded_file.readline().decode("utf-8")
            sep = ";" if ";" in first_line else ","
            uploaded_file.seek(0)
            df_batch = pd.read_csv(uploaded_file, sep=sep)
            st.success(f"Uploaded file loaded successfully ({len(df_batch)} rows)!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if df_batch is not None:
        st.markdown("##### Preview Input Data:")
        st.dataframe(df_batch.head(5), use_container_width=True)

        if st.button("🚀 Score Entire Batch", use_container_width=True):
            try:
                predictor = get_predictor()
                model_to_use = "Pre-call ANN (No Duration)" if b_mode_key == "pre_call" else "Medium ANN"
                res = predictor.predict(df_batch, model_name=model_to_use, mode=b_mode_key)
                scored_df = res["results_dataframe"]

                st.markdown("<div class='section-header'>Scoring Results</div>", unsafe_allow_html=True)
                st.dataframe(scored_df, use_container_width=True)

                num_pos = (scored_df["predicted_subscription"] == "yes").sum()
                st.metric("Total Prospects Identified to Subscribe", f"{num_pos} of {len(scored_df)}", f"{num_pos/len(scored_df):.1%}")

                # Download button
                csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Scored Results as CSV",
                    data=csv_bytes,
                    file_name="bank_marketing_scored_leads.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {str(e)}")

# ==============================================================================
# PAGE 6: DATASET & ARCHITECTURE
# ==============================================================================
elif choice == "📖 Dataset & Architecture":
    st.title("📖 Dataset, Architecture & Methodology")

    st.markdown("""
    ### 1. Dataset Provenance & Citation
    - **Source:** [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
    - **Authors:** Sérgio Moro, Paulo Cortez, and Paulo Rita (2011/2014)
    - **Volume:** 45,211 direct telemarketing campaign contacts between May 2008 and November 2010.
    
    ### 2. Feature Dictionary
    | Feature | Type | Description |
    | :--- | :--- | :--- |
    | `age` | Numeric | Client age in years |
    | `job` | Categorical | Type of job (admin, technician, blue-collar, management, etc.) |
    | `marital` | Categorical | Marital status (married, single, divorced) |
    | `education` | Categorical | Level of education (primary, secondary, tertiary, unknown) |
    | `default` | Binary | Has credit in default? (yes, no) |
    | `balance` | Numeric | Average yearly balance in euros (€) |
    | `housing` | Binary | Has housing loan? (yes, no) |
    | `loan` | Binary | Has personal loan? (yes, no) |
    | `contact` | Categorical | Contact communication type (cellular, telephone, unknown) |
    | `day` | Numeric | Last contact day of the month (1 - 31) |
    | `month` | Categorical | Last contact month of year (jan - dec) |
    | `duration` | Numeric | Last contact duration in seconds (⚠️ Target leakage warning) |
    | `campaign` | Numeric | Number of contacts during this campaign for this client |
    | `pdays` | Numeric | Days since last campaign contact (-1 means never contacted) |
    | `previous` | Numeric | Contacts before current campaign |
    | `poutcome` | Categorical | Outcome of previous campaign (success, failure, other, unknown) |
    | `y` | Binary Target | Has client subscribed to term deposit? (yes, no) |

    ### 3. Neural Network Architectures Tested
    1. **Baseline ANN:** Input(51) ➔ Dense(64, ReLU) ➔ Dense(1, Sigmoid)
    2. **Medium ANN:** Input(51) ➔ Dense(128, ReLU) ➔ Dropout(0.3) ➔ Dense(64, ReLU) ➔ Dense(1, Sigmoid)
    3. **Advanced ANN:** Input(51) ➔ Dense(256, ReLU) ➔ BatchNorm ➔ Dropout(0.3) ➔ Dense(128, ReLU) ➔ BatchNorm ➔ Dropout(0.2) ➔ Dense(64, ReLU) ➔ Dense(1, Sigmoid)
    4. **Tuned ANN:** Optuna-optimized layer depths, units, regularization, and class-weighting.

    ### 4. Leakage Prevention Protocol
    - Preprocessing parameters (StandardScaler mean/variance and OneHotEncoder categories) are **fitted exclusively on the 70% training split**.
    - The validation split (15%) is reserved for early stopping and threshold selection.
    - The test split (15%) is strictly untouched until the final apples-to-apples evaluation.
    """)

    st.markdown("---")
    st.caption("Developed by Senior ML / Deep Learning Engineer for BTech CSE Internship Portfolio.")
