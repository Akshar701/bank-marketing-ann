"""
Bank Marketing Subscription Prediction - Streamlit Analytics Dashboard & Prediction System
Built with Streamlit, Plotly, Scikit-Learn, and Keras 3.
"""
import os
from pathlib import Path
from html import escape
import json
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Verdant | Campaign Intelligence",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Keep the presentation layer separate from prediction logic.
st.markdown(f"<style>{(Path(__file__).parent / 'assets/dashboard.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def chart_style(fig, height=330):
    fig.update_layout(template="plotly_white", height=height,
                      font=dict(family="Arial, sans-serif", color="#3e5b4e"),
                      paper_bgcolor="white", plot_bgcolor="white",
                      margin=dict(l=16, r=20, t=24, b=20))
    fig.update_xaxes(tickfont=dict(color="#3e5b4e"), title_font=dict(color="#3e5b4e"), automargin=True)
    fig.update_yaxes(tickfont=dict(color="#3e5b4e"), title_font=dict(color="#3e5b4e"), automargin=True)
    fig.update_layout(legend_font_color="#3e5b4e")
    fig.update_annotations(font_color="#173e31")
    return fig


def navigate(page):
    st.session_state["navigation"] = page


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
st.sidebar.markdown('<div class="brand"><div class="brand-mark">v.</div><div><div class="brand-name">verdant</div><div class="brand-sub">Campaign intelligence</div></div></div>', unsafe_allow_html=True)
pages = ["Overview", "Customer prediction", "Model comparison", "Explainability", "Batch scoring", "Data & methodology"]
choice = st.sidebar.radio("WORKSPACE", pages, key="navigation")
st.sidebar.markdown('<div class="side-note"><strong>Bank marketing research</strong><br>45,211 customer records<br>UCI Bank Marketing dataset<br>Historical benchmark · 9 models</div>', unsafe_allow_html=True)
st.sidebar.caption("Research workspace / v2.0")
st.markdown('<div class="eyebrow">WORKSPACE / ' + choice.upper() + '</div>', unsafe_allow_html=True)

if choice == "Overview":
    st.title("Campaign overview")
    st.caption("Understand your audience. Compare your models. Make more informed calls.")
    st.markdown('<div class="hero"><div class="eyebrow">FROM DATA TO DECISIONS</div><h2>Find the potential in every conversation.</h2><p>Explore subscription patterns and estimate customer interest with your trained bank marketing models.</p></div>', unsafe_allow_html=True)
    a, b, _ = st.columns([1.4, 1.4, 3])
    a.button("Score a customer →", on_click=navigate, args=("Customer prediction",), use_container_width=True, type="primary")
    b.button("Upload a lead list →", on_click=navigate, args=("Batch scoring",), use_container_width=True)
    df_comp = load_comparison_data()
    best = df_comp.sort_values("PR-AUC", ascending=False).iloc[0] if df_comp is not None and not df_comp.empty else None
    cards = [
        ("CUSTOMER RECORDS", "45,211", "Historical campaign observations"),
        ("SUBSCRIPTION RATE", "11.7%", "5,289 customers subscribed"),
        ("TOP MODEL · TEST PR-AUC", str(best["Model"]) if best is not None else "Unavailable", "Ranked by precision-recall area"),
        ("BEST PR-AUC", f'{best["PR-AUC"]:.3f}' if best is not None else "—", "Measured on held-out test data"),
    ]
    for col, (title, value, detail) in zip(st.columns(4), cards):
        col.markdown(f'<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value">{escape(value)}</div><div class="metric-sub">{detail}</div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Model leaderboard")
        st.caption("Test PR-AUC · higher is better")
        if best is not None:
            ranked = df_comp.sort_values("PR-AUC")
            fig = px.bar(ranked, x="PR-AUC", y="Model", orientation="h", text="PR-AUC", color_discrete_sequence=["#278669"])
            fig.update_traces(texttemplate="%{x:.3f}", textposition="outside", textfont_color="#3e5b4e", marker_line_width=0)
            fig.update_xaxes(range=[0, float(ranked["PR-AUC"].max()) * 1.18], title=None, gridcolor="#edf1ef")
            fig.update_yaxes(title=None)
            st.plotly_chart(chart_style(fig, 390), use_container_width=True, theme=None)
        else:
            st.info("No evaluation results are available yet.")
    with right:
        st.subheader("Audience snapshot")
        st.caption("Historical subscription outcomes")
        fig = go.Figure(go.Pie(labels=["Subscribed", "Did not subscribe"], values=[5289, 39922], hole=.76,
                               marker=dict(colors=["#278669", "#e7eeea"]), textinfo="none", sort=False))
        fig.update_layout(annotations=[dict(text="<b>11.7%</b><br>subscribed", x=.5, y=.5, showarrow=False, font_size=24)],
                          legend=dict(orientation="h", y=-.06, x=.5, xanchor="center"))
        st.plotly_chart(chart_style(fig, 390), use_container_width=True, theme=None)
    st.markdown('<div class="leakage-box"><strong>Choose the right moment to score.</strong> Use pre-call models to prioritize leads. Post-call models include call duration, which is only available after a conversation ends.</div>', unsafe_allow_html=True)

# ==============================================================================
# PAGE 2: CUSTOMER PREDICTION
# ==============================================================================
elif choice == "Customer prediction":
    st.title("Customer prediction")
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
        available_models = ["Pre-call ANN (No Duration)", "Pre-call XGBoost (No Duration)"]

    with mode_col2:
        selected_model = st.selectbox("Select Model Architecture", available_models)
        custom_thresh = st.slider("Decision Threshold", min_value=0.05, max_value=0.95, value=0.40, step=0.01,
                                  help="Customers with predicted probability >= threshold will be classified as 'Likely to Subscribe'.")

    st.markdown("<div class='section-header'>Customer Profile & Campaign Inputs</div>", unsafe_allow_html=True)

    with st.form("customer_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#####  Client Demographics")
            age = st.number_input("Age", min_value=18, max_value=100, value=38)
            job = st.selectbox("Occupation", [
                "management", "technician", "entrepreneur", "blue-collar", "retired",
                "admin.", "services", "self-employed", "unemployed", "housemaid", "student", "unknown"
            ])
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education = st.selectbox("Education Level", ["tertiary", "secondary", "primary", "unknown"])
            default = st.selectbox("Has Credit in Default?", ["no", "yes", "unknown"])

        with col2:
            st.markdown("#####  Financial & Account Details")
            balance = st.number_input("Yearly Average Balance (€)", min_value=-10000, max_value=150000, value=2500)
            housing = st.selectbox("Has Housing Loan?", ["yes", "no", "unknown"])
            loan = st.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
            contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
            month = st.selectbox("Last Contact Month", [
                "may", "jul", "aug", "jun", "nov", "apr", "feb", "jan", "oct", "sep", "mar", "dec"
            ])

        with col3:
            st.markdown("#####  Campaign History")
            day = st.slider("Last Contact Day of Month", min_value=1, max_value=31, value=15)
            campaign = st.number_input("Contacts in Current Campaign", min_value=1, max_value=50, value=1)
            pdays = st.number_input("Days Since Prev Campaign (-1 if never)", min_value=-1, max_value=900, value=-1)
            previous = st.number_input("Contacts Before Current Campaign", min_value=0, max_value=50, value=0)
            poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "failure", "other", "success"])

            if not is_pre_call:
                duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=260)
            else:
                duration = 0  # not used in pre-call

        submitted = st.form_submit_button(" Predict Subscription Probability", use_container_width=True)

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
                    number={'suffix': "%", 'font': {'size': 40, 'color': "#15745b"}},
                    title={'text': "Subscription Probability", 'font': {'size': 18, 'color': "#173e31"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#15745b"},
                        'bgcolor': "#edf2ef",
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
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': '#173e31'}, height=260, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True, theme=None)

            with res_col2:
                if label == "yes":
                    st.markdown("""
                    <div style="background: rgba(16, 185, 129, 0.2); border: 2px solid #10b981; border-radius: 12px; padding: 20px; margin-top: 15px;">
                        <h3 style="color: #15745b; margin: 0;"> Likely to Subscribe</h3>
                        <p style="color: #385447; margin-top: 8px;">
                            The predicted probability exceeds the active threshold. This customer represents a prime marketing prospect.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444; border-radius: 12px; padding: 20px; margin-top: 15px;">
                        <h3 style="color: #a93939; margin: 0;"> Unlikely to Subscribe</h3>
                        <p style="color: #385447; margin-top: 8px;">
                            The probability is below the threshold. Recommend prioritizing other leads or tailoring a specialized promotional offer.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.caption(" **Disclaimer:** Machine learning predictions estimate historical likelihood and do not guarantee individual customer decisions.")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ==============================================================================
# PAGE 3: MODEL COMPARISON
# ==============================================================================
elif choice == "Model comparison":
    st.title("Model comparison")
    st.markdown("All configurations evaluated strictly on the **same untouched 15% test set** (6,782 samples).")

    df_comp = load_comparison_data()
    if df_comp is not None:
        st.dataframe(df_comp, use_container_width=True)

        st.markdown("<div class='section-header'>Comparison Visualizations</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists("reports/figures/pr_curves_comparison.png"):
                st.image("reports/figures/pr_curves_comparison.png", caption="Precision-Recall Curves (Primary Metric for Imbalance)", use_column_width=True)
            else:
                st.info("PR curves figure will be displayed after evaluation completes.")

        with col2:
            if os.path.exists("reports/figures/roc_curves_comparison.png"):
                st.image("reports/figures/roc_curves_comparison.png", caption="Receiver Operating Characteristic (ROC) Curves", use_column_width=True)
            else:
                st.info("ROC curves figure will be displayed after evaluation completes.")

        col3, col4 = st.columns(2)
        with col3:
            if os.path.exists("reports/figures/model_metrics_comparison.png"):
                st.image("reports/figures/model_metrics_comparison.png", caption="Benchmark Metrics (PR-AUC, ROC-AUC, F1, Recall)", use_column_width=True)
        with col4:
            if os.path.exists("reports/figures/ann_learning_curves.png"):
                st.image("reports/figures/ann_learning_curves.png", caption="ANN Architectures Training & Validation Loss Curves", use_column_width=True)

        if os.path.exists("reports/figures/confusion_matrices.png"):
            st.image("reports/figures/confusion_matrices.png", caption="Confusion Matrix Grid Across All Architectures", use_column_width=True)
    else:
        st.info("Model comparison CSV not found. Please train and evaluate models first.")

# ==============================================================================
# PAGE 4: EXPLAINABILITY & SHAP
# ==============================================================================
elif choice == "Explainability":
    st.title("Understand the predictions")
    st.markdown("""
    Black-box neural networks and ensemble models must be demystified for regulated financial environments.
    We deploy **SHAP (SHapley Additive exPlanations)** and **Permutation Importance** to inspect both global feature relevance
    and local individual decision drivers.
    """)

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("reports/figures/shap_summary_plot.png"):
            st.image("reports/figures/shap_summary_plot.png", caption="SHAP Global Summary Plot (Beeswarm)", use_column_width=True)
        else:
            st.info("Run `python -m src.explainability` to generate SHAP summary plots.")

    with col2:
        if os.path.exists("reports/figures/aggregated_feature_comparison.png"):
            st.image("reports/figures/aggregated_feature_comparison.png", caption="Original Feature Importance: XGBoost vs Medium ANN", use_column_width=True)
        else:
            st.info("Aggregated importance plot will appear once explainability runs.")

    st.markdown("<div class='section-header'>Probability Calibration Analysis</div>", unsafe_allow_html=True)
    if os.path.exists("reports/figures/calibration_curves.png"):
        st.image("reports/figures/calibration_curves.png", caption="Reliability Curves (Comparing Predicted Probabilities to Empirical Subscription Frequencies)", use_column_width=True)

    st.markdown("""
    > ** Interpretability Note:** Feature importance and SHAP attribution measure mathematical predictive association within the trained model.
    > They do not establish causal relationships (e.g., forcing a customer to speak longer on the phone does not artificially cause them to purchase a term deposit).
    """)

# ==============================================================================
# PAGE 5: BATCH PREDICTION
# ==============================================================================
elif choice == "Batch scoring":
    st.title("Batch scoring")
    st.markdown("Upload a CSV of marketing leads to score an entire campaign in bulk.")

    batch_mode = st.radio("Prediction Mode", ["Post-Call (16 features)", "Pre-Call (15 features, duration excluded)"])
    b_mode_key = "pre_call" if "Pre-Call" in batch_mode else "post_call"

    uploaded_file = st.file_uploader("Upload CSV File (Semicolon ';' or Comma ',' delimited)", type=["csv"])

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        load_sample = st.button(
            "Load Sample Leads Batch (15 rows)",
            disabled=uploaded_file is not None,
            help="Remove the uploaded file to use sample leads."
        )

    df_batch = st.session_state.get("batch_input")
    if load_sample:
        try:
            from src.data_loader import load_raw_data
            df_full = load_raw_data("data/bank-full.csv", sep=";")
            df_batch = df_full.head(15).drop(columns=["y"]).copy()
            st.session_state["batch_input"] = df_batch
            st.session_state.pop("batch_results", None)
            st.success("Loaded 15 sample records from UCI dataset!")
        except Exception as e:
            df_batch = None
            st.error(f"Could not load sample leads: {e}")

    elif uploaded_file is not None:
        # Never fall back to old sample leads when a new upload is invalid.
        df_batch = None
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

    if df_batch is not None and df_batch.empty:
        st.warning("This CSV has no customer rows. Upload a file with at least one customer.")

    if df_batch is not None and not df_batch.empty:
        batch_key = (hashlib.sha256(df_batch.to_csv(index=False).encode("utf-8")).hexdigest(), b_mode_key)
        if st.session_state.get("batch_result_key") != batch_key:
            st.session_state.pop("batch_results", None)
        st.markdown("##### Preview Input Data:")
        st.dataframe(df_batch.head(5), use_container_width=True)

        if st.button(" Score Entire Batch", use_container_width=True):
            try:
                predictor = get_predictor()
                model_to_use = "Pre-call ANN (No Duration)" if b_mode_key == "pre_call" else "Medium ANN"
                res = predictor.predict(df_batch, model_name=model_to_use, mode=b_mode_key)
                st.session_state["batch_results"] = res["results_dataframe"]
                st.session_state["batch_result_key"] = batch_key
            except Exception as e:
                st.session_state.pop("batch_results", None)
                st.error(f"Batch prediction failed: {str(e)}")

        if "batch_results" in st.session_state:
            scored_df = st.session_state["batch_results"]
            st.markdown("<div class='section-header'>Scoring Results</div>", unsafe_allow_html=True)
            st.dataframe(scored_df, use_container_width=True)
            num_pos = (scored_df["predicted_subscription"] == "yes").sum()
            st.metric("Total Prospects Identified to Subscribe", f"{num_pos} of {len(scored_df)}", f"{num_pos/len(scored_df):.1%}")
            st.download_button(
                label="Download Scored Results as CSV",
                data=scored_df.to_csv(index=False).encode("utf-8"),
                file_name="bank_marketing_scored_leads.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==============================================================================
# PAGE 6: DATASET & ARCHITECTURE
# ==============================================================================
elif choice == "Data & methodology":
    st.title("Data & methodology")

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
    | `duration` | Numeric | Last contact duration in seconds ( Target leakage warning) |
    | `campaign` | Numeric | Number of contacts during this campaign for this client |
    | `pdays` | Numeric | Days since last campaign contact (-1 means never contacted) |
    | `previous` | Numeric | Contacts before current campaign |
    | `poutcome` | Categorical | Outcome of previous campaign (success, failure, other, unknown) |
    | `y` | Binary Target | Has client subscribed to term deposit? (yes, no) |

    ### 3. Neural Network Architectures Tested
    1. **Baseline ANN:** Input(51)  Dense(64, ReLU)  Dense(1, Sigmoid)
    2. **Medium ANN:** Input(51)  Dense(128, ReLU)  Dropout(0.3)  Dense(64, ReLU)  Dense(1, Sigmoid)
    3. **Advanced ANN:** Input(51)  Dense(256, ReLU)  BatchNorm  Dropout(0.3)  Dense(128, ReLU)  BatchNorm  Dropout(0.2)  Dense(64, ReLU)  Dense(1, Sigmoid)
    4. **Tuned ANN:** Optuna-optimized layer depths, units, regularization, and class-weighting.

    ### 4. Leakage Prevention Protocol
    - Preprocessing parameters (StandardScaler mean/variance and OneHotEncoder categories) are **fitted exclusively on the 70% training split**.
    - The validation split (15%) is reserved for early stopping and threshold selection.
    - The test split (15%) is strictly untouched until the final apples-to-apples evaluation.
    """)

    st.markdown("---")
    st.caption("Verdant / Bank marketing research workspace")
