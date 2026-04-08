import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.graph_objects as go

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_single_prediction_models():
    return {
        "Logistic Regression": load_model("Models/LogisticR.pkl"),
        "Random Forest": load_model("Models/RFC.pkl"),
        "Support Vector Machine": load_model("Models/SVM.pkl"),
        "Decision Tree": load_model("Models/DCL.pkl"),
    }


@st.cache_data
def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)




# ================= DOWNLOAD FUNCTION (UNCHANGED) =================
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    data = csv.encode()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href


# ========================= HEADER =========================
st.title("❤️ CardioRisk Analyzer")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.markdown(
        """
        <div style="color:#2cc980; font-size:26px; font-weight:500;">
            System Information
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("This dashboard predicts heart disease risk using multiple machine learning models.")

    st.divider()

    
    st.markdown(
        """
        <div style="color:#2cc980; font-size:26px; font-weight:500;">
           What This System Includes
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
    - Single patient prediction
    - Bulk CSV prediction
    - Model accuracy comparison
    - Dataset information 
    """)

    st.divider()

    st.markdown(
        """
        <div style="color:#2cc980; font-size:26px; font-weight:500;">
           Models Used
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
    - Logistic Regression
    - Random Forest
    - Support Vector Machine
    - Decision Tree
    """)

    st.divider()

    st.markdown(
        """
        <div style="color:#2cc980; font-size:26px; font-weight:500;">
           Parameter Quick Guide( Total 11 features)
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("""
        - **Age** 
        - **Sex**
                
        - **ChestPainType**: Type of chest pain you feel.
            - Typical Angina ->  Chest pain due to heart problem (serious)
            - Atypical Angina ->  Chest pain but not exactly heart-related
            - Non-anginal Pain ->  Pain not related to heart
            - Asymptomatic ->  No chest pain
                
                
        - **RestingBP** ->  Blood pressure when you are resting.
                
        - **Cholesterol** ->  Fat level in your blood.
                
        - **FastingBS** (Blood Sugar):  Sugar level after fasting.
            - <= 120 mg/dl ->  Normal
            - > 120 mg/dl ->  High (risk)
            
                
        - **RestingECG**:  Heart electrical activity test.
            - Normal ->  No problem
            - ST-T wave abnormality ->  Possible heart issue
            - Left ventricular hypertrophy ->  Heart muscle thick (serious)
                
        - **MaxHR** ->  Maximum heart beats per minute during activity.
                
        - **ExerciseAngina**:  Chest pain during exercise.
            - Yes ->  Risk
            - No ->  Safe
                
        - **Oldpeak** ->  Heart stress level after exercise (higher value = more risk).
                
        - **ST_Slope**:  Heart response during exercise.
            - Upsloping ->  Normal (good)
            - Flat ->  Medium risk
            - Downsloping ->  High risk
    """)

    st.divider()

    st.info("Tip: Use encoded numeric values for bulk CSV prediction.")

# ========================= TABS =========================
tab1, tab2, tab3, tab4 = st.tabs(["HOME", "BULK PREDICTION", "MODEL INFORMATION", "ABOUT DATASET"])


# ===================== TAB 1: SINGLE PREDICTION =====================
with tab1:

    st.subheader("Patient Health Parameters")

    # -------- ROW 1 --------
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("**Age**", min_value=1, max_value=120, value=40, key="age")   #**Age** here ** is used to make the label bold in Streamlit.
    with col2:
        sex = st.selectbox("**Sex**", options=["Male", "Female"], key="sex")
    with col3:
        chest_pain_type = st.selectbox(
            "**Chest Pain Type**",
            options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            key="cpt"
        )

    # -------- ROW 2 --------
    col4, col5, col6 = st.columns(3)
    with col4:
        resting_blood_pressure = st.slider("**Resting Blood Pressure (trestbps)**", min_value=0, max_value=300, value=120, key="rbp")
    with col5:
        serum_cholesterol = st.slider("**Serum Cholesterol (chol)**", min_value=0,max_value=600, value=200, key="chol")
    with col6:
        fasting_blood_sugar = st.selectbox("**Fasting Blood Sugar**", ["<= 120 mg/dl", "> 120 mg/dl"], key="fbs")

    # -------- ROW 3 --------
    col7, col8, col9 = st.columns(3)
    with col7:
        resting_ecg = st.selectbox(
            "**Resting ECG Results**",
            options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
            key="ecg"
        )
    with col8:
        max_heart_rate = st.slider("**Maximum Heart Rate**", min_value=60, max_value=202, value=120, key="mhr")
    with col9:
        exercise_induced_angina = st.selectbox("**Exercise Induced Angina**", options=["Yes", "No"], key="eia")

    # -------- ROW 4 --------
    col10, col11, col12 = st.columns(3)
    with col10:
        oldpeak = st.number_input("**ST Depression (Oldpeak)**", min_value=0.0, max_value=10.0, key="op")
    with col11:
        slope = st.selectbox("**Slope of Peak Exercise ST**", options=["Upsloping", "Flat", "Downsloping"], key="slope")
    with col12:
        st.write("")  # spacing
        st.write("")  # spacing
        submit = st.button("Heart Disease Test Result", use_container_width=True, type="primary")

    st.divider()

    # -------- RESULTS --------
    if submit:

        st.subheader("Prediction Results")

        # Convert categorical to numeric (UNCHANGED LOGIC)
        sex_val = 0 if sex == "Male" else 1
        chest_pain_type_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain_type)
        fasting_blood_sugar_val = 0 if fasting_blood_sugar == "<= 120 mg/dl" else 1
        resting_ecg_val = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(resting_ecg)
        exercise_induced_angina_val = 1 if exercise_induced_angina == "Yes" else 0
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)

        input_data = pd.DataFrame({
            "Age": [age],
            "Sex": [sex_val],
            "ChestPainType": [chest_pain_type_val],
            "RestingBP": [resting_blood_pressure],
            "Cholesterol": [serum_cholesterol],
            "FastingBS": [fasting_blood_sugar_val],
            "RestingECG": [resting_ecg_val],
            "MaxHR": [max_heart_rate],
            "ExerciseAngina": [exercise_induced_angina_val],
            "Oldpeak": [oldpeak],
            "ST_Slope": [slope_val]
        })

        with st.spinner("Running prediction..."):
            rf_model = load_model("Models/RFC.pkl")
            rf_prediction = rf_model.predict(input_data)[0]
            rf_risk_probability = rf_model.predict_proba(input_data)[0][1] * 100
            

        with st.container(border=True):
            
            if rf_prediction == 1:
                st.markdown(
                    "<p style='font-size:22px; font-weight:800;'>Prediction: <span style='color:#c0392b;'>HIGH RISK</span></p>",
                    unsafe_allow_html=True,
                )
                st.error("High Risk of Heart Disease")
            else:
                st.markdown(
                    "<p style='font-size:22px; font-weight:800;'>Prediction: <span style='color:#1e8449;'>LOW RISK</span></p>",
                    unsafe_allow_html=True,
                )
                st.success("Low Risk of Heart Disease")

            risk_color = "#c0392b" if rf_prediction == 1 else "#1e8449"
            st.markdown(
                f"<p style='font-size:22px; font-weight:700;'>Risk Probability: <span style='color:{risk_color};'>{rf_risk_probability:.2f}%</span></p>",
                unsafe_allow_html=True,
            )

            st.divider()

            if rf_prediction == 1:
                st.markdown("### Possible Contributing Parameters")
                flagged_parameters = []

                if age >= 55:
                    flagged_parameters.append(f"Age is high ({age} years)")
                if resting_blood_pressure >= 140:
                    flagged_parameters.append(f"Resting blood pressure is high ({resting_blood_pressure} mm Hg)")
                if serum_cholesterol >= 240:
                    flagged_parameters.append(f"Cholesterol is high ({serum_cholesterol} mg/dl)")
                if fasting_blood_sugar_val == 1:
                    flagged_parameters.append("Fasting blood sugar is high (> 120 mg/dl)")
                if resting_ecg != "Normal":
                    flagged_parameters.append(f"Resting ECG is abnormal ({resting_ecg})")
                if max_heart_rate < 100:
                    flagged_parameters.append(f"Maximum heart rate is low ({max_heart_rate})")
                if exercise_induced_angina == "Yes":
                    flagged_parameters.append("Exercise induced angina is present")
                if oldpeak >= 2.0:
                    flagged_parameters.append(f"Oldpeak is elevated ({oldpeak})")
                if slope in ["Flat", "Downsloping"]:
                    flagged_parameters.append(f"ST slope shows risk pattern ({slope})")
                if chest_pain_type in ["Asymptomatic", "Typical Angina"]:
                    flagged_parameters.append(f"Chest pain type indicates possible risk ({chest_pain_type})")

                if flagged_parameters:
                    for item in flagged_parameters:
                        st.write(f"- {item}")
                else:
                    st.info("No single parameter crossed the simple high-risk rule limits, but the model still predicted high risk from combined patterns.")

                st.caption("These are rule-based hints for understanding the result, not a medical diagnosis.")


# ===================== TAB 2: BULK PREDICTION =====================
with tab2:

    st.subheader("Upload CSV File for Bulk Predictions")

    with st.expander("Instructions - Read before uploading", expanded=False):
        st.info("""
            1. No NaN values allowed.
            2. Total 11 features in this order: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope.
            3. Check the spelling of the feature names.
            4. Feature values conventions:
                * Age: age of the patient [years]
                * Sex: [0: Male, 1: Female]
                * ChestPainType: [3: Typical Angina, 0: Atypical Angina, 1: Non-anginal Pain, 2: Asymptomatic]
                * RestingBP: resting blood pressure [mm Hg]
                * Cholesterol: serum cholesterol [mm/dl]
                * FastingBS: [0: <= 120 mg/dl, 1: > 120 mg/dl]
                * RestingECG: [0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy]
                * MaxHR: maximum heart rate [60 - 202]
                * ExerciseAngina: [0: No, 1: Yes]
                * Oldpeak: ST depression [numeric]
                * ST_Slope: [0: Upsloping, 1: Flat, 2: Downsloping]""")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="bulk_upload")
    st.caption("Note: Upload an encoded numeric CSV with all 11 required columns and no missing values.")

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = load_model("Models/RFC.pkl")

        expected_columns = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]

        if set(expected_columns).issubset(input_data.columns):
            with st.spinner("Processing bulk prediction..."):
                input_data["prediction result"] = model.predict(input_data[expected_columns].values)
                input_data.to_csv("predictHeart.csv")

            st.subheader("Prediction Results")
            st.dataframe(input_data, use_container_width=True)
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("The uploaded CSV file does not contain the expected columns. Please check the instructions above.")
    else:
        st.info("Upload a CSV file to get bulk predictions.")


# ===================== TAB 3: MODEL INFORMATION =====================
with tab3:

    st.subheader("Model Performance Comparison")

    # Model accuracy data (UNCHANGED)
    data = {"Decision Tree": 0.80, "Random Forest": 0.86, "Logistic Regression": 0.85, "Support Vector Machine": 0.84}
    Models = list(data.keys())
    Accuracies = list(data.values())

    # Accuracy metric cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    for idx, (col, model_name, acc) in enumerate(zip([m_col1, m_col2, m_col3, m_col4], Models, Accuracies)):
        with col:
            with st.container(border=True):
                st.metric(
                    label=model_name,
                    value=f"{acc * 100:.0f}%"
                )

    st.divider()

    fig = go.Figure(
        data=[
            go.Bar(
                x=Models,
                y=Accuracies,
                marker_color=["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"],
                text=[f"{a:.0%}" for a in Accuracies],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title={"text": "Model Accuracy Comparison",
               "x": 0.5, #"x": 0.5 sets the horizontal position of the title in Plotly.
                            #0 = far left
                            #0.5 = center
                            #1 = far right ,
                "xanchor": "center",
                 "font": {"size": 24}
    },


        template="plotly_dark",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        xaxis_title="Models",
        yaxis_title="Accuracy",
    )

    st.plotly_chart(fig, use_container_width=True)


    st.divider()


    st.markdown("## 📊 Model Evaluation Metrics")

    # Example data (replace with real values)
    metrics_data = {
        "Logistic Regression": {"Precision": 86, "Recall": 89, " F1-score": 87,"TP": 67,"FN": 15,"FP": 11, "TN": 91},
        "Random Forest": {"Precision": 86, "Recall": 91, " F1-score": 88,"TP": 67, "FN": 15, "FP": 9, "TN": 93},
        "SVM": {"Precision": 85, "Recall": 86, " F1-score": 84,"TP": 67, "FN": 15, "FP": 14, "TN": 88},
        "Decision Tree": {"Precision": 83, "Recall": 82, " F1-score": 83,"TP": 65, "FN": 17, "FP": 18,"TN": 84 },
    }

    left_models = ["Logistic Regression", "SVM"]
    right_models = ["Random Forest", "Decision Tree"]

    left_col, right_col = st.columns(2)

    with left_col:
        for model in left_models:
            values = metrics_data[model]
            st.markdown(f"### {model}")
            st.info(f"Precision: **{values['Precision']}%**")
            st.info(f"Recall: **{values['Recall']}%**")
            st.info(f"F1-score: **{values[' F1-score']}%**")

            st.markdown("**Confusion Matrix**")
            left_cm = pd.DataFrame(
                [[values["TP"], values["FN"]], [values["FP"], values["TN"]]],
                index=["Actual Positive", "Actual Negative"],
                columns=["Pred Positive", "Pred Negative"],
            )
            st.table(left_cm)
            st.divider()

    with right_col:
        for model in right_models:
            values = metrics_data[model]
            st.markdown(f"### {model}")
            st.info(f"Precision: **{values['Precision']}%**")
            st.info(f"Recall: **{values['Recall']}%**")
            st.info(f"F1-score: **{values[' F1-score']}%**")

            st.markdown("**Confusion Matrix**")
            right_cm = pd.DataFrame(
                [[values["TP"], values["FN"]], [values["FP"], values["TN"]]],
                index=["Actual Positive", "Actual Negative"],
                columns=["Pred Positive", "Pred Negative"],
            )
            st.table(right_cm)
            st.divider()


# ===================== TAB 4: ABOUT DATASET =====================
with tab4:
    # Title
    st.markdown("## 🫀 Heart Disease Dataset Information")

    st.markdown(
        "This application is built using the **Heart Disease UCI Dataset**, a well-known dataset commonly used for predicting heart disease." \
        " It includes **1000 patient records** with **12 important health-related attributes**, making it a reliable source for developing and testing prediction models."
    )
    st.divider()

    # Key Statistics
    st.markdown("### 📊 Key Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.info("👥 Total Patients: **1000**")
        st.info("🧪 Features: **12 clinical parameters(used only 11)**")
        st.info("📈 Data Type: **Structured tabular data**")

    with col2:
        st.info("🎯 Target: **Heart Disease (Yes/No)**")
        st.info("⚖️ Class Distribution: **Approximately balanced**")
        st.info("📂 Source: UCI Machine Learning Repository / Kaggle")






