import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import matplotlib.pyplot as plt

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




# ================= DOWNLOAD FUNCTION (UNCHANGED) =================
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    data = csv.encode()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href


# ========================= HEADER =========================
st.title("❤️ Heart Disease Prediction App")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Project Information")
    st.caption("This dashboard predicts heart disease risk using multiple machine learning models.")

    st.divider()
    st.subheader("What This Project Includes")
    st.markdown("""
    - Single patient prediction
    - Bulk CSV prediction
    - Model accuracy comparison
    """)

    st.subheader("Models Used")
    st.markdown("""
    - Logistic Regression
    - Random Forest
    - Support Vector Machine
    - Decision Tree
    """)

    st.subheader("Expected Input Features (11)")
    
    "Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,\n"
    "RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope",
        

    st.info("Tip: Use encoded numeric values for bulk CSV prediction.")

# ========================= TABS =========================
tab1, tab2, tab3 = st.tabs(["HOME", "BULK PREDICTION", "MODEL INFORMATION"])


# ===================== TAB 1: SINGLE PREDICTION =====================
with tab1:

    st.subheader("Patient Health Parameters")

    # -------- ROW 1 --------
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, key="age")
    with col2:
        sex = st.selectbox("Sex", options=["Male", "Female"], key="sex")
    with col3:
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            key="cpt"
        )

    # -------- ROW 2 --------
    col4, col5, col6 = st.columns(3)
    with col4:
        resting_blood_pressure = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, key="rbp")
    with col5:
        serum_cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, key="chol")
    with col6:
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"], key="fbs")

    # -------- ROW 3 --------
    col7, col8, col9 = st.columns(3)
    with col7:
        resting_ecg = st.selectbox(
            "Resting ECG Results",
            options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
            key="ecg"
        )
    with col8:
        max_heart_rate = st.number_input("Maximum Heart Rate", min_value=60, max_value=202, key="mhr")
    with col9:
        exercise_induced_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"], key="eia")

    # -------- ROW 4 --------
    col10, col11, col12 = st.columns(3)
    with col10:
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, key="op")
    with col11:
        slope = st.selectbox("Slope of Peak Exercise ST", options=["Upsloping", "Flat", "Downsloping"], key="slope")
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
            models = load_single_prediction_models()
            algonames = list(models.keys())
            predictions = []

            for modelname in algonames:
                model = models[modelname]
                pred = model.predict(input_data)[0]
                predictions.append(pred)

        # Show results in 2x2 grid
        r_col1, r_col2 = st.columns(2)

        for i in range(len(predictions)):
            target_col = r_col1 if i % 2 == 0 else r_col2
            with target_col:
                with st.container(border=True):
                    st.caption(algonames[i])
                    if predictions[i] == 1:
                        st.error("High Risk of Heart Disease")
                    else:
                        st.success("Low Risk of Heart Disease")


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
                * MaxHR: maximum heart rate [60 – 202]
                * ExerciseAngina: [0: No, 1: Yes]
                * Oldpeak: ST depression [numeric]
                * ST_Slope: [0: Upsloping, 1: Flat, 2: Downsloping]""")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="bulk_upload")
    st.caption("Note: Upload an encoded numeric CSV with all 11 required columns and no missing values.")

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = load_model("Models/LogisticR.pkl")

        expected_columns = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]

        if set(expected_columns).issubset(input_data.columns):
            with st.spinner("Processing bulk prediction..."):
                input_data["prediction LR"] = model.predict(input_data[expected_columns].values)
                input_data.to_csv("predictHeartLR.csv")

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

    # Simple matplotlib chart matching original dark background
    bg_color = "#010308"
    bar_colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    bars = ax.bar(Models, Accuracies, color=bar_colors, width=0.8)
    ax.set_title("Model Accuracy Comparison", color="white", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", color="#2f3b4f", alpha=0.6)
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3a4456")
    ax.spines["bottom"].set_color("#3a4456")

    for bar, acc in zip(bars, Accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.0%}", ha="center", color="white")

    plt.tight_layout()
    st.pyplot(fig)
