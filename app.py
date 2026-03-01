import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

#function to create a download link for a Datafram as a Csv file
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    data = csv.encode()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href

st.title("Heart Disease Prediction App")
tab1,tab2,tab3 = st.tabs(["predict","Bulk predict","Model information"])


with tab1:
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    chest_pain_type = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    resting_blood_pressure = st.number_input("Resting Blood Pressure", min_value=0, max_value=300)
    serum_cholesterol = st.number_input("Serum Cholesterol", min_value=0)
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar",["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=202)
    exercise_induced_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0)
    slope = st.selectbox("Slope", options=["Upsloping", "Flat", "Downsloping"])

    # convert categorical input to numeric

    sex = 0 if sex == "Male" else 1
    chest_pain_type = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain_type)
    fasting_blood_sugar = 0 if fasting_blood_sugar == "<= 120 mg/dl" else 1
    resting_ecg = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(resting_ecg)
    exercise_induced_angina = 1 if exercise_induced_angina == "Yes" else 0
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope)

    #create a dataframe for the input
    input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "ChestPainType": [chest_pain_type],
    "RestingBP": [resting_blood_pressure],
    "Cholesterol": [serum_cholesterol],
    "FastingBS": [fasting_blood_sugar],
    "RestingECG": [resting_ecg],
    "MaxHR": [max_heart_rate],
    "ExerciseAngina": [exercise_induced_angina],
    "Oldpeak": [oldpeak],
    "ST_Slope": [slope]
})

    algonames = ["Logistic Regression", "Random Forest", "Support Vector Machine", "Decision Tree"]
    modelnames = ["Models/LogisticR.pkl","Models/RFC.pkl","Models/SVM.pkl","Models/DCL.pkl"]


    predictions = []
    def predict_heart_disease(data):
        for modelname in modelnames:
            with open(modelname, "rb") as f:
                model = pickle.load(f)
            pred = model.predict(data)[0]
            predictions.append(pred)
        return predictions
    
    # make prediction when button is clicked
    if st.button("submit"):
        st.subheader("Results...")
        st.markdown("------------------------------------")
        
        result = predict_heart_disease(input_data)

        for i in range(len(predictions)):
            st.subheader(f"{algonames[i]} Prediction:")
            if result[i] == 1:
                st.write(f"{algonames[i]}: High risk of heart disease")
            else:
                st.write(f"{algonames[i]}: Low risk of heart disease")
            
            st.markdown("------------------------------------")


with tab2:
    st.title("Upload CSv File")
    st.subheader("Instructions to note before uploading the file")

    st.info("""
            1.No NaN values allowed.\n
            2.Total 11 features in this order(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope).\n
            3.Check the spelling of the feature names.\n
            4.Feature values conventions: \n
                * Age: age of the patient[years]\n
                * Sex: sex of the patient[0:Male, 1:Female]\n
                * ChestPainType: chest pain type[3:Typical Angina, 0:Atypical Angina, 1:Non-anginal Pain, 2:Asymptomatic]\n
                * RestingBP: resting blood pressure[mm Hg]\n
                * Cholesterol: serum cholesterol[mm/dl]\n
                * FastingBS: fasting blood sugar[0: <= 120 mg/dl, 1: > 120 mg/dl]\n
                * RestingECG: resting electrocardiogram results[0:Normal, 1:ST-T wave abnormality, 2:Left ventricular hypertrophy]\n
                * MaxHR: maximum heart rate achieved[Numeric value between 60 and 202]\n
                * ExerciseAngina: exercise induced angina[0:No, 1:Yes]\n
                * Oldpeak: oldpeak = ST [Numeric value measured in depression]\n
                * ST_Slope: the slope of the peak exercise ST segment[0:Upsloping, 1:Flat, 2:Downsloping]""")
    
    # create a file uploader in the sidebar

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        #read the uploaded CSV file into a dataframe
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open("Models/LogisticR.pkl", "rb"))


        #Ensure that the input Dataframe matches the expected columns and format
        expected_columns =["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]

        if set(expected_columns).issubset(input_data.columns):

            input_data["prediction LR"] = ""

            for i in range(len(input_data)):
                arr = input_data.iloc[i,:-1].values
                input_data["prediction LR"][i] = model.predict([arr])[0]
            input_data.to_csv("predictHeartLR.csv")

            # Display the predictions
            st.subheader("Predictions:")
            st.write(input_data)

            #create a button to download the updated CSV file
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("The uploaded CSV file does not contain the expected columns.")
    else:
        st.info("Please upload a CSV file to get predictions.")



with tab3:
    import plotly.express as px
    data = {"Decision Tree": 0.80, "Random Forest": 0.86, "Logistic Regression": 0.85, "Support Vector Machine": 0.84}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=["Models", "Accuracies"])
    fig = px.bar(df, x="Models", y="Accuracies", title="Model Accuracies")
    st.plotly_chart(fig)