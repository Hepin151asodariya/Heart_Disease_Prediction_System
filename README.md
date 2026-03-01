# 🫀 Heart Disease Prediction

> 🩺 A powerful machine learning web application that predicts the likelihood of heart disease based on patient medical parameters using multiple classification algorithms.

---

## 📖 Project Description

This project leverages **machine learning** to predict whether a patient has heart disease based on various medical attributes such as:

🔹 Age  
🔹 Blood Pressure  
🔹 Cholesterol Levels  
🔹 Cardiac Indicators  

### 🎯 What Can You Do?

✅ **Individual Predictions** - Make predictions for single patients through an intuitive form  
✅ **Bulk Processing** - Upload CSV files to predict for multiple patients at once  
✅ **Model Comparison** - Compare results from 4 different ML models  
✅ **Export Results** - Download prediction results for further analysis  

---

## ✨ Features

### 🤖 Multiple ML Models
Utilizes **4 powerful algorithms** for robust and reliable predictions:

- 🔵 **Logistic Regression** - Fast and interpretable baseline model
- 🌳 **Random Forest Classifier** - Ensemble learning for high accuracy
- 🎯 **Support Vector Machine (SVM)** - Effective with complex boundaries
- 🌲 **Decision Tree Classifier** - Clear decision-making rules

### 💡 Key Capabilities

| Feature | Description |
|---------|-------------|
| 👤 **Single Patient Prediction** | Interactive form for individual patient data input |
| 📊 **Bulk Predictions** | Upload CSV files to predict for multiple patients |
| 🔄 **Model Comparison** | View and compare predictions from all models simultaneously |
| 💾 **CSV Export** | Download prediction results as CSV files |
| 🎨 **User-Friendly Interface** | Clean and intuitive Streamlit dashboard with tabbed navigation |

---

## 🛠️ Technologies Used

| Technology | Purpose | Version |
|-----------|---------|---------|
| 🐍 **Python** | Core programming language | 3.8+ |
| 🤖 **Scikit-learn** | Machine learning library | 1.8.0 |
| 🎈 **Streamlit** | Web application framework | 1.54.0 |
| 🐼 **Pandas** | Data manipulation and analysis | 2.3.3 |
| 🔢 **NumPy** | Numerical computing | 2.4.2 |
| 📦 **Pickle** | Model serialization | Built-in |
| ☁️ **Google Colab** | Cloud-based training environment | Latest |

---

## 📁 Project Structure

```
Heart_Disease_Prediction_System/
│
├── 🚀 app.py                          # Main Streamlit application
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # Project documentation
│
├── 📓 notebooks/
│   └── Heart_Disease_prediction.ipynb    # Google Colab training notebook
│
├── 📊 data/
│   ├── heart (1).csv              # Dataset
│   └── predictHeartLR.csv         # Sample predictions
│
└── 🤖 models/
    ├── LogisticR.pkl              # Logistic Regression model
    ├── RFC.pkl                    # Random Forest model
    ├── SVM.pkl                    # Support Vector Machine model
    └── DCL.pkl                    # Decision Tree model
```

## 📱 Usage

### 👤 Single Patient Prediction

**Step 1:** Navigate to the **🔍 Predict** tab

**Step 2:** Enter patient medical information:

- 📅 **Age**
- 👥 **Sex**
- 💔 **Chest Pain Type**
- 🩸 **Resting Blood Pressure**
- 🧪 **Cholesterol**
- 🍬 **Fasting Blood Sugar**
- 📈 **Resting ECG**
- ❤️ **Max Heart Rate**
- 🏃 **Exercise Induced Angina**
- 📊 **Oldpeak**
- 📉 **ST Slope**

**Step 3:** Click **✅ Submit** to see predictions from all models

### 📊 Bulk Predictions

**Step 1:** Navigate to the **📁 Bulk Predict** tab

**Step 2:** Prepare a CSV file with the following columns (in order):

```
Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, 
RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
```

**Step 3:** 📤 Upload your CSV file

**Step 4:** 👀 View predictions and 💾 download results

---

## 🧠 Model Training

The machine learning models were trained using **Google Colab**:

### 📚 Training Pipeline

| Step | Description |
|------|-------------|
| 1️⃣ **Dataset** | Heart disease dataset with 11 medical features |
| 2️⃣ **Preprocessing** | Data cleaning, feature encoding, and scaling |
| 3️⃣ **Training** | Multiple algorithms trained and evaluated |
| 4️⃣ **Evaluation** | Models compared using accuracy, precision, recall, F1-score |
| 5️⃣ **Serialization** | Trained models saved as `.pkl` files using Pickle |

### 🔄 Want to Retrain the Models?

Follow these steps:

1. 📂 Open `notebooks/Heart_Disease_prediction.ipynb` in Google Colab
2. ▶️ Run all cells sequentially
3. 💾 Download the generated `.pkl` model files
4. 📁 Place them in the project root directory

---

## 📊 Input Features

The models require the following **11 medical features**:

| Feature | Icon | Description | Type |
|---------|------|-------------|------|
| **Age** | 📅 | Patient age in years | 🔢 Numeric |
| **Sex** | 👥 | Gender (0: Male, 1: Female) | 🏷️ Categorical |
| **ChestPainType** | 💔 | Type of chest pain (0-3) | 🏷️ Categorical |
| **RestingBP** | 🩸 | Resting blood pressure (mm Hg) | 🔢 Numeric |
| **Cholesterol** | 🧪 | Serum cholesterol (mg/dl) | 🔢 Numeric |
| **FastingBS** | 🍬 | Fasting blood sugar (0: ≤120, 1: >120 mg/dl) | 🏷️ Categorical |
| **RestingECG** | 📈 | Resting ECG results (0-2) | 🏷️ Categorical |
| **MaxHR** | ❤️ | Maximum heart rate (60-202) | 🔢 Numeric |
| **ExerciseAngina** | 🏃 | Exercise induced angina (0: No, 1: Yes) | 🏷️ Categorical |
| **Oldpeak** | 📊 | ST depression (0.0-10.0) | 🔢 Numeric |
| **ST_Slope** | 📉 | Slope of ST segment (0-2) | 🏷️ Categorical |

---

## 📦 Requirements

### 🔑 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| 🎈 streamlit | ≥1.54.0 | Web application framework |
| 🐼 pandas | ≥2.3.3 | Data manipulation |
| 🔢 numpy | ≥2.4.2 | Numerical computing |
| 🤖 scikit-learn | ≥1.8.0 | Machine learning |
| 📊 plotly | ≥5.0.0 | Data visualization |

📄 See `requirements.txt` for complete list

---

## 🚀 Future Enhancements

Ideas for future improvements:

- 🔍 **Model Explainability** - Add SHAP values to explain predictions
- 📊 **Confidence Scores** - Display prediction confidence percentages
- 📈 **Performance Dashboard** - Real-time model performance monitoring
- 🗄️ **Database Integration** - Track patient history and records
- 🌐 **REST API** - Create API endpoints for system integration
- 📱 **Mobile App** - Develop native mobile application
- 🎨 **Advanced UI** - Enhanced visualizations and themes

---

## ⚠️ Disclaimer

> **🔴 Important Notice:**
>
> This application is designed for **educational and research purposes only**.
>
> ❌ **DO NOT** use this for actual medical diagnosis  
> ❌ **DO NOT** replace professional medical advice  
> ✅ **ALWAYS** consult qualified healthcare professionals for medical concerns
>
> The predictions are based on ML models trained on historical data and should not be considered authoritative medical advice.

---

## 📝 License

This project is available for educational and research purposes.

---

<div align="center">

### 🌟 If you found this project helpful, please consider giving it a star! 🌟

**Made with ❤️ for the healthcare community**

📧 For questions or feedback, feel free to reach out!

---

**Last Updated:** March 2026 | **Version:** 1.0.0

</div>


