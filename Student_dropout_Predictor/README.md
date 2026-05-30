# 🎓 Student Early Predictor System

An early-warning tool that helps school staff identify students at risk of dropping out using Machine Learning.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (first-time setup)
```bash
python train_model.py
```
This generates:
- `dropout_model.joblib` — the trained ML model
- `sample_students.csv`  — sample data for upload testing

### 3. Run the web app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔑 Login Credentials

| Username | Password  |
|----------|-----------|
| admin    | admin123  |
| staff    | staff2024 |
| demo     | demo      |

---

## 📁 Project Structure

```
dropout_system/
├── app.py                  # Main Streamlit application
├── train_model.py          # ML training script
├── requirements.txt        # Python dependencies
├── dropout_model.joblib    # Saved model (after training)
├── sample_students.csv     # Sample CSV (after training)
└── README.md
```

---

## 🤖 Machine Learning

Three models are trained and compared:
- **Logistic Regression**
- **Random Forest** ✅ (usually best)
- **Decision Tree**

Best model by AUC-ROC is automatically selected and saved.

### Features used
| Feature             | Type        |
|---------------------|-------------|
| Age                 | Numeric     |
| Gender              | Categorical |
| GPA                 | Numeric     |
| Attendance Rate     | Numeric     |
| Year of Study       | Ordinal     |
| Assignment Score    | Numeric     |
| Test Score          | Numeric     |
| Engagement Score    | Numeric     |
| Financial Support   | Categorical |

---

## 📊 Risk Levels

| Level   | Probability | Action              |
|---------|-------------|---------------------|
| 🟢 Low  | < 35%       | Regular monitoring  |
| 🟡 Medium | 35–65%   | Counselling session |
| 🔴 High | > 65%       | Immediate intervention |

---

## 📄 CSV Upload Format

Required columns: `age, gender, gpa, attendance_rate, year_of_study, assignment_score, test_score, engagement_score, financial_support`

- `gender`: Male / Female  
- `financial_support`: Low / Medium / High  
- `year_of_study`: 1, 2, 3, or 4  

Download the sample CSV from the Upload page to get started.
