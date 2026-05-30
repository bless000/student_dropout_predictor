"""
Student Dropout Prediction - Model Training Script
Generates synthetic data, trains models, and saves the best one.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def generate_synthetic_data(n=1200):
    """Generate realistic synthetic student dataset."""
    n_dropout = int(n * 0.32)
    n_active  = n - n_dropout

    def make_students(n, dropout):
        gpa   = np.random.normal(1.8 if dropout else 3.2, 0.5, n).clip(0, 4.0)
        att   = np.random.normal(0.62 if dropout else 0.88, 0.12, n).clip(0.3, 1.0)
        age   = np.random.randint(15, 26, n)
        year  = np.random.choice([1, 2, 3, 4], n, p=[0.35, 0.30, 0.20, 0.15] if dropout else [0.25,0.28,0.25,0.22])
        assign= np.random.normal(45 if dropout else 75, 15, n).clip(0, 100)
        test  = np.random.normal(42 if dropout else 72, 14, n).clip(0, 100)
        engage= np.random.normal(3.0 if dropout else 7.5, 1.8, n).clip(1, 10)
        gender= np.random.choice(['Male','Female'], n)
        fin   = np.random.choice(['Low','Medium','High'], n, p=[0.55,0.30,0.15] if dropout else [0.20,0.45,0.35])
        return pd.DataFrame({
            'age': age, 'gender': gender,
            'gpa': gpa.round(2), 'attendance_rate': att.round(3),
            'year_of_study': year,
            'assignment_score': assign.round(1),
            'test_score': test.round(1),
            'engagement_score': engage.round(1),
            'financial_support': fin,
            'dropout': int(dropout)
        })

    df = pd.concat([make_students(n_dropout, True), make_students(n_active, False)], ignore_index=True)
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def preprocess(df):
    df = df.copy()
    le_gender  = LabelEncoder()
    le_finance = LabelEncoder()
    df['gender']           = le_gender.fit_transform(df['gender'])
    df['financial_support']= le_finance.fit_transform(df['financial_support'])
    return df, le_gender, le_finance


def train_and_save():
    print("Generating dataset...")
    df = generate_synthetic_data(1200)
    df.to_csv('/home/claude/dropout_system/sample_students.csv', index=False)

    df_enc, le_gender, le_finance = preprocess(df)
    X = df_enc.drop('dropout', axis=1)
    y = df_enc['dropout']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Random Forest':       RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        Xtr = X_train_s if name == 'Logistic Regression' else X_train
        Xte = X_test_s  if name == 'Logistic Regression' else X_test
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        proba = model.predict_proba(Xte)[:,1]
        acc   = accuracy_score(y_test, preds)
        auc   = roc_auc_score(y_test, proba)
        results[name] = {'model': model, 'acc': acc, 'auc': auc,
                         'uses_scaler': name == 'Logistic Regression'}
        print(f"{name:22s} | Acc: {acc:.4f} | AUC: {auc:.4f}")

    best_name = max(results, key=lambda k: results[k]['auc'])
    print(f"\nBest model: {best_name}")

    bundle = {
        'model':        results[best_name]['model'],
        'scaler':       scaler,
        'uses_scaler':  results[best_name]['uses_scaler'],
        'le_gender':    le_gender,
        'le_finance':   le_finance,
        'feature_cols': list(X.columns),
        'model_name':   best_name,
        'all_results':  {k: {'acc': v['acc'], 'auc': v['auc']} for k, v in results.items()},
    }
    joblib.dump(bundle, '/home/claude/dropout_system/dropout_model.joblib')
    print("Model saved to dropout_model.joblib")
    return bundle


if __name__ == '__main__':
    train_and_save()
