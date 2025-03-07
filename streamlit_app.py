import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, precision_score, 
                             recall_score, roc_curve, auc, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import requests
import io

# Настройка страницы
st.set_page_config(page_title="Анализ дефицита тестостерона", page_icon="⚙️", layout="wide")
st.title("Анализ дефицита тестостерона")

# ------------------------ ЗАГРУЗКА ДАННЫХ ------------------------
raw_url = "https://raw.githubusercontent.com/TcrewJamik/project_testosterone/refs/heads/master/ptestost.xlsx"
try:
    response = requests.get(raw_url)
    response.raise_for_status()
    excel_file = io.BytesIO(response.content)
    df = pd.read_excel(excel_file)
    
    # Переименование столбцов для удобства
    df = df.rename(columns={
        'Age': 'Возраст',
        'DM': 'Наличие Диабета',
        'TG': 'Триглицериды (мг/дл)',
        'HT': 'Наличие Гипертонии',
        'HDL': 'HDL_холестерин',
        'AC': 'Окружность_талии',
        'T': 'Дефицит Тестостерона'
    })
    st.success("Данные успешно загружены из GitHub!")
except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    st.stop()

# ------------------------ ПРЕДОБРАБОТКА ДАННЫХ ------------------------
target = df['Дефицит Тестостерона']
X = df.drop(columns=['Дефицит Тестостерона'])
y = target

# Кодируем категориальные признаки
non_numeric = X.select_dtypes(include=['object']).columns
if len(non_numeric) > 0:
    le = LabelEncoder()
    for col in non_numeric:
        X[col] = le.fit_transform(X[col])

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
feature_names = X.columns.tolist()

# ------------------------ БОКОВАЯ ПАНЕЛЬ ------------------------
with st.sidebar:
    st.header("🛠️ Настройки модели")
    model_choice = st.selectbox(
        "Выберите модель:",
        ["Логистическая регрессия", "CatBoost", "XGBoost", "Decision Tree", "Random Forest"]
    )
    
    train_button = st.button("🔥 Обучить модель")

# ------------------------ ОБУЧЕНИЕ МОДЕЛИ ------------------------
if train_button:
    clf = None
    if model_choice == "Логистическая регрессия":
        clf = LogisticRegression(random_state=42, max_iter=1000)
    elif model_choice == "CatBoost":
        clf = CatBoostClassifier(random_state=42, verbose=0)
    elif model_choice == "XGBoost":
        clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    elif model_choice == "Decision Tree":
        clf = DecisionTreeClassifier(random_state=42)
    elif model_choice == "Random Forest":
        clf = RandomForestClassifier(random_state=42)

    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    st.subheader("🏆 Оценка модели")
    st.metric("Точность (Accuracy)", f"{accuracy_score(y_test, y_pred):.3f}")
    st.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.3f}")
    st.metric("F1-мера", f"{f1_score(y_test, y_pred):.3f}")

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax_cm)
    ax_cm.set_xlabel("Предсказанные классы")
    ax_cm.set_ylabel("Истинные классы")
    ax_cm.set_title("Матрица ошибок")
    st.pyplot(fig_cm)
    
    # Важность признаков для случайного леса, XGBoost и CatBoost
    if model_choice in ["Random Forest", "XGBoost", "CatBoost"]:
        st.subheader("📊 Важность признаков")
        importances = clf.feature_importances_
        imp_df = pd.DataFrame({'Признак': feature_names, 'Важность': importances})
        imp_df = imp_df.sort_values(by='Важность', ascending=False)
        
        fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
        sns.barplot(x=imp_df['Важность'], y=imp_df['Признак'], palette='coolwarm', ax=ax_imp)
        ax_imp.set_title("Важность признаков")
        st.pyplot(fig_imp)
