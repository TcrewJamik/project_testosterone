import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import requests
import io  # Импорт для работы с байтовыми потоками

# Заголовок приложения
st.title("Анализ дефицита тестостерона")

# Загрузка данных напрямую из raw ссылки GitHub
raw_url = "https://raw.githubusercontent.com/TcrewJamik/project_testosterone/refs/heads/master/ptestost.xlsx"  # Замените на вашу raw ссылку

try:
    response = requests.get(raw_url)
    response.raise_for_status()  # Проверка на ошибки HTTP
    excel_file = io.BytesIO(response.content)  # Чтение контента в BytesIO
    df = pd.read_excel(excel_file)  # Чтение из BytesIO объекта как из файла

    df = df.rename(columns={
        'Age': 'Возраст',
        'DM': 'Наличие Диабета',
        'TG': 'Триглицериды (мг/дл)',
        'HT': 'Наличие Гипертонии',
        'HDL': 'HDL_холестерин',
        'AC': 'Окружность_талии',
        'T': 'Дефицит Тестостерона'
    })
    st.write("Данные успешно загружены из raw ссылки GitHub!")
except FileNotFoundError:
    st.error("Файл ptestost.xlsx не найден локально.")
    st.stop()
except requests.exceptions.RequestException as e:
    st.error(f"Ошибка при загрузке данных из URL: {e}")
    st.stop()
except Exception as e:
    st.error(f"Произошла ошибка при чтении Excel файла из URL: {e}")
    st.stop()

# Информация о данных
st.subheader("Информация о данных")
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())
st.write("Описательная статистика:")
st.write(df.describe())
st.write("Пропущенные значения:")
st.write(df.isna().sum())

# Визуализация данных
st.subheader("Визуализация данных")
st.write("Гистограммы распределения признаков:")
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for i, column in enumerate(df.columns):
    sns.histplot(df[column], bins=30, kde=True, ax=axes[i // 3, i % 3])
    axes[i // 3, i % 3].set_title(f'Распределение {column}')
plt.tight_layout()
st.pyplot(fig)

st.write("Box Plot:")
fig = plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.title("Box Plot")
st.pyplot(fig)

st.write("Столбчатые диаграммы:")
for col in df.columns:
    fig = plt.figure(figsize=(8, 6))
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Bar Plot для {col}")
    plt.ylabel("Частота")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Корреляция
st.subheader("Корреляционный анализ")
correlation_matrix = df.corr()
st.write("Матрица корреляций:")
st.write(correlation_matrix)
fig = plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Тепловая карта корреляций")
st.pyplot(fig)

# Подготовка данных
target = df['Дефицит Тестостерона']
X = df.drop(columns=['Дефицит Тестостерона'])
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
feature_names = X.columns

# Функция для оценки модели
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    st.write(f"\n{model_name} - Метрики с порогом 0.5:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.4f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")
    st.write(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    st.write(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # ROC-кривая
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-кривая - {model_name}')
    plt.legend(loc="lower right")
    st.pyplot(fig)

    # Матрица ошибок
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Матрица ошибок - {model_name}')
    st.pyplot(fig)

    # Важность признаков
    if model_name == "Логистическая регрессия":
        importance = np.abs(model.coef_[0])
    else:
        importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    st.write(f"\n{model_name} - Важность признаков:")
    st.write(feature_importance)
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', legend=False)
    plt.title(f'Важность признаков - {model_name}')
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    st.pyplot(fig)

# Обучение и оценка моделей
st.subheader("Обучение и оценка моделей")

# Логистическая регрессия с выбранными гиперпараметрами
lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1, class_weight='balanced')
st.write("Логистическая регрессия - Используем гиперпараметры: C=1, class_weight='balanced'")
evaluate_model(lr_model, X_train_scaled, X_test_scaled, y_train, y_test, "Логистическая регрессия", feature_names)

# CatBoost с выбранными гиперпараметрами
cb_model = CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced',
                              depth=6, iterations=200, learning_rate=0.1)
st.write("CatBoost - Используем гиперпараметры: depth=6, iterations=200, learning_rate=0.1")
evaluate_model(cb_model, X_train_scaled, X_test_scaled, y_train, y_test, "CatBoost", feature_names)

# XGBoost с выбранными гиперпараметрами
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss',
                          max_depth=6, n_estimators=200, learning_rate=0.1, scale_pos_weight=1)
st.write("XGBoost - Используем гиперпараметры: max_depth=6, n_estimators=200, learning_rate=0.1, scale_pos_weight=1")
evaluate_model(xgb_model, X_train_scaled, X_test_scaled, y_train, y_test, "XGBoost", feature_names)

# Random Forest с выбранными гиперпараметрами
rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10,
                                  min_samples_split=2, min_samples_leaf=1, class_weight='balanced')
st.write("Random Forest - Используем гиперпараметры: n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1, class_weight='balanced'")
evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest", feature_names)
