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

# ------------------------ ИНФОРМАЦИЯ О ДАННЫХ ------------------------
with st.expander("Информация о данных"):
    st.dataframe(df.head())
    st.write("Описательная статистика:")
    st.dataframe(df.describe())
    st.write("Пропущенные значения:")
    st.write(df.isna().sum())

# ------------------------ ВИЗУАЛИЗАЦИЯ ------------------------
with st.expander("Визуализация данных"):
    st.subheader("Гистограммы распределения признаков")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, column in enumerate(df.columns[:6]):  # пример для первых 6 столбцов
        sns.histplot(df[column], bins=30, kde=True, ax=axes[i // 3, i % 3])
        axes[i // 3, i % 3].set_title(f'Распределение: {column}')
    st.pyplot(fig)

    st.subheader("Корреляционный анализ")
    fig_corr = plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(fig_corr)

# ------------------------ ПРЕДОБРАБОТКА ДАННЫХ ------------------------
target = df['Дефицит Тестостерона']
X = df.drop(columns=['Дефицит Тестостерона'])
y = target

# Если есть категориальные столбцы (object), кодируем их
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

    # Выбор модели
    model_choice = st.selectbox(
        "Выберите модель:",
        ["Логистическая регрессия", "CatBoost", "XGBoost", "Decision Tree", "Random Forest"]
    )

    # Гиперпараметры
    hyperparams = {}
    if model_choice == "Логистическая регрессия":
        hyperparams['C'] = st.slider("C (Регуляризация)", 0.001, 10.0, 1.0, step=0.01)
        penalty_options = ['l1', 'l2', 'none']
        hyperparams['penalty'] = st.selectbox("Penalty", penalty_options, index=1)
        solver_options = ['lbfgs', 'liblinear']
        if hyperparams['penalty'] == 'l1':
            solver_options = ['liblinear', 'saga']
        elif hyperparams['penalty'] == 'none':
            solver_options = ['lbfgs', 'newton-cg', 'sag', 'saga']
        hyperparams['solver'] = st.selectbox("Solver", solver_options, index=0)

    elif model_choice == "CatBoost":
        hyperparams['depth'] = st.slider("Глубина (depth)", 4, 10, 6, step=1)
        hyperparams['iterations'] = st.slider("Iterations", 50, 300, 200, step=10)
        hyperparams['learning_rate'] = st.slider("Learning rate", 0.01, 0.5, 0.1, step=0.01)

    elif model_choice == "XGBoost":
        hyperparams['max_depth'] = st.slider("max_depth", 3, 10, 6, step=1)
        hyperparams['n_estimators'] = st.slider("n_estimators", 50, 300, 200, step=10)
        hyperparams['learning_rate'] = st.slider("learning_rate", 0.01, 0.5, 0.1, step=0.01)
        hyperparams['scale_pos_weight'] = st.slider("scale_pos_weight", 1, 10, 1, step=1)

    elif model_choice == "Decision Tree":
        hyperparams['criterion'] = st.selectbox("Criterion", ['gini', 'entropy'], index=0)
        hyperparams['max_depth'] = st.slider("max_depth", 1, 20, 5, step=1)
        hyperparams['min_samples_split'] = st.slider("min_samples_split", 2, 20, 2, step=1)
        hyperparams['min_samples_leaf'] = st.slider("min_samples_leaf", 1, 10, 1, step=1)

    elif model_choice == "Random Forest":
        hyperparams['n_estimators'] = st.slider("n_estimators", 50, 300, 200, step=10)
        hyperparams['max_depth'] = st.slider("max_depth", 1, 20, 10, step=1)
        hyperparams['min_samples_split'] = st.slider("min_samples_split", 2, 20, 2, step=1)
        hyperparams['min_samples_leaf'] = st.slider("min_samples_leaf", 1, 10, 1, step=1)

    st.markdown("---")
    st.header("📊 Выбор признаков и единичное предсказание")

    # Выбор признаков
    selected_features = st.multiselect(
        "Выберите признаки для обучения и единичного предсказания:",
        feature_names,
        default=feature_names  # по умолчанию все
    )

    # Кнопка обучения
    train_button = st.button("🔥 Обучить модель")

    st.markdown("---")
    st.subheader("Настройка входных значений для единичного предсказания")

    prediction_data = {}
    if selected_features:
        for feature in selected_features:
            # Диапазон для слайдера
            min_val = float(X_train[feature].min())
            max_val = float(X_train[feature].max())
            default_val = float(X_train[feature].mean())
            prediction_data[feature] = st.slider(
                f"Значение для {feature}:",
                min_val,
                max_val,
                default_val
            )

    # Кнопка единичного предсказания
    single_predict_button = st.button("✨ Предсказать для единичного образца")

# ------------------------ ОСНОВНАЯ ОБЛАСТЬ: ОЦЕНКА МОДЕЛИ ------------------------
if train_button:
    if not selected_features:
        st.warning("Выберите хотя бы один признак для обучения.")
    else:
        # Отбор нужных столбцов
        selected_idx = [feature_names.index(feat) for feat in selected_features]
        X_train_sel = X_train_scaled[:, selected_idx]
        X_test_sel = X_test_scaled[:, selected_idx]

        # Создание классификатора
        if model_choice == "Логистическая регрессия":
            clf = LogisticRegression(random_state=42, max_iter=1000, **hyperparams)
        elif model_choice == "CatBoost":
            clf = CatBoostClassifier(random_state=42, verbose=0, **hyperparams)
        elif model_choice == "XGBoost":
            clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **hyperparams)
        elif model_choice == "Decision Tree":
            clf = DecisionTreeClassifier(random_state=42, **hyperparams)
        elif model_choice == "Random Forest":
            clf = RandomForestClassifier(random_state=42, **hyperparams)
        else:
            clf = LogisticRegression(random_state=42, max_iter=1000)

        clf.fit(X_train_sel, y_train)
        y_pred = clf.predict(X_test_sel)
        y_prob = clf.predict_proba(X_test_sel)[:, 1]

        # Сохраняем результаты в session_state
        st.session_state['clf'] = clf
        st.session_state['selected_features'] = selected_features
        st.session_state['X_test_sel'] = X_test_sel
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['y_prob'] = y_prob

        # Выводим оценку модели
        st.subheader("🏆 Оценка модели")
        st.write(f"**Модель**: {model_choice}")
        st.write("**Гиперпараметры**:", hyperparams)
        st.metric("Точность (Accuracy)", f"{accuracy_score(y_test, y_pred):.3f}")
        st.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.3f}")
        st.metric("F1-мера", f"{f1_score(y_test, y_pred):.3f}")
        st.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
        st.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax_cm)
        ax_cm.set_xlabel("Предсказанные классы")
        ax_cm.set_ylabel("Истинные классы")
        ax_cm.set_title("Матрица ошибок")
        st.pyplot(fig_cm)

        # ROC-кривая
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC-кривая (AUC = {roc_auc_val:.2f})',
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
        )
        fig_roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig_roc.update_traces(fillcolor='rgba(99, 255, 132, 0.6)')
        st.plotly_chart(fig_roc)

        st.subheader("Отчет о классификации")
        st.text(classification_report(y_test, y_pred))

# ------------------------ ОСНОВНАЯ ОБЛАСТЬ: ЕДИНИЧНОЕ ПРЕДСКАЗАНИЕ ------------------------
if single_predict_button:
    if 'clf' not in st.session_state:
        st.warning("Сначала обучите модель!")
    else:
        # Формируем DataFrame из введённых пользователем значений
        sample_df = pd.DataFrame([prediction_data])
        # Оставляем только выбранные признаки
        sample_sel = sample_df[st.session_state['selected_features']]
        # Масштабируем
        sample_scaled = scaler.transform(sample_sel)

        pred_class = st.session_state['clf'].predict(sample_scaled)
        pred_prob = st.session_state['clf'].predict_proba(sample_scaled)[:, 1]

        # Выводим результат единичного предсказания отдельно
        st.subheader("🔮 Результат единичного предсказания")
        st.write(f"**Предсказанный класс**: {pred_class[0]}")
        st.write(f"**Вероятность класса 1**: {pred_prob[0]:.3f}")

st.markdown("---")
st.markdown("Разработано компанией Jamshed Corporation совместно с ZyplAI")

добавь важность признаков
