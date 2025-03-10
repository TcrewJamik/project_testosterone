import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.title('Предсказание дефицита тестостерона')

# Инициализация состояния сессии
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'last_model_choice' not in st.session_state:
    st.session_state.last_model_choice = None

# --- Функции ---
@st.cache_data
def load_data():
    excel_url = "https://raw.githubusercontent.com/TcrewJamik/project_testosterone/refs/heads/master/ptestost.xlsx"  # Замените на ваш актуальный URL
    try:
        df = pd.read_excel(excel_url)
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке данных из URL: {e}")
        return None

def preprocess_data(df):
    target = df['Дефицит Тестостерона']
    X = df.drop(columns=['Дефицит Тестостерона'])
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = X.columns
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, X_train

def evaluate_model(model, X_test, y_test, model_name, feature_names, best_params):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    st.subheader(f"{model_name} - Метрики с порогом 0.5:")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
    col5.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_proba):.4f}")

    st.write(f"**Лучшие гиперпараметры:**")
    st.json(best_params)

    # ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'ROC Curve - {model_name}')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'], ax=ax_cm)
    ax_cm.set_ylabel('True Label')
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_title(f'Confusion Matrix - {model_name}')
    st.pyplot(fig_cm)

    # Feature Importance
    if model_name == "Логистическая регрессия":
        importance = np.abs(model.coef_[0])
    else:
        importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    st.subheader(f"{model_name} - Feature Importance:")
    st.dataframe(feature_importance_df)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax_fi)
    ax_fi.set_title(f'Feature Importance - {model_name}')
    ax_fi.set_xlabel('Importance')
    ax_fi.set_ylabel('Feature')
    plt.tight_layout()
    st.pyplot(fig_fi)

def predict_deficiency(model, input_data_scaled):
    try:
        proba = model.predict_proba(input_data_scaled)[:, 1]
        return proba[0]
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
        return None

# --- Боковая панель ---
st.sidebar.header("Настройки модели")
model_choice = st.sidebar.selectbox("Выберите модель", ["Логистическая регрессия", "CatBoost", "XGBoost", "Random Forest"])

# Сброс модели при изменении выбора
if st.session_state.last_model_choice != model_choice:
    st.session_state.model_trained = False
    st.session_state.trained_model = None
    st.session_state.last_model_choice = model_choice

st.sidebar.header("Входные параметры")

# --- Основной код ---
df = load_data()

if df is not None:
    df = df.rename(columns={
        'Age': 'Возраст',
        'DM': 'Наличие Диабета',
        'TG': 'Триглицериды (мг/дл)',
        'HT': 'Наличие Гипертонии',
        'HDL': 'HDL_холестерин',
        'AC': 'Окружность_талии',
        'T': 'Дефицит Тестостерона'
    })

    input_age = st.sidebar.slider("Возраст", 45, 85, 60)
    input_diabetes = st.sidebar.selectbox("Наличие Диабета", [0, 1])
    input_triglycerides = st.sidebar.slider("Триглицериды (мг/дл)", 12, 980, 155)
    input_hypertension = st.sidebar.selectbox("Наличие Гипертонии", [0, 1])
    input_hdl = st.sidebar.slider("HDL_холестерин", 13.0, 116.0, 46.3)
    input_waist_circumference = st.sidebar.slider("Окружность_талии", 43.0, 198.0, 98.9)

    st.header('Обзор данных')
    with st.expander("Настройки отображения данных"):
        show_data = st.checkbox("Показать данные", value=False)
        show_info = st.checkbox("Информация о данных", value=False)
        show_describe = st.checkbox("Описательная статистика", value=False)
        show_na = st.checkbox("Проверка на пропуски", value=False)
        show_histograms = st.checkbox("Гистограммы признаков", value=False)
        show_boxplot = st.checkbox("Box plot", value=False)
        show_barplots = st.checkbox("Столбчатые диаграммы", value=False)
        show_correlation = st.checkbox("Матрица корреляций", value=False)

    if show_data:
        st.subheader("Данные")
        st.dataframe(df)
    if show_info:
        st.subheader("Информация о данных")
        st.write(df.info())
    if show_describe:
        st.subheader("Описательная статистика")
        st.write(df.describe())
    if show_na:
        st.subheader("Проверка на пропуски")
        st.write(df.isna().sum())

    st.header('Визуализация данных')
    if show_histograms:
        st.subheader("Гистограммы распределения признаков")
        num_cols = len(df.columns)
        rows = (num_cols + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()
        for i, column in enumerate(df.columns):
            sns.histplot(df[column], bins=30, kde=True, ax=axes[i])
            axes[i].set_title(f'Распределение {column}')
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)

    if show_boxplot:
        st.subheader("Box plot")
        fig_box, ax_box = plt.subplots(figsize=(15, 5))
        sns.boxplot(data=df, ax=ax_box)
        ax_box.set_title("Box plot")
        plt.xticks(rotation=45)
        st.pyplot(fig_box)

    if show_barplots:
        st.subheader("Столбчатые диаграммы для каждого признака")
        for col in df.columns:
            fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
            df[col].value_counts().plot(kind='bar', ax=ax_bar)
            ax_bar.set_title(f"Bar Plot для {col}")
            ax_bar.set_ylabel("Частота")
            plt.xticks(rotation=45)
            st.pyplot(fig_bar)

    if show_correlation:
        st.subheader("Матрица корреляций")
        correlation_matrix = df.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

    st.header('Моделирование')
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, X_train_original = preprocess_data(df)

    best_hyperparams = {
        "Логистическая регрессия": {'C': 0.1, 'class_weight': 'balanced'},
        "CatBoost": {'depth': 6, 'iterations': 100, 'learning_rate': 0.01},
        "XGBoost": {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'scale_pos_weight': 5},
        "Random Forest": {'class_weight': 'balanced', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
    }
    models = {
        "Логистическая регрессия": LogisticRegression(random_state=42, max_iter=1000, **best_hyperparams["Логистическая регрессия"]),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0, **best_hyperparams["CatBoost"]),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **best_hyperparams["XGBoost"]),
        "Random Forest": RandomForestClassifier(random_state=42, **best_hyperparams["Random Forest"])
    }

    if st.button('Обучить и оценить модель'):
        st.session_state.model_trained = True
        st.subheader(f"Оценка модели: {model_choice}")
        with st.spinner(f'Обучение модели {model_choice}...'):
            model = models[model_choice]
            model.fit(X_train_scaled, y_train)
            st.session_state.trained_model = model  # Сохраняем обученную модель
            evaluate_model(model, X_test_scaled, y_test, model_choice, feature_names, best_hyperparams[model_choice])
        st.success(f'Модель {model_choice} успешно обучена и оценена!')

    st.header('Предсказание дефицита тестостерона')
    input_data = pd.DataFrame({
        'Возраст': [input_age],
        'Наличие Диабета': [input_diabetes],
        'Триглицериды (мг/дл)': [input_triglycerides],
        'Наличие Гипертонии': [input_hypertension],
        'HDL_холестерин': [input_hdl],
        'Окружность_талии': [input_waist_circumference]
    })

    input_scaled = scaler.transform(input_data)

    if st.button('Предсказать'):
        if not st.session_state.model_trained or st.session_state.trained_model is None:
            st.warning("Пожалуйста, обучите модель, нажав кнопку 'Обучить и оценить модель' перед выполнением предсказания.")
        else:
            selected_model = st.session_state.trained_model
            with st.spinner('Выполнение предсказания...'):
                prediction_proba = predict_deficiency(selected_model, input_scaled)
            if prediction_proba is not None:
                st.subheader('Результат предсказания:')
                st.write(f'Вероятность дефицита тестостерона: **{prediction_proba:.4f}**')
                if prediction_proba >= 0.5:
                    st.warning("На основе введенных данных, модель предсказывает **высокую вероятность** дефицита тестостерона.")
                else:
                    st.success("На основе введенных данных, модель предсказывает **низкую вероятность** дефицита тестостерона.")

else:
    st.error("Не удалось загрузить данные. Проверьте URL и подключение к интернету. **Убедитесь, что URL-адрес raw-файла XLSX правильный и доступен.**")
    st.stop()
