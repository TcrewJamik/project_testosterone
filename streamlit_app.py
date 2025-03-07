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

# --- Functions ---
@st.cache_data
def load_data():
    excel_url = "https://raw.githubusercontent.com/your_username/your_repo_name/main/ptestost.xlsx" # Replace with your actual raw GitHub URL
    df = pd.read_excel(excel_url)
    df = df.rename(columns={
        'Age': 'Возраст',
        'DM': 'Наличие Диабета',
        'TG': 'Триглицериды (мг/дл)',
        'HT': 'Наличие Гипертонии',
        'HDL': 'HDL_холестерин',
        'AC': 'Окружность_талии',
        'T': 'Дефицит Тестостерона'
    })
    return df

def preprocess_data(df):
    target = df['Дефицит Тестостерона']
    X = df.drop(columns=['Дефицит Тестостерона'])
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = X.columns
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def evaluate_model(model, X_test, y_test, model_name, feature_names):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    st.subheader(f"{model_name} - Метрики с порогом 0.5:")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
    col5.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_proba):.4f}")

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


# --- Sidebar for Model Choice ---
st.sidebar.header("Настройки модели")
model_choice = st.sidebar.selectbox("Выберите модель", ["Логистическая регрессия", "CatBoost", "XGBoost", "Random Forest"])


# --- Main App ---
df = load_data()

if df is not None:
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
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = preprocess_data(df)

    if st.button('Обучить и оценить модель'):
        st.subheader(f"Оценка модели: {model_choice}")
        with st.spinner(f'Обучение модели {model_choice}...'):
            if model_choice == "Логистическая регрессия":
                model = LogisticRegression(random_state=42, max_iter=1000, C=0.1, class_weight='balanced')
            elif model_choice == "CatBoost":
                model = CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced', depth=6, iterations=100, learning_rate=0.01)
            elif model_choice == "XGBoost":
                model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', learning_rate=0.1, max_depth=3, n_estimators=100, scale_pos_weight=5)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=5, min_samples_leaf=2, min_samples_split=5, n_estimators=100)
            else:
                st.error("Модель не выбрана.")
                st.stop()

            model.fit(X_train_scaled, y_train)
            evaluate_model(model, X_test_scaled, y_test, model_choice, feature_names)
        st.success(f'Модель {model_choice} успешно обучена и оценена!')


else:
    st.error("Не удалось загрузить данные. Проверьте URL и подключение к интернету.")
    st.stop()
