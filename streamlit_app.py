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
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.title('Предсказание дефицита тестостерона')

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
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
    else:
        return None

uploaded_file = st.file_uploader("Загрузите Excel файл", type=["xlsx"])
df = load_data(uploaded_file)

if df is not None:
    st.header('Обзор данных')
    if st.checkbox('Показать данные'):
        st.dataframe(df)

    if st.checkbox('Информация о данных'):
        st.write(df.info())

    if st.checkbox('Описательная статистика'):
        st.write(df.describe())

    if st.checkbox('Проверка на пропуски'):
        st.write(df.isna().sum())

    st.header('Визуализация данных')
    if st.checkbox('Гистограммы распределения признаков'):
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

    if st.checkbox('Ящик с усами (Box plot)'):
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.boxplot(data=df, ax=ax)
        ax.set_title("Box plot")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if st.checkbox('Столбчатые диаграммы для каждого признака'):
        for col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Bar Plot для {col}")
            ax.set_ylabel("Частота")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    if st.checkbox('Матрица корреляций'):
        correlation_matrix = df.corr()
        st.write("\nМатрица корреляций:")
        st.dataframe(correlation_matrix)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

    st.header('Моделирование')
    target = df['Дефицит Тестостерона']
    X = df.drop(columns=['Дефицит Тестостерона'])
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = X.columns

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

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC-кривая - {model_name}')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

        conf_matrix = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'], ax=ax_cm)
        ax_cm.set_ylabel('True Label')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_title(f'Матрица ошибок - {model_name}')
        st.pyplot(fig_cm)

        if model_name == "Логистическая регрессия":
            importance = np.abs(model.coef_[0])
        else:
            importance = model.feature_importances_

        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        st.subheader(f"{model_name} - Важность признаков:")
        st.dataframe(feature_importance)

        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', ax=ax_fi)
        ax_fi.set_title(f'Важность признаков - {model_name}')
        ax_fi.set_xlabel('Важность')
        ax_fi.set_ylabel('Признак')
        plt.tight_layout()
        st.pyplot(fig_fi)

    model_choice = st.selectbox("Выберите модель", ["Логистическая регрессия", "CatBoost", "XGBoost", "Random Forest"])

    if st.button('Обучить и оценить модель'):
        if model_choice == "Логистическая регрессия":
            best_lr = LogisticRegression(random_state=42, max_iter=1000, C=0.1, class_weight='balanced')
            best_lr.fit(X_train_scaled, y_train)
            evaluate_model(best_lr, X_test_scaled, y_test, "Логистическая регрессия", feature_names)

        elif model_choice == "CatBoost":
            best_cb = CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced', depth=6, iterations=100, learning_rate=0.01)
            best_cb.fit(X_train_scaled, y_train)
            evaluate_model(best_cb, X_test_scaled, y_test, "CatBoost", feature_names)

        elif model_choice == "XGBoost":
            best_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', learning_rate=0.1, max_depth=3, n_estimators=100, scale_pos_weight=5)
            best_xgb.fit(X_train_scaled, y_train)
            evaluate_model(best_xgb, X_test_scaled, y_test, "XGBoost", feature_names)

        elif model_choice == "Random Forest":
            best_rf = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=5, min_samples_leaf=2, min_samples_split=5, n_estimators=100)
            best_rf.fit(X_train_scaled, y_train)
            evaluate_model(best_rf, X_test_scaled, y_test, "Random Forest", feature_names)

else:
    st.info("Пожалуйста, загрузите Excel файл для начала работы.")
