import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.title("Предсказание дефицита тестостерона")
st.write("Веб-приложение для предсказания дефицита тестостерона на основе медицинских данных.")

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_excel('ptestost.xlsx')
    df = df.rename(columns={
        'Age': 'Возраст',
        'DM': 'Наличие Диабета',
        'TG': 'Триглицериды (мг/дл)',
        'HT': 'Наличие Гипертонии',
        'HDL': 'HDL_холестерин',
        'AC': 'Окружность_талии',
        'T': 'Дефицит Тестостерона'
    })
    target = df['Дефицит Тестостерона']
    X = df.drop(columns=['Дефицит Тестостерона'])
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = X.columns
    return df, X_train_scaled, X_test_scaled, y_train, y_test, feature_names

df, X_train_scaled, X_test_scaled, y_train, y_test, feature_names = load_and_preprocess_data()

st.sidebar.header("Настройки визуализации и модели")
show_data = st.sidebar.checkbox("Показать данные")
show_eda = st.sidebar.checkbox("Показать разведочный анализ данных (EDA)")
model_name = st.sidebar.selectbox("Выберите модель", ["Логистическая регрессия", "CatBoost", "XGBoost", "Random Forest"])

if show_data:
    st.subheader("Исходные данные")
    st.dataframe(df)

    st.subheader("Описание данных")
    st.write(df.describe())

    st.subheader("Пропущенные значения")
    st.write(df.isna().sum())

if show_eda:
    st.subheader("Разведочный анализ данных (EDA)")

    st.subheader("Гистограммы распределения признаков")
    num_cols = len(df.columns)
    rows = (num_cols + 2) // 3
    cols = min(num_cols, 3)
    fig_hist, axes_hist = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes_hist = axes_hist.flatten() if rows > 1 else axes_hist
    for i, column in enumerate(df.columns):
        sns.histplot(df[column], bins=30, kde=True, ax=axes_hist[i])
        axes_hist[i].set_title(f'Распределение {column}')
    for j in range(i + 1, len(axes_hist)):
        fig_hist.delaxes(axes_hist[j])
    plt.tight_layout()
    st.pyplot(fig_hist)

    st.subheader("Ящик с усами (Box plot)")
    fig_box, ax_box = plt.subplots(figsize=(15, 5))
    sns.boxplot(data=df, ax=ax_box)
    ax_box.set_title("Box plot")
    st.pyplot(fig_box)

    st.subheader("Столбчатые диаграммы")
    for col in df.columns:
        fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
        df[col].value_counts().plot(kind='bar', ax=ax_bar)
        ax_bar.set_title(f"Bar Plot для {col}")
        ax_bar.set_ylabel("Частота")
        ax_bar.tick_params(axis='x', rotation=45)
        st.pyplot(fig_bar)

    st.subheader("Тепловая карта корреляций")
    correlation_matrix = df.corr()
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_heatmap)
    ax_heatmap.set_title("Тепловая карта корреляций")
    st.pyplot(fig_heatmap)

@st.cache_resource
def train_model(model_name, X_train_scaled, y_train):
    if model_name == "Логистическая регрессия":
        lr_param_grid = {
            'C': [0.1, 1], # Simplified C values
            'class_weight': ['balanced'] # Kept only 'balanced'
        }
        lr_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000),
                               lr_param_grid, cv=5, scoring='f1', n_jobs=-1)
        lr_grid.fit(X_train_scaled, y_train)
        best_model = lr_grid.best_estimator_
    elif model_name == "CatBoost":
        cb_param_grid = {
            'depth': [4, 6], # Simplified depth values
            'iterations': [100], # Reduced iterations
            'learning_rate': [0.1] # Single learning rate
        }
        cb_grid = GridSearchCV(CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'),
                               cb_param_grid, cv=5, scoring='f1', n_jobs=-1)
        cb_grid.fit(X_train_scaled, y_train)
        best_model = cb_grid.best_estimator_
    elif model_name == "XGBoost":
        xgb_param_grid = {
            'max_depth': [3, 6], # Simplified depth values
            'n_estimators': [100], # Reduced estimators
            'learning_rate': [0.1], # Single learning rate
            'scale_pos_weight': [1] # Single scale_pos_weight
        }
        xgb_grid = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                                xgb_param_grid, cv=5, scoring='f1', n_jobs=-1)
        xgb_grid.fit(X_train_scaled, y_train)
        best_model = xgb_grid.best_estimator_
    elif model_name == "Random Forest":
        rf_param_grid = {
            'n_estimators': [100], # Reduced estimators
            'max_depth': [5, 10], # Simplified depth values
            'min_samples_split': [2], # Single value
            'min_samples_leaf': [1], # Single value
            'class_weight': ['balanced'] # Kept only 'balanced'
        }
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                               rf_param_grid, cv=5, scoring='f1', n_jobs=-1)
        rf_grid.fit(X_train_scaled, y_train)
        best_model = rf_grid.best_estimator_
    return best_model

def evaluate_model_streamlit(model, X_test_scaled, y_test, model_name, feature_names):
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    st.subheader(f"Оценка модели: {model_name}")

    st.write(f"**Метрики с порогом 0.5:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
    col5.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_proba):.4f}")

    # ROC Curve
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

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'], ax=ax_cm)
    ax_cm.set_ylabel('True Label')
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_title(f'Матрица ошибок - {model_name}')
    st.pyplot(fig_cm)

    # Feature Importance
    if model_name == "Логистическая регрессия":
        importance = np.abs(model.coef_[0])
    else:
        importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    st.write(f"\n**{model_name} - Важность признаков:**")
    st.dataframe(feature_importance)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', ax=ax_fi)
    ax_fi.set_title(f'Важность признаков - {model_name}')
    ax_fi.set_xlabel('Важность')
    ax_fi.set_ylabel('Признак')
    plt.tight_layout()
    st.pyplot(fig_fi)


if st.sidebar.button("Обучить и оценить модель"):
    with st.spinner(f'Обучение модели {model_name}...'):
        best_model = train_model(model_name, X_train_scaled, y_train)
    st.success(f'Модель {model_name} обучена!')
    evaluate_model_streamlit(best_model, X_test_scaled, y_test, model_name, feature_names)
