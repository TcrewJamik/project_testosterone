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

    # Выбор признаков
    selected_features = st.multiselect(
        "Выберите признаки для обучения и предсказания:",
        feature_names,
        default=feature_names  # по умолчанию все
    )

    # Кнопка обучения
    train_button = st.button("🔥 Обучить модель")

    # Настройка входных значений и предсказание (отображается только после обучения)
    if 'clf' in st.session_state:
        st.subheader("Настройка входных значений")
        prediction_data = {}
        for feature in selected_features:
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
