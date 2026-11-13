
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data_loader import load_data
from src.eda import (
    get_data_overview,
    plot_distributions,
    plot_correlation_matrix
)
from src.preprocessing import clean_data, build_preprocessor
from src.task_detection import detect_task
from src.model_training import train_models
from src.evaluation import evaluate_models, get_best_model
from src.model_export import save_model
from src.experiment_tracking import log_experiment


st.set_page_config(
    page_title="AutoML System",
    page_icon="🤖",
    layout="wide"
)

st.title("AutoML System")
st.write("Upload your CSV dataset and train machine learning models.")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = load_data(uploaded_file)
        st.success("Dataset uploaded successfully")

        st.subheader("1. Select Target Column")

        target_column = st.selectbox(
            "Choose the column you want to predict",
            data.columns,
            key="target_column"
        )

        if st.session_state.get("last_target") != target_column:
            st.session_state.pop("suggested_task", None)
            st.session_state["last_target"] = target_column

        st.subheader("2. Detect Problem Type")

        if st.button("Detect Problem Type"):
            try:
                data_for_detection = data.dropna(
                    subset=[target_column]
                )

                suggested_task = detect_task(
                    data_for_detection,
                    target_column
                )

                st.session_state["suggested_task"] = suggested_task

                st.success(
                    f"Suggested problem type: {suggested_task.title()}"
                )

            except ValueError as error:
                st.session_state.pop("suggested_task", None)
                st.error(str(error))

        task_type = None

        if "suggested_task" in st.session_state:
            st.subheader("3. Confirm Problem Type")

            task_type = st.selectbox(
                "Confirm or change the problem type",
                ["classification", "regression"],
                index=(
                    0 if st.session_state["suggested_task"] == "classification"
                    else 1
                )
            )

        st.divider()
        st.subheader("4. Dataset Preview")
        st.dataframe(data.head())

        overview = get_data_overview(data)

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", overview["shape"][0])
        col2.metric("Columns", overview["shape"][1])
        col3.metric("Duplicates", int(overview["duplicates"]))

        st.subheader("Data Information")

        st.write("Missing Values")
        st.write(overview["missing_values"])

        st.write("Data Types")
        st.write(overview["data_types"])

        st.subheader("5. Data Distribution")

        distribution_fig = plot_distributions(data)

        if distribution_fig is not None:
            st.pyplot(distribution_fig)
            plt.close(distribution_fig)
        else:
            st.info("No numeric columns available for distribution plots.")

        st.subheader("6. Correlation Matrix")

        correlation_fig = plot_correlation_matrix(data)

        if correlation_fig is not None:
            st.pyplot(correlation_fig)
            plt.close(correlation_fig)
        else:
            st.info("Not enough numeric columns for a correlation matrix.")

        st.subheader("7. Model Training")

        if task_type is not None:
            if st.button("Train Models"):
                try:
                    stage_times = {}
                    mlflow_times = []
                    total_start = time.perf_counter()

                    # Data Preparation
                    stage_start = time.perf_counter()

                    training_data = clean_data(data)

                    missing_targets = training_data[
                        target_column
                    ].isnull().sum()

                    if missing_targets > 0:
                        training_data = training_data.dropna(
                            subset=[target_column]
                        )

                        st.warning(
                            f"{missing_targets} rows with missing target values were removed."
                        )

                    if training_data.empty:
                        raise ValueError(
                            "No data available after cleaning."
                        )

                    X = training_data.drop(columns=[target_column])
                    y = training_data[target_column]

                    if X.empty:
                        raise ValueError(
                            "No feature columns available."
                        )

                    stratify = None

                    if task_type == "classification":
                        class_counts = y.value_counts()

                        if class_counts.min() < 2:
                            raise ValueError(
                                "Each class needs at least two samples."
                            )

                        test_count = max(
                            1,
                            int(len(training_data) * 0.2)
                        )

                        train_count = len(training_data) - test_count

                        if (
                            test_count >= y.nunique()
                            and train_count >= y.nunique()
                        ):
                            stratify = y

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=42,
                        stratify=stratify
                    )

                    preprocessor = build_preprocessor(X_train)

                    numeric_features = X_train.select_dtypes(
                        include=["int64", "float64"]
                    ).columns

                    numeric_count = len(numeric_features)

                    feature_engineering_enabled = (
                        2 <= numeric_count <= 10
                    )

                    if feature_engineering_enabled:
                        interaction_count = (
                            numeric_count * (numeric_count - 1) // 2
                        )
                    else:
                        interaction_count = 0

                    stage_times["Data Preparation"] = (
                        time.perf_counter() - stage_start
                    )

                    # Model Training
                    stage_start = time.perf_counter()

                    with st.spinner("Training models..."):
                        trained_models, failed_models = train_models(
                            X_train,
                            y_train,
                            preprocessor,
                            task_type
                        )

                    stage_times["Model Training"] = (
                        time.perf_counter() - stage_start
                    )

                    # Model Evaluation
                    stage_start = time.perf_counter()

                    results = evaluate_models(
                        trained_models,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        task_type
                    )

                    best_model_name, best_model = get_best_model(
                        trained_models,
                        results,
                        task_type
                    )

                    stage_times["Model Evaluation"] = (
                        time.perf_counter() - stage_start
                    )

                    # MLflow Logging
                    stage_start = time.perf_counter()

                    with st.spinner("Logging experiments in MLflow..."):
                        for model_name, metrics in results.items():
                            model_start = time.perf_counter()

                            try:
                                log_experiment(
                                    model_name,
                                    trained_models[model_name],
                                    metrics,
                                    task_type
                                )
                            except Exception as error:
                                st.warning(
                                    f"Could not log {model_name} in MLflow: {error}"
                                )

                            mlflow_times.append({
                                "Model": model_name,
                                "Logging Time (seconds)": round(
                                    time.perf_counter() - model_start, 2
                                )
                            })

                    stage_times["MLflow Logging"] = (
                        time.perf_counter() - stage_start
                    )

                    # Feature Engineering
                    st.subheader("8. Feature Engineering")

                    st.write(
                        "The system creates interaction features "
                        "between numeric columns using PolynomialFeatures."
                    )

                    feature_col1, feature_col2, feature_col3 = st.columns(3)

                    feature_col1.metric(
                        "Original Numeric Features",
                        numeric_count
                    )

                    feature_col2.metric(
                        "Interaction Features Created",
                        interaction_count
                    )

                    if feature_engineering_enabled:
                        feature_status = "Enabled"
                    else:
                        feature_status = "Not Applied"

                    feature_col2.metric(
                        "Feature Engineering",
                        feature_status
                    )

                    stage_start = time.perf_counter()

                    transformed_sample = (
                        best_model.named_steps["preprocessor"].transform(
                            X_train.iloc[:5]
                        )
                    )

                    total_features = transformed_sample.shape[1]

                    stage_times["Feature Transformation"] = (
                        time.perf_counter() - stage_start
                    )

                    feature_col3.metric(
                        "Total Processed Features",
                        total_features
                    )

                    if feature_engineering_enabled:
                        st.success(
                            f"Feature Engineering created "
                            f"{interaction_count} interaction features."
                        )
                    else:
                        st.info(
                            "Feature Engineering was not applied. "
                            "It requires between 2 and 10 numeric features."
                        )

                    # Hyperparameter Tuning Results
                    st.subheader("9. Hyperparameter Tuning Results")

                    st.write(
                        "RandomizedSearchCV searches for the best "
                        "hyperparameters for each model."
                    )

                    tuning_results = []

                    for model_name, model in trained_models.items():

                        best_params = getattr(
                            model,
                            "best_params_",
                            {}
                        )

                        cv_score = getattr(
                            model,
                            "cv_best_score_",
                            None
                        )

                        if cv_score is not None:
                            tuning_results.append({
                                "Model": model_name,
                                "Best Parameters": ", ".join(
                                    f"{key}: {value}"
                                    for key, value in best_params.items()
                                ),
                                "CV Score": round(float(cv_score), 4)
                            })

                    st.dataframe(
                        pd.DataFrame(tuning_results),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Model Comparison
                    st.subheader("10. Model Comparison")

                    results_df = pd.DataFrame(results).T

                    if task_type == "classification":
                        results_df = results_df.sort_values(
                            by="CV F1 Score",
                            ascending=False
                        )
                    else:
                        results_df = results_df.sort_values(
                            by="CV RMSE",
                            ascending=True
                        )

                    st.dataframe(results_df)

                    # Champion Model
                    st.subheader("11. Champion Model")
                    st.success(f"Best Model: {best_model_name}")

                    # Prediction and Visualization
                    stage_start = time.perf_counter()

                    predictions = best_model.predict(X_test)

                    if task_type == "classification":
                        cm = confusion_matrix(y_test, predictions)
                        fig, ax = plt.subplots()

                        display = ConfusionMatrixDisplay(
                            confusion_matrix=cm
                        )

                        display.plot(ax=ax)
                        st.pyplot(fig)
                        plt.close(fig)

                    else:
                        fig, ax = plt.subplots()

                        ax.scatter(y_test, predictions)
                        ax.set_xlabel("Actual Values")
                        ax.set_ylabel("Predicted Values")
                        ax.set_title("Actual vs Predicted")

                        st.pyplot(fig)
                        plt.close(fig)

                    stage_times["Prediction and Plotting"] = (
                        time.perf_counter() - stage_start
                    )

                    # Model Saving
                    stage_start = time.perf_counter()

                    model_path = save_model(best_model)

                    stage_times["Model Saving"] = (
                        time.perf_counter() - stage_start
                    )

                    st.success("Champion model saved successfully.")

                    with open(model_path, "rb") as model_file:
                        st.download_button(
                            label="Download Best Model",
                            data=model_file,
                            file_name="best_model.pkl",
                            mime="application/octet-stream"
                        )

                    if failed_models:
                        st.subheader("Models That Could Not Be Trained")

                        for name, error in failed_models.items():
                            st.warning(f"{name}: {error}")

                    # Time Results
                    total_time = time.perf_counter() - total_start

                    st.divider()
                    st.subheader("12. Process Time Analysis")

                    timing_df = pd.DataFrame(
                        list(stage_times.items()),
                        columns=["Process", "Time (seconds)"]
                    )

                    timing_df["Time (seconds)"] = timing_df[
                        "Time (seconds)"
                    ].round(2)

                    st.dataframe(
                        timing_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.metric(
                        "Total Processing Time",
                        f"{total_time:.2f} seconds"
                    )

                    st.subheader("MLflow Logging Time")

                    st.dataframe(
                        pd.DataFrame(mlflow_times),
                        use_container_width=True,
                        hide_index=True
                    )

                except Exception as error:
                    st.error(f"An error occurred: {error}")

        else:
            st.info(
                "Please detect the problem type before training models."
            )

    except Exception as error:
        st.error(f"Could not load the dataset: {error}")

