
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

                    # Separate features and target
                    X = training_data.drop(
                        columns=[target_column]
                    )

                    y = training_data[target_column]

                    # Debug information
                    st.write("Selected target:", target_column)
                    st.write("X columns:", X.columns.tolist())
                    st.write("Training data columns:", training_data.columns.tolist())

                    if target_column in X.columns:
                        raise ValueError(
                            "Target column is still inside the features."
                        )

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

                    st.write("X_train columns:", X_train.columns.tolist())
                    st.write("y_train name:", y_train.name)

                    with st.spinner("Training models..."):

                        trained_models, failed_models = train_models(
                            X_train,
                            y_train,
                            preprocessor,
                            task_type
                        )

                    # Evaluate models
                    results = evaluate_models(
                        trained_models,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        task_type
                    )

                    # Log experiments in MLflow
                    for model_name, metrics in results.items():

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

                    # Model comparison
                    st.subheader("8. Model Comparison")

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

                    # Select champion model
                    best_model_name, best_model = get_best_model(
                        trained_models,
                        results,
                        task_type
                    )

                    st.subheader("9. Champion Model")

                    st.success(
                        f"Best Model: {best_model_name}"
                    )

                    # Display evaluation plot
                    predictions = best_model.predict(X_test)

                    if task_type == "classification":

                        cm = confusion_matrix(
                            y_test,
                            predictions
                        )

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

                    # Save champion model
                    model_path = save_model(best_model)

                    st.success(
                        "Champion model saved successfully."
                    )

                    with open(model_path, "rb") as model_file:

                        st.download_button(
                            label="Download Best Model",
                            data=model_file,
                            file_name="best_model.pkl",
                            mime="application/octet-stream"
                        )

                    # Show failed models
                    if failed_models:

                        st.subheader(
                            "Models That Could Not Be Trained"
                        )

                        for name, error in failed_models.items():
                            st.warning(
                                f"{name}: {error}"
                            )

                except Exception as error:

                    st.error(
                        f"An error occurred: {error}"
                    )

        else:
            st.info(
                "Please detect the problem type before training models."
            )

    except Exception as error:

        st.error(
            f"Could not load the dataset: {error}"
        )
