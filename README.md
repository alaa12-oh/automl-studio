# AutoML Studio

An end-to-end Automated Machine Learning (AutoML) application that allows users to train and evaluate machine learning models using their own CSV datasets — without writing code.

## Overview

AutoML Studio simplifies the machine learning workflow by providing an interactive interface where users can upload a dataset, select a target column, choose a machine learning task, and compare multiple models.

The application supports both **Classification** and **Regression** tasks and selects the best-performing model based on evaluation metrics.

## Features

- **CSV Upload:** Upload your own dataset.
- **Data Exploration:** View dataset information, distributions, and correlation matrix.
- **Data Preprocessing:** Handle missing values and prepare numerical and categorical features.
- **Classification & Regression:** Support for two main machine learning task types.
- **Model Training:** Train and compare multiple machine learning algorithms.
- **Model Evaluation:** Evaluate models using appropriate performance metrics.
- **Champion Model Selection:** Automatically identify the best-performing model.
- **Model Export:** Save the selected model as a `.pkl` file.
- **Experiment Tracking:** Track experiments and model performance using MLflow.
- **Interactive Interface:** Use the application through a Streamlit web interface.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- MLflow
- Joblib

## Project Structure

```text
automl-studio/
│
├── app.py
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── task_detection.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── model_export.py
│   └── experiment_tracking.py
│
├── artifacts/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/alaa12-oh/automl-studio.git
```

2. Navigate to the project directory:

```bash
cd automl-studio
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the virtual environment:

**Windows:**

```bash
.venv\Scripts\activate
```

5. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run app.py
```

2. Open the application in your browser.
3. Upload a CSV dataset.
4. Select the target column.
5. Choose the task type: Classification or Regression.
6. Train and compare the available models.
7. Review the evaluation results and selected Champion Model.
8. Export the trained model.

## MLflow Tracking

AutoML Studio uses MLflow to track machine learning experiments, parameters, and evaluation metrics.

To open the MLflow interface, run:

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

## Model Export

The selected model is saved in `.pkl` format, allowing it to be loaded and used later for predictions on new data.

## Future Improvements

- Support for additional file formats.
- More machine learning algorithms.
- Advanced feature selection.
- Improved model evaluation and visualization.
- Enhanced user interface.

## Author

**Alaa Ibrahim**

GitHub: [alaa12-oh](https://github.com/alaa12-oh)

---

*Built as an end-to-end AutoML project using Python and machine learning tools.*
