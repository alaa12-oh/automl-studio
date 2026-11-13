# 🚀 AutoML Studio

### 🤖 End-to-End Automated Machine Learning Platform

An interactive AutoML application designed to automate the process of building, training, evaluating, and comparing machine learning models using CSV datasets, without requiring users to write code.

---

## 🛠️ Built With

Python · Pandas · NumPy · Scikit-learn · Streamlit · MLflow · Matplotlib · Seaborn · Joblib

---

## 🌟 Overview

AutoML Studio simplifies the machine learning workflow by automating its main stages through an easy-to-use interactive interface built with Streamlit.

Users can upload their own CSV datasets, select a target column, choose the machine learning task type, and run automated data preprocessing, model training, evaluation, and comparison.

The platform supports two types of machine learning tasks:

- **Classification**
- **Regression**

At the end of the process, the system automatically selects the best-performing model, known as the **Champion Model**.

## ✨ Key Features

### 📂 CSV Data Upload
- Upload custom datasets in CSV format.

### 🔍 Exploratory Data Analysis (EDA)
- Display dataset previews and basic information.
- Analyze missing values.
- Visualize feature distributions.
- Generate correlation matrices.

### 🧹 Data Cleaning & Preprocessing
- Handle missing values.
- Remove duplicate records.
- Handle numerical outliers using the IQR method.
- Apply numerical feature scaling.
- Encode categorical features using One-Hot Encoding.

### ⚙️ Feature Engineering
- Generate polynomial interaction features.

### 🎯 Task Selection
- Select between Classification and Regression tasks.
- Choose the target column from the uploaded dataset.

### 🤖 Model Training
- Train and compare five machine learning algorithms for each task type.

### 🔧 Hyperparameter Tuning
- Optimize model hyperparameters using RandomizedSearchCV.

### 📊 Model Evaluation
- Evaluate models using appropriate performance metrics.
- Visualize evaluation results to compare model performance.

### 🏆 Champion Model Selection
- Automatically select the best-performing model based on cross-validation results.

### 💾 Model Export
- Save the selected model along with its preprocessing steps in a `.pkl` file.

### 📈 MLflow Experiment Tracking
- Track model parameters.
- Log model evaluation metrics.

### ⏱️ Process Time Analysis
- Measure the execution time of different stages of the AutoML pipeline.

### 🖥️ Interactive User Interface
- Explore and run the entire machine learning workflow through a Streamlit interface.

## 🤖 Machine Learning Algorithms

### Classification
1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)
4. Gradient Boosting Classifier
5. K-Nearest Neighbors (KNN)

### Regression
1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regression (SVR)

## 📊 Evaluation Metrics

### Classification
- **Accuracy:** Measures the proportion of correct predictions.
- **Weighted F1 Score:** Evaluates the balance between precision and recall while accounting for class distribution.

### Regression
- **Root Mean Squared Error (RMSE):** Measures the square root of the average squared prediction errors.
- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **R² Score:** Measures how well the model explains the variance in the target variable.

The Champion Model is selected based on:

- The highest Cross-Validation F1 Score for Classification.
- The lowest Cross-Validation RMSE for Regression.

## 🧰 Technologies & Libraries

| Library | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Data preprocessing, model training, tuning, and evaluation |
| Streamlit | Interactive application interface |
| MLflow | Machine learning experiment tracking |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| Joblib | Model saving and loading |

## 📁 Project Structure

```text
automl-studio/
│
├── app.py
│
├── src/
│   ├── __init__.py
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
│   └── best_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/alaa12-oh/automl-studio.git
```

### 2. Navigate to the Project Directory

```bash
cd automl-studio
```

### 3. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
```

**Linux:**

```bash
python3 -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

**Linux:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

Then open the local URL displayed in your terminal.

## 📝 How to Use

1. Upload a dataset in CSV format.
2. Select the target column.
3. Choose the machine learning task type.
4. Start the model training process.
5. Explore and compare model evaluation results.
6. View the selected Champion Model.
7. Find the exported model in the `artifacts` directory.

## 📈 MLflow Experiment Tracking

AutoML Studio integrates with MLflow to track machine learning experiments and record model results.

MLflow logs the following information for each trained model:

- Model name and task type.
- Best hyperparameters.
- Evaluation metrics.

To launch the MLflow UI, run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://localhost:5000
```

## 🏆 Model Export

The selected Champion Model is saved at:

```text
artifacts/best_model.pkl
```

The exported file contains the complete machine learning pipeline, including preprocessing steps and the trained model.

This allows users to load the saved model and make predictions on new data without running the entire training process again.

## 🔮 Future Improvements

- Support additional data file formats.
- Introduce more machine learning algorithms.
- Improve training and experiment tracking performance.
- Enhance the user interface with additional analysis and visualization options.

## 👩‍💻 Project Credits

Developed by **Alaa Ibrahim**

GitHub: [https://github.com/alaa12-oh](https://github.com/alaa12-oh)

---

⭐ AutoML Studio was developed using Python and machine learning technologies to simplify and automate the model development workflow.