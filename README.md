# 🚀 AutoML Studio

### 🤖 End-to-End Automated Machine Learning Platform

An interactive AutoML application that allows users to build, train, evaluate, and compare Machine Learning models using their own CSV datasets — **without writing code!**

---

## 🛠️ Built With

<p align="center">

**Python** · **Pandas** · **NumPy** · **Scikit-learn** · **Streamlit** · **MLflow** · **Matplotlib** · **Seaborn** · **Joblib**

</p>

---

## 🌟 Overview

AutoML Studio simplifies the Machine Learning workflow by automating the main steps of model development through an easy-to-use Streamlit interface.

Users can upload their datasets, select a target column, identify the machine learning task, and automatically train and compare multiple algorithms.

The platform supports both **Classification** and **Regression**, then selects the best-performing model as the **Champion Model**.

## ✨ Key Features

📂 **CSV Data Upload**
- Upload your own datasets in CSV format.

🔍 **Exploratory Data Analysis (EDA)**
- Dataset preview and information.
- Missing values analysis.
- Feature distributions.
- Correlation matrix visualization.

🧹 **Data Cleaning & Preprocessing**
- Handle missing values.
- Remove duplicate records.
- Handle numerical outliers using the IQR method.
- Numerical scaling.
- Categorical encoding using One-Hot Encoding.

⚙️ **Feature Engineering**
- Generate polynomial interaction features when applicable.

🎯 **Classification & Regression**
- Automatically identify the machine learning task.

🤖 **Model Training**
- Train and compare five machine learning algorithms for each task.

🔧 **Hyperparameter Tuning**
- Optimize model parameters using RandomizedSearchCV.

📊 **Model Evaluation**
- Evaluate models using appropriate performance metrics.
- Visualize model performance.

🏆 **Champion Model Selection**
- Automatically select the best-performing model based on cross-validation results.

💾 **Model Export**
- Save the selected model and preprocessing pipeline as a `.pkl` file.

📈 **MLflow Experiment Tracking**
- Track model parameters and evaluation metrics.

⏱️ **Process Time Analysis**
- Measure the time consumed by different stages of the AutoML pipeline.

🖥️ **Interactive User Interface**
- Explore and use the entire pipeline through Streamlit.

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

**Classification:**
- Accuracy
- Weighted F1 Score

**Regression:**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

🏅 The Champion Model is selected using the highest cross-validation F1 Score for Classification or the lowest cross-validation RMSE for Regression.

## 🧰 Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Preprocessing, training, tuning, and evaluation |
| Streamlit | Interactive web application |
| MLflow | Experiment tracking |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Model serialization and loading |

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
├── tests/
│   ├── test_data_loader.py
│   └── test_preprocessing.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/alaa12-oh/automl-studio.git
```

**2. Navigate to the project directory**

```bash
cd automl-studio
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate the virtual environment**

Windows:

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

## 🚀 Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the application in your browser.

### 📝 How to Use

1. Upload a CSV dataset.
2. Select the target column.
3. Review the detected task type.
4. Start the model training process.
5. Explore model comparisons and evaluation results.
6. View the selected Champion Model.
7. Download the trained model.

## 📈 MLflow Experiment Tracking

AutoML Studio integrates MLflow to track machine learning experiments.

For each trained model, MLflow records:

- Model name and task type.
- Best hyperparameters.
- Evaluation metrics.

To launch the MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the following address:

```text
http://localhost:5000
```

## 🏆 Model Export

The selected Champion Model is saved as:

```text
artifacts/best_model.pkl
```

The exported file contains the complete machine learning pipeline, including preprocessing steps and the trained model.

This allows users to load the saved model and make predictions on new data without repeating the entire training process.

## 🔮 Future Improvements

- Support for additional data formats.
- More machine learning algorithms.
- Advanced feature selection techniques.
- Additional visualization options.
- Further performance optimization.

## 👩‍💻 Author

**Alaa Ibrahim**

🔗 GitHub: [alaa12-oh](https://github.com/alaa12-oh)

---

⭐ *Built with Python, Machine Learning, and a passion for automation.*
