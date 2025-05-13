# 🤖 Automated ML Pipeline

This project provides a complete **Automated Machine Learning Pipeline** built with Python and **Streamlit**. It allows users to upload CSV files, perform **Exploratory Data Analysis (EDA)**, handle **data preprocessing**, and train **ML models** (classification, regression, clustering) — all from a user-friendly interface.

## 🚀 Overview of the ML Pipeline:

The pipeline automates the entire process of training and evaluating machine learning models. Here's an overview of the key steps involved:

### 1. **Data Ingestion**:
   - Users upload a CSV file that contains their dataset. The app reads the file and displays it. Users are prompted to select the target column (the variable they want to predict).
   - The data is then split into training and test sets for model training and evaluation.

### 2. **Exploratory Data Analysis (EDA)**:
   - The pipeline uses `ydata-profiling` to automatically generate an in-depth EDA report. The report includes:
     - Data type analysis, distribution, and missing values
     - Correlations between variables
     - Visualizations like histograms, box plots, and scatter plots
     - Anomalies and outliers
   - The EDA report is downloadable for further analysis and insight into the dataset's structure.

### 3. **Data Transformation**:
   - The app preprocesses the data, handling missing values, encoding categorical variables, and normalizing or scaling numerical features.
   - It splits the dataset into training and testing subsets to ensure the model is trained on clean, processed data.

### 4. **Model Training**:
   - The pipeline supports three types of machine learning problems:
     - **Classification** (e.g., spam detection, disease classification)
     - **Regression** (e.g., predicting house prices, stock market forecasting)
     - **Clustering** (e.g., customer segmentation)
   - Multiple machine learning models are trained, such as Logistic Regression, Random Forest, and KMeans, depending on the selected problem type.
   - The models are evaluated based on their performance metrics (accuracy for classification, RMSE for regression, and silhouette score for clustering).

### 5. **Model Evaluation**:
   - The best-performing model is selected and displayed, showing key evaluation metrics like:
     - **Train and Test Scores**: Performance of the model on both the training and test datasets
     - **Confusion Matrix** (for classification models)
     - **F1-Score, Accuracy, RMSE** (for regression models)
     - **Silhouette Score** (for clustering models)
   - Users can download the best model's performance report for further insights.

### 6. **Model Deployment** (Future Scope):
   - Although not yet implemented in this version, future improvements could involve integrating model deployment, allowing users to deploy the trained models for real-time predictions.
---

## 📌 Features

- ✅ Upload CSV and select the type of ML problem
- 📊 Automated EDA using `ydata-profiling`
- ⚙️ Data preprocessing and transformation
- 🧠 Model training with performance evaluation
- 📈 Interactive display of model comparison
- 📥 Downloadable EDA report
- 🌟 Streamlit-based user interface



![image](https://github.com/user-attachments/assets/5d271280-0729-4194-943d-b595aba2043a)
