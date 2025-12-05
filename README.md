
THE CONTENT OF THIS README FILE WAS GENERATED USING GOOGLE GEMINI 3 PRO.

Wine Quality and Wine Type Prediction
Machine Learning in Practice – Homework 2.
	1.	Task Description

The goal of this project was to build a complete machine-learning pipeline using the Wine Quality dataset from the UCI Machine Learning Repository. The assignment required exploratory data analysis, data cleaning and preprocessing, feature engineering, model training, evaluation, and visualization of results.

In addition to the required regression task, I extended the project with a classification problem and implemented an interactive Streamlit application.

⸻

	2.	Dataset

Two datasets were used:
	•	winequality-white.csv
	•	winequality-red.csv

These datasets contain physicochemical properties of wines, provided separately for white and red wines.

Preprocessing steps included:
	•	Merging the two datasets
	•	Creating a categorical variable called “type” (white or red)
	•	Checking for missing values (none were found)
	•	Standardizing numerical features where appropriate

⸻

	3.	Problem Formulation

3.1 Regression Task

The objective of the regression task was to predict wine quality based on chemical characteristics and wine type.

Target variable:
	•	quality

Input features:
	•	Physicochemical attributes
	•	Wine type (red or white)

3.2 Classification Task (Extension)

The objective of the classification task was to determine whether a wine is red or white using only chemical properties.

Target variable:
	•	type

Important modeling decision:
	•	The feature “quality” was explicitly excluded from the classification models to avoid target leakage.

⸻

	4.	Exploratory Data Analysis (EDA)

The following analyses and visualizations were performed:
	•	Boxplots showing the relationship between alcohol content and wine quality
	•	Distribution of wine quality scores
	•	Correlation heatmap of numerical features
	•	Distribution of wine types

These analyses supported model selection and highlighted relationships between variables.

⸻

	5.	Models and Methods

5.1 Regression Models

Baseline models:
	•	Linear Regression
	•	Decision Tree Regressor

Advanced model:
	•	Random Forest Regressor with hyperparameter tuning using GridSearchCV

Evaluation metrics:
	•	Mean Absolute Error (MAE)
	•	Root Mean Squared Error (RMSE)
	•	R-squared score

Diagnostics:
	•	Predicted versus actual values plot
	•	Residual plot to assess model assumptions

⸻

5.2 Classification Models

Baseline model:
	•	Logistic Regression

Advanced model:
	•	Random Forest Classifier

Evaluation metrics:
	•	Accuracy
	•	Confusion matrix
	•	Precision, recall, and F1-score

⸻

	6.	Machine Learning Pipelines

All models were implemented using scikit-learn pipelines that include preprocessing steps such as standardization and encoding together with model training. This ensured consistent preprocessing, prevented data leakage, and allowed reproducible model deployment.

⸻

	7.	Streamlit Application

An interactive Streamlit web application was developed as part of the project.

The application includes:
	•	Separate tabs for exploratory data analysis, regression diagnostics, classification diagnostics, and prediction
	•	User-selectable prediction tasks (wine quality prediction or wine type classification)
	•	Real-time predictions based on user input
	•	Model diagnostics and visualizations embedded directly into the app

⸻

	8.	Extra Work Beyond Requirements

The following components go beyond the original assignment requirements:
	•	Additional wine type classification problem
	•	Fully interactive Streamlit application
	•	Saved trained models using joblib
	•	Separate training and inference scripts
	•	Explicit prevention of target leakage
	•	Diagnostic plots integrated into the user interface

⸻

	9.	Project Structure

wine_app/
app.py
regression_model.py
classification_model.py
README.txt
winequality-white.csv
winequality-red.csv
models/
regression_rf.joblib
classification_rf.joblib

⸻

	10.	How to Run the Project

To train the models:
python regression_model.py
python classification_model.py

To launch the application:
streamlit run app.py
