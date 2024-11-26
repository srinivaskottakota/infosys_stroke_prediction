# Milestone 1: Data Exploration and Preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings to keep the output clean
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\infosys_stroke_prdiction\\infosys_stroke_prediction\\milestone_4\\data.csv")

# Basic Exploration

# Descriptive Statistics
print("Basic statistical description of numerical data:")
print(df.describe())

# Dataset Information
print("\nDataset information (columns, types, and non-null counts):")
print(df.info())

# Shape of the Dataset
print("\nShape of the dataset (rows, columns):")
print(df.shape)

# Categorical Data Description
print("\nBasic statistical description of categorical data:")
print(df.describe(include='object'))

# Unique Values and Null Value Analysis
print("\nUnique values in 'gender' column:")
print(df['gender'].unique())

print("\nUnique values in 'smoking_status' column:")
print(df['smoking_status'].unique())

# Null Values Count
null_values = df.isnull().sum()
print("\nNull values in each column:")
print(null_values)

# Percentage of Null Values
null_percentage = df.isnull().mean() * 100 
print("\nPercentage of null values in each column:")
print(null_percentage)

# Handling Missing Values
# Dropping Rows with Missing 'bmi' Values
df_dropped = df.dropna(subset=['bmi'])
print(f"After dropping rows with missing 'bmi' values, the dataset contains {df_dropped.shape[0]} rows.")

# Imputing Missing Values with Mean
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# Checking Null Values After Imputation
null_values_after = df.isnull().sum()
print("\nNull values after imputing missing 'bmi' values:")
print(null_values_after)

# Task 2: Data Visualization

# Framing Questions
# 1. What is the age distribution among patients?
# 2. How does average glucose level differ between patients with and without a stroke?
# 3. Is there a relationship between hypertension and the likelihood of having a stroke?
# 4. What is the distribution of residence types (Urban/Rural) among stroke patients?
# 5. How do BMI and average glucose levels interact in relation to stroke status?

# Set the visualization style
sns.set(style="whitegrid")

# Task 2: Plot the Required Graphs

# 1. Age Distribution among patients
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig('age_distribution.png') 
plt.show()

# Observation: The age distribution is roughly bimodal, with peaks around 40-60 and 80 years.

# 2. Average Glucose Levels by Stroke Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='stroke', y='avg_glucose_level', data=df)
plt.title("Average Glucose Level by Stroke Status")
plt.xlabel("Stroke")
plt.ylabel("Average Glucose Level")
plt.savefig('glucose_by_stroke.png')  
plt.show()

# Observation: Stroke patients tend to have higher glucose levels than non-stroke patients.

# 3. Hypertension vs Stroke Status
plt.figure(figsize=(8, 6))
sns.countplot(x='hypertension', data=df)
plt.title("Hypertension vs Stroke Status")
plt.xlabel("Hypertension")
plt.ylabel("Count")
plt.legend(title="Stroke")
plt.savefig('hypertension_vs_stroke.png') 
plt.show()

# Observation: The majority of hypertensive patients in this dataset have not had a stroke.

# 4. Residence Type Distribution among Stroke Patients
stroke_residence = df[df['stroke'] == 1]['Residence_type'].value_counts()
plt.figure(figsize=(6, 6))
stroke_residence.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title("Residence Type Distribution Among Stroke Patients")
plt.ylabel("")  
plt.savefig('residence_type_distribution.png')  
plt.show()

# Observation: A higher proportion of stroke patients live in urban areas (54.2%) compared to rural areas (45.8%).

# 5. BMI vs Average Glucose Level by Stroke Status
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='avg_glucose_level', hue='stroke', data=df, palette="viridis")
plt.title("BMI vs Average Glucose Level by Stroke Status")
plt.xlabel("BMI")
plt.ylabel("Average Glucose Level")
plt.legend(title="Stroke")
plt.savefig('bmi_vs_glucose.png') 
plt.show()

# Observation: Stroke patients appear to have higher glucose levels and a wider range of BMIs.

# Task 3: Data Encoding

# Convert Residence_type to Urban/Rural
df['Urban/Rural'] = df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)

# Convert work_type to Binary Columns
df['work_type_Never_worked'] = df['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)
df['work_type_Private'] = df['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
df['work_type_Self_employed'] = df['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)

# Convert smoking_status to Binary Columns
df['smoking_status_formerly_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
df['smoking_status_never_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
df['smoking_status_smokes'] = df['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)

# Create a new dataset for the model and drop the original categorical columns
model_data = df.copy()
model_data.drop(['Residence_type', 'work_type', 'smoking_status'], axis=1, inplace=True)

# Save the transformed data for modeling
model_data.to_csv('processed_data.csv', index=False)

# Print the first few rows of the transformed dataset
print(model_data.head())

# Milestone 4: Machine Learning Model - Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
import seaborn as sns

# Split data into features and target
X = model_data.drop('stroke', axis=1)
y = model_data['stroke']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)

# Model Evaluation
log_reg_acc = accuracy_score(y_test, log_reg_preds)
log_reg_rmse = np.sqrt(mean_squared_error(y_test, log_reg_preds))
log_reg_precision = precision_score(y_test, log_reg_preds)
log_reg_recall = recall_score(y_test, log_reg_preds)
log_reg_f1 = f1_score(y_test, log_reg_preds)

print(f"Logistic Regression Accuracy: {log_reg_acc * 100:.2f}%")
print(f"Logistic Regression RMSE: {log_reg_rmse:.2f}")
print(f"Logistic Regression Precision: {log_reg_precision:.2f}")
print(f"Logistic Regression Recall: {log_reg_recall:.2f}")
print(f"Logistic Regression F1 Score: {log_reg_f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, log_reg_preds)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('log_reg_confusion_matrix.png')
plt.show()

# Observations and Recommendations:
# 1. The accuracy of the Logistic Regression model is relatively high.
# 2. Precision, recall, and F1 score provide a good balance between positive and negative classes.
# 3. The confusion matrix indicates how well the model is distinguishing between stroke and non-stroke patients.
# 4. Consider balancing the dataset if it's imbalanced, using techniques like SMOTE, or adjusting class weights.

