import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("C:\Users\Dell\OneDrive\Desktop\infosys_stroke_prdiction\infosys_stroke_prediction\milestone_4\data.csv")

# Fill missing values in 'bmi' column
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# One-hot encode categorical columns
df['gender_Male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['gender_Female'] = df['gender'].apply(lambda x: 1 if x == 'Female' else 0)
df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)

df['work_type_Private'] = df['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
df['work_type_Self_employed'] = df['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)
df['work_type_Govt_job'] = df['work_type'].apply(lambda x: 1 if x == 'Govt_job' else 0)
df['work_type_children'] = df['work_type'].apply(lambda x: 1 if x == 'children' else 0)
df['work_type_Never_worked'] = df['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)

df['Residence_type'] = df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)

df['smoking_status_formerly_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
df['smoking_status_never_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
df['smoking_status_smokes'] = df['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)
df['smoking_status_Unknown'] = df['smoking_status'].apply(lambda x: 1 if x == 'Unknown' else 0)

# Remove original categorical columns
df_model = df.copy()
df_model.drop(['Residence_type', 'work_type', 'smoking_status', 'gender', 'ever_married'], axis=1, inplace=True)

# Define X (features) and y (target variable)
X = df_model.drop('stroke', axis=1)
y = df_model['stroke']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Logistic Regression model
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
logistic_pred = logistic_reg.predict(X_test)

# Calculate RMSE and Accuracy for Logistic Regression
log_reg_rmse = np.sqrt(mean_squared_error(y_test, logistic_pred))
logistic_reg_acc = logistic_reg.score(X_test, y_test)
print(f"Logistic Regression  Score: {logistic_reg_acc * 100:.2f}%")
print(f"Logistic Regression RMSE: {log_reg_rmse * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, logistic_pred)

# Display Confusion Matrix
cm_display = PrecisionRecallDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
cm_display.plot()
plt.show()

# Precision, Recall, F1 Scores
precision = precision_score(y_test, logistic_pred)
f1 = f1_score(y_test, logistic_pred)
recall = recall_score(y_test, logistic_pred)
print(f"Precision Score: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Recall Score: {recall:.2f}")

# Precision-Recall Curve
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, logistic_pred)
print(f"Thresholds: {thresholds}")
display = PrecisionRecallDisplay.from_estimator(logistic_reg, X_test, y_test, name="Logistic Regression", plot_chance_level=True)
display.ax_.set_title("2-class Precision-Recall curve")
plt.show()

# Comparison of models
results = pd.DataFrame({
    'Model': ['Logistic Regression'],
    'Accuracy': [logistic_reg_acc],
    'RMSE': [log_reg_rmse]
})
print(results)

# Visualizing the results
plt.figure(figsize=(8, 4))

# Accuracy Bar Plot
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Accuracy', data=results, palette='Blues_d')
plt.title('Model Accuracy')
plt.xticks(rotation=45)

# RMSE Bar Plot
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='RMSE', data=results, palette='Reds_d')
plt.title('Model RMSE')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

