import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("C:/Users/Dell/OneDrive/Desktop/infosys internship/data.csv")

# Fill missing BMI values with the median
data['bmi'] = data['bmi'].fillna(data['bmi'].median())

# Convert categorical variables into dummy/indicator variables
data['is_male'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
data['is_female'] = data['gender'].apply(lambda x: 1 if x == 'Female' else 0)
data['is_married'] = data['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
data['work_private'] = data['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
data['work_self_employed'] = data['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)
data['work_gov_job'] = data['work_type'].apply(lambda x: 1 if x == 'Govt_job' else 0)
data['work_children'] = data['work_type'].apply(lambda x: 1 if x == 'children' else 0)
data['work_never_worked'] = data['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)
data['is_urban'] = data['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)

# Convert smoking status to indicator variables
data['smokes_formerly'] = data['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
data['smokes_never'] = data['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
data['smokes_currently'] = data['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)
data['smoking_status_unknown'] = data['smoking_status'].apply(lambda x: 1 if x == 'Unknown' else 0)

# Create a new dataframe for modeling
model_data = data.copy()
model_data.drop(['Residence_type', 'work_type', 'smoking_status', 'gender', 'ever_married'], axis=1, inplace=True)

# Split data into features and target
X = model_data.drop('stroke', axis=1)
y = model_data['stroke']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_preds = lin_reg.predict(X_test)
lin_reg_rmse = np.sqrt(mean_squared_error(y_test, lin_reg_preds)) * 100
lin_reg_acc = lin_reg.score(X_test, y_test) * 100
print(f"Linear Regression Accuracy: {lin_reg_acc:.2f}%")
print(f"Linear Regression RMSE: {lin_reg_rmse:.2f}%")

# Lasso Regression model
lasso = Lasso()
lasso.fit(X_train, y_train)
lasso_preds = lasso.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_preds)) * 100
lasso_acc = lasso.score(X_test, y_test) * 100
print(f"Lasso Regression Accuracy: {lasso_acc:.2f}%")
print(f"Lasso Regression RMSE: {lasso_rmse:.2f}%")

# Ridge Regression model
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds)) * 100
ridge_acc = ridge.score(X_test, y_test) * 100
print(f"Ridge Regression Accuracy: {ridge_acc:.2f}%")
print(f"Ridge Regression RMSE: {ridge_rmse:.2f}%")

# Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)
log_reg_rmse = np.sqrt(mean_squared_error(y_test, log_reg_preds)) * 100
log_reg_acc = log_reg.score(X_test, y_test) * 100
print(f"Logistic Regression Accuracy: {log_reg_acc:.2f}%")
print(f"Logistic Regression RMSE: {log_reg_rmse:.2f}%")

# Store results in a dataframe
model_results = pd.DataFrame({
    'Model': ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Logistic Regression'],
    'Accuracy': [lin_reg_acc, lasso_acc, ridge_acc, log_reg_acc],
    'RMSE': [lin_reg_rmse, lasso_rmse, ridge_rmse, log_reg_rmse]
})

print(model_results)

# Plot Accuracy vs RMSE
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Accuracy', data=model_results, palette='Blues_d')
plt.title('Model Accuracy')
plt.xticks(rotation=45)

# RMSE plot
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='RMSE', data=model_results, palette='Reds_d')
plt.title('Model RMSE')
plt.xticks(rotation=45)

# Layout and Save
plt.tight_layout()
plt.savefig("Accuracy_vs_RMSE.png")
plt.show()

