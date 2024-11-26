import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay

# Suppress warnings to keep the output clean
warnings.filterwarnings('ignore')

# Load the dataset
# The dataset contains various features like gender, smoking status, work type, and more.
# 'stroke' is the target variable that indicates whether the person had a stroke (1) or not (0).
df = pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\infosys_stroke_prdiction\\infosys_stroke_prediction\\milestone_4\\data.csv")

# Handle missing values in the 'bmi' column
# The 'bmi' column may have missing values; we replace them with the median value to avoid errors during training.
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# One-hot encoding of categorical features to convert them into numerical values.
# This step is necessary as machine learning algorithms can only work with numerical data.
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

# Drop the original categorical columns (now encoded in the dataset) to avoid multicollinearity.
df_model = df.copy()
df_model.drop(['Residence_type', 'work_type', 'smoking_status', 'gender', 'ever_married'], axis=1, inplace=True)

# Define feature matrix X (input variables) and target vector y (output variable)
# 'X' contains all the features used for prediction (like age, bmi, work_type, etc.)
# 'y' is the target variable indicating if a stroke occurred (1) or not (0).
X = df_model.drop('stroke', axis=1)  # Features
y = df_model['stroke']  # Target variable

# Split the data into training and testing sets
# We use 80% of the data for training and 20% for testing to evaluate the model's performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Logistic Regression Model
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)  # Train the model on the training data
logistic_pred = logistic_reg.predict(X_test)  # Make predictions on the test data

# Calculate the Root Mean Squared Error (RMSE) and Accuracy of the Logistic Regression model
# RMSE is a good indicator of how much the predicted values deviate from the actual values.
log_reg_rmse = np.sqrt(mean_squared_error(y_test, logistic_pred))
logistic_reg_acc = logistic_reg.score(X_test, y_test)  # Accuracy is the proportion of correct predictions
print(f"Logistic Regression  Score: {logistic_reg_acc * 100:.2f}%")
print(f"Logistic Regression RMSE: {log_reg_rmse * 100:.2f}%")

# Generate and display the confusion matrix to assess the model's performance in terms of true positives, 
# true negatives, false positives, and false negatives.
conf_matrix = confusion_matrix(y_test, logistic_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('Confused Matrix')
plt.show()

# Calculate and display precision, recall, and F1 score to further assess the classification performance.
# - Precision: How many selected items are relevant (out of all predicted positive cases).
# - Recall: How many relevant items are selected (out of all actual positive cases).
# - F1 Score: A harmonic mean of precision and recall, which balances both metrics.
precision = precision_score(y_test, logistic_pred)
f1 = f1_score(y_test, logistic_pred)
recall = recall_score(y_test, logistic_pred)
print(f"Precision Score: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Recall Score: {recall:.2f}")

# Generate and display the Precision-Recall curve using the estimator
# This shows how the precision and recall values change as the decision threshold is varied.
display = PrecisionRecallDisplay.from_estimator(logistic_reg, X_test, y_test, name="Logistic Regression", plot_chance_level=True)
display.ax_.set_title("2-class Precision-Recall curve")
plt.savefig('2-class Precision-Recall curve')
plt.show()

# Create a DataFrame to compare the performance metrics of the Logistic Regression model.
results = pd.DataFrame({
    'Model': ['Logistic Regression'],
    'Accuracy': [logistic_reg_acc],
    'RMSE': [log_reg_rmse]
})
print(results)

# Visualizing the model performance metrics (Accuracy and RMSE)
plt.figure(figsize=(8, 4))

# Accuracy Bar Plot: Displays how accurate the model is in predicting the target variable.
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Accuracy', data=results, palette='Blues_d')
plt.title('Model Accuracy')
plt.xticks(rotation=45)

# RMSE Bar Plot: Displays the error in the model's predictions (lower is better).
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='RMSE', data=results, palette='Reds_d')
plt.title('Model RMSE')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('Model RMSE')
plt.show()

