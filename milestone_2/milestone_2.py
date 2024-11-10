import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('C:/Users/Dell/OneDrive/Desktop/infosys internship/data.csv')


# Fill missing BMI values with the mean
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

# Set the visualization style
sns.set(style="whitegrid")

# --- Data Visualization --- #

# Task 1: Framing Questions
# 1. What is the age distribution among patients?
# 2. How does average glucose level differ between patients with and without a stroke?
# 3. Is there a relationship between hypertension and the likelihood of having a stroke?
# 4. What is the distribution of residence types (Urban/Rural) among stroke patients?
# 5. How do BMI and average glucose levels interact in relation to stroke status?

# Task 2: Plot the Required Graphs

# 1. Age Distribution among patients
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig('age_distribution.png') 
plt.show()

# Observation: The age distribution is roughly bimodal, with peaks around 40-60 and 80 years.

# 2. Average Glucose Levels by Stroke Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='stroke', y='avg_glucose_level', data=data)
plt.title("Average Glucose Level by Stroke Status")
plt.xlabel("Stroke")
plt.ylabel("Average Glucose Level")
plt.savefig('glucose_by_stroke.png')  
plt.show()

# Observation: Stroke patients tend to have higher glucose levels than non-stroke patients.

# 3. Hypertension vs Stroke Status
plt.figure(figsize=(8, 6))
sns.countplot(x='hypertension', data=data)
plt.title("Hypertension vs Stroke Status")
plt.xlabel("Hypertension")
plt.ylabel("Count")
plt.legend(title="Stroke")
plt.savefig('hypertension_vs_stroke.png') 
plt.show()

# Observation: The majority of hypertensive patients in this dataset have not had a stroke.

# 4. Residence Type Distribution among Stroke Patients
stroke_residence = data[data['stroke'] == 1]['Residence_type'].value_counts()
plt.figure(figsize=(6, 6))
stroke_residence.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title("Residence Type Distribution Among Stroke Patients")
plt.ylabel("")  
plt.savefig('residence_type_distribution.png')  
plt.show()

# Observation: A higher proportion of stroke patients live in urban areas (54.2%) compared to rural areas (45.8%).

# 5. BMI vs Average Glucose Level by Stroke Status
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='avg_glucose_level', hue='stroke', data=data, palette="viridis")
plt.title("BMI vs Average Glucose Level by Stroke Status")
plt.xlabel("BMI")
plt.ylabel("Average Glucose Level")
plt.legend(title="Stroke")
plt.savefig('bmi_vs_glucose.png') 
plt.show()

# Observation: Stroke patients appear to have higher glucose levels and a wider range of BMIs.

# --- Data Encoding --- #

# Task 1: Convert Residence_type to Urban/Rural
data['Urban/Rural'] = data['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)

# Task 2: Convert work_type to Binary Columns
data['work_type_Never_worked'] = data['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)
data['work_type_Private'] = data['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
data['work_type_Self_employed'] = data['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)

# Task 3: Convert smoking_status to Binary Columns
data['smoking_status_formerly_smoked'] = data['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
data['smoking_status_never_smoked'] = data['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
data['smoking_status_smokes'] = data['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)

# Task 4: All columns should only have 0 or 1 values (already handled in the previous steps)

# Task 5: Create a new dataset for the model and drop the original categorical columns
model_data = data.copy()
model_data.drop(['Residence_type', 'work_type', 'smoking_status'], axis=1, inplace=True)

# Save the transformed data for modeling
model_data.to_csv('processed_data.csv', index=False)

# Print the first few rows of the transformed dataset
print(model_data.head())
