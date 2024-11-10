# 1. Importing Libraries
import pandas as pd

# 2. Loading the Dataset
df = pd.read_csv('C:/Users/Dell/OneDrive/Desktop/infosys internship/data.csv')  

# 3. Basic Exploration

# Descriptive Statistics
print("Basic statistical description of numerical data:")
print(df.describe())

# Dataset Information
print("\nDataset information (columns, types, and non-null counts):")
print(df.info())

# Shape of the Dataset
print("\nShape of the dataset (rows, columns):")
print(df.shape)

# 4. Categorical Data Description
print("\nBasic statistical description of categorical data:")
print(df.describe(include='object'))

# 5. Unique Values and Null Value Analysis
# Unique Values in 'gender'
print("\nUnique values in 'gender' column:")
gender_unique = df['gender'].unique()
print(gender_unique)

# Unique Values in 'smoking_status'
print("\nUnique values in 'smoking_status' column:")
smoking_status_unique = df['smoking_status'].unique()
print(smoking_status_unique)

# Null Values Count
null_values = df.isnull().sum()
print("\nNull values in each column:")
print(null_values)

# Percentage of Null Values
null_percentage = df.isnull().mean() * 100 
print("\nPercentage of null values in each column:")
print(null_percentage)

# 6. Observations
# Data Types
print("\nData types of each column:")
print(df.dtypes)

# Key Observations
print(f"1. The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
missing_bmi_count = null_values['bmi']
if missing_bmi_count > 0:
    print(f"2. The 'bmi' column originally contained {missing_bmi_count} missing values.")
else:
    print("2. The 'bmi' column contains no missing values.")
print(f"3. The 'gender' column contains the following unique values: {gender_unique}")
print(f"4. The 'smoking_status' column contains the following unique values: {smoking_status_unique}")
print(f"5. The dataset contains missing data in the following columns (with percentages):")
print(null_percentage[null_percentage > 0])

# 7. Handling Missing Values
# Dropping Rows with Missing Values
df_dropped = df.dropna(subset=['bmi'])
print(f"6. After dropping rows with missing 'bmi' values, the dataset contains {df_dropped.shape[0]} rows.")

# Imputing Missing Values with Mean
mean_bmi = df['bmi'].mean()
df['bmi'].fillna(mean_bmi, inplace=True)
print(f"\nImputing missing 'bmi' values with mean value: {mean_bmi}")

# Checking Null Values After Imputation
null_values_after = df.isnull().sum()
print("\nNull values after imputing missing 'bmi' values:")
print(null_values_after)

# 8. Checking for Duplicate Rows
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# 9. Analyzing Stroke Rate by Gender
# Stroke Rate Calculation
print("\nStroke rate by gender:")
print(df.groupby('gender')['stroke'].mean())

# Stroke Percentage Calculation
total = df['stroke'].sum()
strokes_gender = df[df['stroke'] == 1].groupby('gender')['stroke'].count()
stroke_per = (strokes_gender / total) * 100
print("\nStroke percentage by gender (relative to all stroke cases):")
print(stroke_per)
