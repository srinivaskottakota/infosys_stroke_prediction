# infosys_stroke_prediction

------------------------------------------------------------------------
## 1. Importing Libraries

import pandas as pd

 - Explanation: This line imports the Pandas library, which is crucial for data manipulation and analysis in Python. It provides powerful data structures like DataFrames.

## 2. Loading the Dataset

df = pd.read_csv('dataset/data.csv')

 - Explanation: This command loads the dataset from a specified CSV file into a Pandas DataFrame called df. This structure allows for easy data manipulation and analysis.

## 3. Basic Exploration

### Descriptive Statistics

print("Basic statistical description of numerical data:")
print(df.describe())

 - Explanation: This part computes and prints basic statistical metrics (mean, median, standard deviation, min, max, etc.) for all numerical columns in the DataFrame. It helps summarize the distribution of the data.
 - 
### Dataset Information
 
print("\nDataset information (columns, types, and non-null counts):")
print(df.info())
 - Explanation: This command provides a summary of the DataFrame, including the number of entries, column names, data types, and counts of non-null values for each column. This information is useful for understanding the structure of the dataset.

### Shape of the Dataset

print("\nShape of the dataset (rows, columns):")
print(df.shape)
 - Explanation: This line outputs the shape of the DataFrame, indicating the number of rows and columns. It provides a quick overview of the dataset size.

### 4. Categorical Data Description

print("\nBasic statistical description of categorical data:")
print(df.describe(include='object'))
 - Explanation: This command provides a summary of categorical columns, showing counts, unique values, the most frequent value, and its frequency. It helps understand the distribution of categorical data.

## 5. Unique Values and Null Value Analysis
### Unique Values in 'gender'

print("\nUnique values in 'gender' column:")
gender_unique = df['gender'].unique()
print(gender_unique)
 - Explanation: This section retrieves and prints the unique values present in the 'gender' column, helping to understand the diversity in this categorical variable.

### Unique Values in 'smoking_status'

print("\nUnique values in 'smoking_status' column:")
smoking_status_unique = df['smoking_status'].unique()
print(smoking_status_unique)
 - Explanation: Similar to the previous part, this retrieves unique values from the 'smoking_status' column, providing insight into the smoking categories present in the dataset.

### Null Values Count

null_values = df.isnull().sum()
print("\nNull values in each column:")
print(null_values)
 - Explanation: This command counts the number of null (missing) values in each column and prints the results. It is essential for assessing data quality.

### Percentage of Null Values

null_percentage = df.isnull().mean() * 100 
print("\nPercentage of null values in each column:")
print(null_percentage)
- Explanation: This calculates and prints the percentage of null values in each column, providing a clearer picture of missing data relative to the size of each column.

## 6. Observations
### Data Types

print("\nData types of each column:")
print(df.dtypes)
 - Explanation: This displays the data types of each column in the DataFrame. Understanding data types is critical for performing appropriate analyses.

### Key Observations

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
- Explanation: This section summarizes key observations:
- The number of rows and columns in the dataset.
- The count of missing values in the 'bmi' column.
- The unique values present in the 'gender' and 'smoking_status' columns.
- A summary of columns with missing data and their respective percentages.

## 7. Handling Missing Values
### Dropping Rows with Missing Values

df_dropped = df.dropna(subset=['bmi'])
print(f"6. After dropping rows with missing 'bmi' values, the dataset contains {df_dropped.shape[0]} rows.")
 - Explanation: This part drops any rows where the 'bmi' column has missing values, and it prints the new row count. This is one way to handle missing data.

### Imputing Missing Values with Mean

mean_bmi = df['bmi'].mean()
df['bmi'].fillna(mean_bmi, inplace=True)
print(f"\nImputing missing 'bmi' values with mean value: {mean_bmi}")
 - Explanation: Instead of dropping rows, this option fills any missing 'bmi' values with the mean of the column. It helps retain more data while addressing missing values.

### Checking Null Values After Imputation

null_values_after = df.isnull().sum()
print("\nNull values after imputing missing 'bmi' values:")
print(null_values_after)
 - Explanation: This checks and prints the count of null values after the imputation process, confirming whether the missing values have been successfully addressed.

### 8. Checking for Duplicate Rows

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
- Explanation: This counts and prints the number of duplicate rows in the dataset. Identifying duplicates is important for ensuring data integrity

## 9. Analyzing Stroke Rate by Gender
### Stroke Rate Calculation

print("\nStroke rate by gender:")
print(df.groupby('gender')['stroke'].mean())
- Explanation: This computes and prints the average stroke rate by gender, showing the proportion of individuals with strokes in each gender category.

### Stroke Percentage Calculation

total = df['stroke'].sum()
strokes_gender = df[df['stroke'] == 1].groupby('gender')['stroke'].count()
stroke_per = (strokes_gender / total) * 100
print("\nStroke percentage by gender (relative to all stroke cases):")
print(stroke_per)
- Explanation: This section calculates the total number of strokes and the count of strokes grouped by gender. It then computes the percentage of strokes by gender relative to all stroke cases and prints the results.


