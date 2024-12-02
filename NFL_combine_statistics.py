# NFL combine statistics 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
# Simulated results based on data (classification accuracy and regression RMSE)
classification_results = {
    "Decision Tree": 0.589,
    "Gradient Booster": 0.685,
    "SVC": 0.739,
    "GaussianNB": 0.739,
    "Random Forest": 0.730,
    "Logistic Regression": 0.722,
}

regression_results = {
    "Decision Tree": 1731.3,  # Simulated RMSE
    "Gradient Booster": 1289.3,
    "Random Forest": 1276.0,
    "Linear Regression": 1210.1,
}

# Classification Accuracy Bar Plot
plt.figure(figsize=(10, 6))
plt.bar(classification_results.keys(), classification_results.values(), color="skyblue")
plt.title("Classification Model Accuracy", fontsize=16)
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(rotation=45)
for i, v in enumerate(classification_results.values()):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=12)
plt.tight_layout()
plt.show()

# Regression RMSE Bar Plot
plt.figure(figsize=(10, 6))
plt.bar(regression_results.keys(), regression_results.values(), color="lightcoral")
plt.title("Regression Model RMSE", fontsize=16)
plt.ylabel("RMSE (Root Mean Squared Error)", fontsize=14)
plt.xticks(rotation=45)
for i, v in enumerate(regression_results.values()):
    plt.text(i, v + 30, f"{v:.1f}", ha="center", fontsize=12)
plt.tight_layout()
plt.show()

file_paths = [
    '/Users/matteomartone/Desktop/Github/DS-340W-Matteo/NFL_2013_edit.xlsx',
    '/Users/matteomartone/Desktop/Github/DS-340W-Matteo/NFL_2014_edit.xlsx',
    '/Users/matteomartone/Desktop/Github/DS-340W-Matteo/NFL_2015_edit.xlsx',
    '/Users/matteomartone/Desktop/Github/DS-340W-Matteo/NFL_2016_edit.xlsx',
    '/Users/matteomartone/Desktop/Github/DS-340W-Matteo/NFL_2017_edit.xlsx'
]

# Initialize a list to store DataFrames
dfs = []

# Read and store each file
for file_path in file_paths:
    df = pd.read_excel(file_path)
    dfs.append(df)

# Merge all DataFrames on the 'Year' column
merged_data = pd.concat(dfs, ignore_index=True)

# Selecting relevant columns for analysis
columns_of_interest = ['Pos','Year', '40yd', 'BP', 'Vertical', 'Broad Jump', 'Shuttle', '3Cone']
merged_data = merged_data[columns_of_interest]

for column in ['40yd', 'BP', 'Vertical', 'Broad Jump', 'Shuttle', '3Cone']:
    merged_data[column] = pd.to_numeric(merged_data[column], errors='coerce')

# Checking for remaining non-numeric values or NaNs
missing_values_summary = merged_data.isna().sum()
missing_values_summary

# Plotting the boxplot for the six tests
plt.figure(figsize=(12, 6))
merged_data[['40yd', 'BP', 'Vertical', 'Broad Jump', 'Shuttle', '3Cone']].boxplot()
plt.title('Boxplot of NFL Combine Test Results')
plt.ylabel('Scores')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Checking data types of the relevant columns
data_types = merged_data.dtypes

# Display non-numeric columns in the data
non_numeric_columns = data_types[data_types != 'float64'][data_types != 'int64']
non_numeric_columns


# Computing descriptive statistics grouped by 'Pos' for the six combine tests
descriptive_stats = merged_data.groupby('Pos')[['40yd', 'BP', 'Vertical', 'Broad Jump', 'Shuttle', '3Cone']].describe()

# Displaying the descriptive statistics as a table
print("Descriptive Statistics by Position:")
print(descriptive_stats)


# First graph: Correlation heatmap for combine tests
import seaborn as sns

correlation_matrix = merged_data[['40yd', 'BP', 'Vertical', 'Broad Jump', 'Shuttle', '3Cone']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Combine Test Results')
plt.show()

# Second graph: Position comparison for average scores across all tests
average_by_pos = merged_data.groupby('Pos')[['40yd', 'BP', 'Vertical', 'Broad Jump', 'Shuttle', '3Cone']].mean()

plt.figure(figsize=(14, 8))
average_by_pos.plot(kind='bar', figsize=(14, 8), width=0.8, alpha=0.7)
plt.title('Comparison of Average Combine Scores by Position')
plt.xlabel('Position')
plt.ylabel('Average Score')
plt.legend(title="Tests", loc='upper right', bbox_to_anchor=(1.15, 1))
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


