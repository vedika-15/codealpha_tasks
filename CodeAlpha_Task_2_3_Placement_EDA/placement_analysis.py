import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("placement_data.csv")

# -------- EDA --------
print(df.head())
print(df.info())
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Placement count
print("\nPlacement Status:")
print(df['status'].value_counts())

# -------- Visualization --------

# Correlation heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# Placement distribution
sns.countplot(x='status', data=df)
plt.title("Placement Status Distribution")
plt.show()

# Gender vs Placement
sns.countplot(x='gender', hue='status', data=df)
plt.title("Gender vs Placement")
plt.show()

# SSC percentage vs placement
sns.boxplot(x='status', y='ssc_percentage', data=df)
plt.title("SSC Percentage vs Placement")
plt.show()

# Interview score vs placement
sns.boxplot(x='status', y='interview_score', data=df)
plt.title("Interview Score vs Placement")
plt.show()

# Work experience vs placement
sns.countplot(x='work_experience', hue='status', data=df)
plt.title("Work Experience vs Placement")
plt.show()

print("EDA and Visualization Completed Successfully")
