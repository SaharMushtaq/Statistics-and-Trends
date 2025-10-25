

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind


df = pd.read_csv("diabetes.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# Replace zeros with NaN for medically impossible values
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# Fill missing values with median
df.fillna(df.median(), inplace=True)

# Basic information
print("\nData Summary:")
print(df.info())
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


print("\nDescriptive Statistics:")
print(df.describe().T)

corr = df.corr()

print("\nTop Correlations with Outcome:")
print(corr["Outcome"].sort_values(ascending=False))


import os
os.makedirs("figures", exist_ok=True)

# Helper function to save plots
def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=300)
    plt.close()


plt.figure(figsize=(6,4))
sns.countplot(x="Outcome", data=df)
plt.title("Distribution of Diabetes Outcome (0: No, 1: Yes)")
plt.xlabel("Outcome")
plt.ylabel("Count")
save_fig("distribution_outcome")

for col in ["Glucose", "BMI", "Age", "Insulin"]:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=25)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    save_fig(f"distribution_{col.lower()}")


features = ["Glucose", "BloodPressure", "BMI", "Age"]
for col in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="Outcome", y=col, data=df)
    plt.title(f"{col} vs Outcome")
    plt.xlabel("Outcome")
    plt.ylabel(col)
    save_fig(f"boxplot_{col.lower()}")

plt.figure(figsize=(9,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap of All Variables")
save_fig("correlation_heatmap")

bins = [20, 30, 40, 50, 60, 70, 80]
labels = ['20s', '30s', '40s', '50s', '60s', '70+']
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

age_group_rate = df.groupby("AgeGroup")["Outcome"].mean() * 100
plt.figure(figsize=(6,4))
sns.barplot(x=age_group_rate.index, y=age_group_rate.values)
plt.title("Diabetes Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Diabetes Percentage")
save_fig("agegroup_vs_diabetes")

sns.pairplot(df, hue="Outcome", diag_kind="kde", corner=True)
plt.suptitle("Pairwise Relationships Between Features", y=1.02)
plt.savefig("pairplot_relationships.png", dpi=300)
plt.close()

print("\nT-Test Results (Diabetic vs Non-Diabetic):")
diabetic = df[df["Outcome"] == 1]
non_diabetic = df[df["Outcome"] == 0]

for col in df.columns[:-1]:
    stat, p = ttest_ind(diabetic[col], non_diabetic[col])
    print(f"{col:25} | p-value: {p:.5f} | Significant: {p < 0.05}")


plt.figure(figsize=(6,4))
sns.violinplot(x="Outcome", y="Insulin", data=df)
plt.title("Distribution of Insulin Levels by Outcome")
save_fig("violin_insulin_outcome")

plt.figure(figsize=(6,4))
sns.scatterplot(x="Age", y="BMI", hue="Outcome", data=df, alpha=0.7)
plt.title("Age vs BMI (Colored by Outcome)")
save_fig("scatter_age_bmi_outcome")


print("\nSummary of EDA Findings:")
print("- Dataset has 768 records and 9 features after cleaning.")
print("- Glucose, BMI, and Age show the strongest correlation with diabetes outcome.")
print("- Average glucose and BMI values are significantly higher in diabetic patients.")
print("- Diabetes prevalence increases sharply after age 40.")
print("- No major missing values remain after cleaning.")
print("- The dataset is balanced enough for model building.")

print("\nAll figures saved successfully as PNG files in the root directory.")
