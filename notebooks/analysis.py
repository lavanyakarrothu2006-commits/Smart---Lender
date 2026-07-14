import os
import pandas as pd
import matplotlib.pyplot as plt

# Show current working directory
print("Current Working Directory:", os.getcwd())

# Load the dataset
df = pd.read_csv("dataset/train.csv")

# Print dataset details
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Loan Status Distribution
plt.figure(figsize=(5, 4))
df["Loan_Status"].value_counts().plot(kind="bar")
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()

# Gender Distribution
plt.figure(figsize=(5, 4))
df["Gender"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.show()

# Property Area Distribution
plt.figure(figsize=(5, 4))
df["Property_Area"].value_counts().plot(kind="bar")
plt.title("Property Area Distribution")
plt.show()

# Applicant Income Distribution
plt.figure(figsize=(6, 4))
plt.hist(df["ApplicantIncome"], bins=20)
plt.title("Applicant Income Distribution")
plt.xlabel("Applicant Income")
plt.ylabel("Frequency")
plt.show()