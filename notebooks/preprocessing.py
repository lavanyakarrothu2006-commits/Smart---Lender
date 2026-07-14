import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset/train.csv", sep="\t")
print(df.shape)
print(df.columns)
print(df.head())
print("Original Dataset Shape:", df.shape)

# Fill missing values
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

encoder = LabelEncoder()

for col in [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]:
    df[col] = encoder.fit_transform(df[col])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

df.to_csv("dataset/cleaned_train.csv", index=False)

print("\nCleaned dataset saved successfully!")