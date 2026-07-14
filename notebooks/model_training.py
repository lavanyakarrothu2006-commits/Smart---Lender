import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_train.csv")

# Drop Loan_ID
df = df.drop("Loan_ID", axis=1)

# Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

best_model = None
best_accuracy = 0

print("\nModel Accuracies\n")

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save best model
joblib.dump(best_model, "model/best_model.pkl")

print("\nBest Model Saved Successfully!")
print("Best Accuracy:", best_accuracy)