import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())


# -----------------------------
# Split Features and Target
# -----------------------------

X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]


# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# -----------------------------
# Train KNN Model
# -----------------------------

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train,
    y_train
)


# -----------------------------
# Model Evaluation
# -----------------------------

prediction = knn_model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    prediction
)

print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    classification_report(
        y_test,
        prediction
    )
)


# -----------------------------
# Save Model
# -----------------------------

with open(
    "model.pkl",
    "wb"
) as file:

    pickle.dump(
        {
            "model": knn_model,
            "scaler": scaler
        },
        file
    )


print("Model Saved Successfully as model.pkl")