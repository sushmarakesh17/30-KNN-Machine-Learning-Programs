import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


st.set_page_config(
    page_title="30 KNN Programs",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 30 KNN Programs - Machine Learning")
st.write(
    "K-Nearest Neighbors (KNN) implementation with multiple datasets"
)


# -------------------------------
# Select Program
# -------------------------------

program = st.sidebar.selectbox(
    "Select KNN Program",
    [
        "1. Iris Flower Classification",
        "2. Breast Cancer Classification",
        "3. Wine Classification",
        "4. Diabetes Prediction",
        "5. Student Performance Prediction",
        "6. Customer Churn Prediction",
        "7. Loan Approval Prediction",
        "8. Heart Disease Prediction",
        "9. Car Price Prediction",
        "10. House Price Prediction"
    ]
)


# -------------------------------
# Dataset Creation
# -------------------------------

def load_dataset(program):

    from sklearn.datasets import (
        load_iris,
        load_breast_cancer,
        load_wine,
        load_diabetes
    )

    if "Iris" in program:

        data = load_iris()

        X = pd.DataFrame(
            data.data,
            columns=data.feature_names
        )

        y = pd.Series(data.target)

        task = "classification"


    elif "Breast Cancer" in program:

        data = load_breast_cancer()

        X = pd.DataFrame(
            data.data,
            columns=data.feature_names
        )

        y = pd.Series(data.target)

        task = "classification"


    elif "Wine" in program:

        data = load_wine()

        X = pd.DataFrame(
            data.data,
            columns=data.feature_names
        )

        y = pd.Series(data.target)

        task = "classification"


    elif "Diabetes" in program:

        data = load_diabetes()

        X = pd.DataFrame(
            data.data,
            columns=data.feature_names
        )

        y = pd.Series(data.target)

        task = "regression"


    else:

        np.random.seed(42)

        X = pd.DataFrame(
            np.random.rand(200,5),
            columns=[
                "Feature1",
                "Feature2",
                "Feature3",
                "Feature4",
                "Feature5"
            ]
        )

        y = pd.Series(
            np.random.randint(0,2,200)
        )

        task="classification"


    return X,y,task



# -------------------------------
# Load Data
# -------------------------------

X,y,task = load_dataset(program)


st.subheader("Dataset Preview")

st.dataframe(X.head())


# -------------------------------
# Train Test Split
# -------------------------------

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# -------------------------------
# KNN Model
# -------------------------------

k = st.slider(
    "Select Number of Neighbours (K)",
    1,
    20,
    5
)


if task=="classification":

    model = KNeighborsClassifier(
        n_neighbors=k
    )

else:

    model = KNeighborsRegressor(
        n_neighbors=k
    )


model.fit(
    X_train,
    y_train
)


# -------------------------------
# Prediction
# -------------------------------

prediction = model.predict(
    X_test
)



st.subheader("Model Result")


if task=="classification":

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    st.success(
        f"Accuracy : {accuracy:.2%}"
    )


else:

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    r2 = r2_score(
        y_test,
        prediction
    )


    st.success(
        f"MAE : {mae:.2f}"
    )

    st.success(
        f"R2 Score : {r2:.2f}"
    )



# -------------------------------
# Custom Prediction
# -------------------------------

st.subheader("Make New Prediction")


input_data=[]


for col in X.columns:

    value = st.number_input(
        col,
        value=float(X[col].mean())
    )

    input_data.append(value)



if st.button("Predict"):

    user_input = scaler.transform(
        [input_data]
    )

    result = model.predict(
        user_input
    )


    st.info(
        f"Prediction Result : {result[0]}"
    )



st.sidebar.success(
    "KNN Machine Learning Project"
)