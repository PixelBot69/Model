import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix


st.title("Marketing Campaign Prediction App")


st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Overview")
    st.write(data.head())

    st.sidebar.header("Select Features and Target")
    features = st.sidebar.multiselect("Select Features for Modeling", data.columns)
    target = st.sidebar.selectbox("Select Target Variable", data.columns)

    if features and target:
        st.write(f"### Selected Features: {features}")
        st.write(f"### Target Variable: {target}")

 
        X = data[features]
        y = data[target]

        st.sidebar.header("Select Model Type")
        model_type = st.sidebar.selectbox("Choose a Model", ["Linear Regression", "Random Forest Classifier"])

        num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_features = X.select_dtypes(include=["object"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(), cat_features),
            ]
        )

  
        if model_type == "Linear Regression":
            model = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])
        else:
            model = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier(random_state=42))])

  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)


        y_pred = model.predict(X_test)

     
        st.write("### Model Evaluation")
        if model_type == "Linear Regression":
            st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        else:
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))


        st.write("### Make Predictions on New Data")
        input_data = {}
        for feature in features:
            if feature in cat_features:
                input_data[feature] = st.text_input(f"Enter value for {feature}")
            else:
                input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            st.write(f"### Prediction: {prediction[0]}")
