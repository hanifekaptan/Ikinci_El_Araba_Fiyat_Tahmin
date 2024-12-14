import pandas as pd
from sklearn.model_selection import train_test_split  # Splits the dataset
from sklearn.linear_model import LinearRegression  # Sets up the linear regression model
from sklearn.metrics import r2_score,mean_squared_error  # Measures the performance of the model
from sklearn.compose import ColumnTransformer  # Performs column transformation operations
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Performs transformation and scaling operations
from sklearn.pipeline import Pipeline  # Data processing pipeline
import streamlit as st  # Used for deploying the model

data = pd.read_excel("cars.xls")

X = data.drop("Price", axis=1)  # Independent variable
y = data["Price"]  # Dependent variable

# Data preprocessing, standardization, and One-Hot Encoding
preprocess = ColumnTransformer(
    transformers=[("num", StandardScaler(), ["Mileage", "Cylinder", "Liter", "Doors"]), # Standard scaler is applied to numeric data
                  ("cat", OneHotEncoder(), ["Make", "Model", "Trim", "Type"])])  # One-hot encoding is applied to categorical data

LR = LinearRegression()

# Pipeline
pipe = Pipeline(steps= [("preprocessor", preprocess), ("model", LR)])  # Ensures the transformation of data from the website

# It may be preferable to use the entire dataset for training since it provides a good prediction result
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# pipe.fit(X_train, y_train)  
# y_pred = pipe.predict(X_test)
# print("RMSE", mean_squared_error(y_test,y_pred, squared = False))
# print("R2", r2_score(y_test, y_pred))

pipe.fit(X, y)

# Prediction function
def price(mileage, make, model, trim, type, cylinder, liter, doors, cruise, sound, leather):
    input_data= pd.DataFrame({"Mileage": [mileage],
                              "Make": [make],
                              "Model": [model],
                              "Trim": [trim],
                              "Type": [type],
                              "Cylinder": [cylinder],
                              "Liter": [liter],
                              "Doors": [doors],
                              "Cruise": [cruise],
                              "Sound": [sound],
                              "Leather": [leather]})
    prediction = pipe.predict(input_data)[0]
    return prediction

st.title("Used Car Price Prediction :red_car:")
st.write("Please select the car features.")
mileage = st.number_input("Mileage", 100, 200000)
make = st.selectbox("Brand", data["Make"].unique())
model = st.selectbox("Model", data[data["Make"] == make]["Model"].unique())
trim = st.selectbox("Trim", data[(data["Make"] == make) & (data["Model"] == model)]["Trim"].unique())
car_type = st.selectbox("Type", data[(data["Make"] == make) & (data["Model"] == model) & (data["Trim"] == trim)]["Type"].unique())
cylinder = st.selectbox("Cylinder", data["Cylinder"].unique())
liter = st.number_input("Fuel Volume", 1, 10)
doors = st.selectbox("Doors", data['Doors'].unique())
cruise = st.radio("Speed Constant", [True, False])
sound = st.radio("Sound System", [True, False])
leather = st.radio("Leather", [True, False])
if st.button("Prediction"):
    pred = price(mileage, make, model, trim, car_type, cylinder, liter, doors, cruise, sound, leather)
    st.write("Price ($):", round(pred, 2))
