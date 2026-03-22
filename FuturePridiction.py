import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("🚀 AI Sales Prediction App")

#================= File Upload =============
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file is not None:

    # Load file
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("### 📊 Data Preview")
    st.write(df.head())

    # Cleaning
    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)

    # Total Sales
    if 'Total_Sales' not in df.columns:
        if 'Price' in df.columns and 'Quantity' in df.columns:
            df['Total_Sales'] = df['Quantity'] * df['Price']

    #================ ANALYSIS ==============
    if 'City' in df.columns:
        st.write("## 📍 City Sales")
        city_sales = df.groupby('City')['Total_Sales'].sum()
        st.bar_chart(city_sales)

    if 'Product' in df.columns:
        st.write("## 🛒 Product Sales")
        product_sales = df.groupby('Product')['Total_Sales'].sum()
        st.bar_chart(product_sales)

    #================ MODEL =================
    if 'Price' in df.columns and 'Quantity' in df.columns:

        X = df[['Price', 'Quantity']]
        y = df['Total_Sales']

        X_train , X_test, y_train , y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        score = r2_score(y_test, model.predict(X_test))

        st.write("### 🤖 Model Score")
        st.success(f"Accuracy: {round(score,2)}")

        #============= PREDICTION ============
        st.write("### 🔮 Sales Prediction")

        price = st.number_input("Enter Price", value=10)
        quantity = st.number_input("Enter Quantity", value=5)

        if st.button("Predict"):
            pred = model.predict([[price, quantity]])
            st.success(f"💰 Predicted Sales: {round(pred[0],2)}")

    else:
        st.error("❌ Required columns missing: Price / Quantity")