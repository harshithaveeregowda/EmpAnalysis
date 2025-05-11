import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit app title
st.title("💰 Predicting Amount and Feature Importance")

# --- File Upload for Data ---
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data based on file type
    if uploaded_file.name.endswith('csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('xlsx'):
        df = pd.read_excel(uploaded_file)

    st.write("Dataset Preview:")
    st.dataframe(df.head())  # Show first few rows of the dataset

    # --- Data Preprocessing ---
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Drop rows with missing 'amount' values
    df_reg = df.dropna(subset=["amount"])

    # Select features for prediction (drop identifiers like 'name' and target 'amount')
    features_to_drop = ["name", "amount", "sick_on_mon_more_six"]
    X = df_reg.drop(columns=features_to_drop)
    y = df_reg["amount"]

    # Convert categorical variables to dummy variables (one-hot encoding)
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Regressor model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = model.predict(X_test)

    # --- 📉 Regression Metrics ---
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Display regression metrics
    st.markdown("### 🔍 Regression Metrics for Amount Prediction")
    st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
    st.write(f"**MSE (Mean Squared Error):** {mse:.2f}")
    st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # --- 🎯 Feature Importance ---
    importances = model.feature_importances_

    # Create a DataFrame for the feature importances
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Display feature importance
    st.markdown("### 🎯 Feature Importance for Predicting Amount")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
    ax.set_title("Most Influential Features for Amount Prediction")
    st.pyplot(fig)
else:
    st.write("Please upload a file to begin.")
