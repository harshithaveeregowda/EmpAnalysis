import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit app title
st.title("📊 Employee Classification with Multiple Models")

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

    # --- Preprocess the Data ---
    # Convert `amount` to numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Drop rows with missing values in critical columns for classification
    df = df.dropna(subset=["sick_on_mon_more_six", "grade", "city_type"])

    # --- Feature Engineering ---
    # Drop non-predictive columns
    features_to_drop = ["name", "amount", "year"]  # Drop name, amount, and year columns for classification
    X = df.drop(columns=features_to_drop)

    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Prepare classification targets
    targets = {
        "Sick on Monday": "sick_on_mon_more_six",
        "Employee Grade": "grade",
        "City Type": "city_type"
    }

    # --- Model Training and Prediction ---
    models = {
        #"Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        #"SVM": SVC(),
        #"XGBoost": xgb.XGBClassifier(eval_metric='mlogloss')
    }

    for target_name, target_column in targets.items():
        st.markdown(f"### {target_name} Classification")

        # Prepare the target variable
        y = df[target_column]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Iterate over all models and evaluate them
        for model_name, model in models.items():
            st.markdown(f"#### {model_name} - Model Results")

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate accuracy and classification report
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            st.write(f"**Accuracy**: {accuracy:.2f}")
            st.text(f"**Classification Report**:\n{report}")

            # Confusion Matrix
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_test, y_pred)

            # Plot Confusion Matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f"Confusion Matrix for {model_name}")
            st.pyplot(fig)

        st.write("---")
else:
    st.write("Please upload a file to begin.")
