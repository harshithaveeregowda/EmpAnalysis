import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with your file path or data loading method)
df = pd.read_excel('data1.xlsx')



# Prepare the features (X) and target (y)
X = df[['age', 'grade', 'time_employed', 'education_hours_per_year', 'total_sick_days_per_year', 'years_in_department', 'homeoffice_days_per_week']]  # Example features
y = df['sick_on_mon_more_six']  # Target variable (sick leave on Monday)

# Label encode the target variable (y) from 'yes'/'no' to 1/0
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode categorical variables (if any)
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the features (optional, but useful for some models like Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for positive class (1)

# --- 1. ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
st.pyplot()  # Streamlit's method to show plots

# --- 2. Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
st.pyplot()  # Streamlit's method to show plots

# --- 3. Predicted vs Actual Plot ---
# Create a DataFrame to compare the predicted vs actual values
comparison_df = pd.DataFrame({'Actual': label_encoder.inverse_transform(y_test), 'Predicted': label_encoder.inverse_transform(y_pred)})

# Plot the comparison
plt.figure(figsize=(10, 6))
sns.countplot(x='Actual', data=comparison_df, color='lightblue', label='Actual')
sns.countplot(x='Predicted', data=comparison_df, color='salmon', label='Predicted', alpha=0.7)
plt.title('Predicted vs Actual Sick Leave on Monday')
plt.xlabel('Sick Leave on Monday')
plt.ylabel('Count')
plt.legend()
st.pyplot()  # Streamlit's method to show plots
