import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import numpy as np

st.set_page_config(page_title="Employee Analytics", layout="wide")

st.title("📊 Employee Data Dashboard")
st.markdown("This dashboard allows you to explore employee data, perform classification and regression analysis, and view key trends.")

# Load data
@st.cache_data
def load_data():
    #return pd.read_csv("data.csv")
    return pd.read_excel("data1.xlsx")

df = load_data()

# Display raw data
with st.expander("🔍 View Raw Data"):
    st.dataframe(df)

# Encode categorical variables
df_encoded = df.copy()
label_cols = ['grade', 'party_hometown_on_sun', 'bridge_days_on_tue', 'city_type', 'sick_on_mon_more_six']
for col in label_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Filter for employees sick on Mondays more than 6 times
st.subheader("👩‍⚕️ Employees Frequently Sick on Mondays")
# Filter the employees
df_sick_mon = df[df["sick_on_mon_more_six"] == "yes"]

# Check if there are any
if df_sick_mon.empty:
        st.info("✅ No employees have been sick on Mondays more than 6 times.")
else:
        # Show only selected columns for clarity
        st.dataframe(df_sick_mon[[
            "name", "grade", "age", "year", "city_type"
        ]])


if not df_sick_mon.empty:
    st.subheader("🎯 Sick-on-Monday Distribution by Grade")
    fig_pie = px.pie(df_sick_mon, names="grade", title="Grades of Employees Frequently Sick on Mondays")
    st.plotly_chart(fig_pie)

st.subheader("🧾 Top 10 Least Employed Employees per Year")

# Ensure 'time_employed' is numeric (in case it's read as object)
df["time_employed"] = pd.to_numeric(df["time_employed"], errors="coerce")

# Drop rows where year or time_employed is missing
df_yearly = df.dropna(subset=["year", "time_employed"])

# Get top 10 per year (least employed)
top10_least_employed = (
    df_yearly.sort_values(["year", "time_employed"])
             .groupby("year")
             .head(10)
             .reset_index(drop=True)
)

# Display the result
st.dataframe(top10_least_employed[["year", "name", "grade", "time_employed", "age", "amount"]])
#---------------------------
st.subheader("🧾 Top 10 Least Employed Employees")

# Ensure numeric types are handled correctly
df["time_employed"] = pd.to_numeric(df["time_employed"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Dropdown for user to select sorting method
sort_option = st.selectbox(
    "Sort top 10 least employed employees by:",
    ["year", "age", "grade", "amount"]
)

# Sort the dataframe accordingly
df_sorted = df.sort_values(by=["time_employed", sort_option], ascending=[True, True])

# Get top 10 (least employed) after sorting
top10_custom_sorted = df_sorted.head(10)

# Display table
st.dataframe(top10_custom_sorted[["name", "grade", "age", "amount", "year", "time_employed"]])

# -------- CLASSIFICATION --------
st.header("🧠 Classification: Will an Employee Be Sick on Monday More Than 6 Times?")
X_cls = df_encoded.drop(['name', 'sick_on_mon_more_six'], axis=1)
y_cls = df_encoded['sick_on_mon_more_six']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)

unique_classes = yc_train.nunique()
if unique_classes < 2:
    st.warning("🚫 Not enough class variety in the training data to perform classification.")
else:
    cls_model = LogisticRegression()
    cls_model.fit(Xc_train, yc_train)
    yc_pred = cls_model.predict(Xc_test)

    # Classification results
    st.subheader("📋 Classification Report")
    st.text(classification_report(yc_test, yc_pred))

    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(yc_test, yc_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

# -------- REGRESSION --------
st.header("📈 Regression: Predict Expense Amount")
X_reg = df_encoded.drop(['name', 'amount'], axis=1)
y_reg = df_encoded['amount']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

reg_model = LinearRegression()
reg_model.fit(Xr_train, yr_train)
yr_pred = reg_model.predict(Xr_test)

st.subheader("🔢 Regression Metrics")
st.write(f"**Mean Squared Error:** {mean_squared_error(yr_test, yr_pred):.2f}")

# Coefficient chart
st.subheader("📌 Feature Importance for Predicting Amount")
coef_df = pd.DataFrame({
    'Feature': X_reg.columns,
    'Coefficient': reg_model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

st.bar_chart(coef_df.set_index("Feature"))

# -------- VISUALIZATION --------
st.header("📊 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Amount vs. Total Sick Days")
    fig1 = px.scatter(df, x="total_sick_days_per_year", y="amount", color="grade",
                      size="education_hours_per_year", hover_name="name", title="Spending vs Sick Days")
    st.plotly_chart(fig1)

with col2:
    st.subheader("📚 Education Hours per Grade")
    fig2 = px.box(df, x="grade", y="education_hours_per_year", color="grade", title="Education Hours by Grade")
    st.plotly_chart(fig2)

st.subheader("🏠 Home Office Days by City Type")
fig3 = px.violin(df, x="city_type", y="homeoffice_days_per_week", color="city_type", box=True, points="all")
st.plotly_chart(fig3)

st.subheader("🧪 Sick Days Distribution")
fig4, ax4 = plt.subplots()
sns.histplot(df['total_sick_days_per_year'], kde=True, bins=10, ax=ax4)
st.pyplot(fig4)

st.success("✅ Analysis Complete")
