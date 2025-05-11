import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Employee Insight Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Drop "name" if exists, for analysis
    df_for_analysis = df.drop(columns=['name'], errors='ignore')

    # Encode categorical columns
    df_encoded = df_for_analysis.copy()
    cat_cols = df_encoded.select_dtypes(include='object').columns
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

    # Summary stats
    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe())

    # Correlation heatmap
    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Grade vs Amount
    st.subheader("💼 Grade vs Compensation")
    grade_amt = df.groupby('grade')['amount'].mean().reset_index()
    st.dataframe(grade_amt)

    fig1, ax1 = plt.subplots()
    sns.barplot(data=grade_amt, x='grade', y='amount', ax=ax1)
    ax1.set_title("Average Compensation by Grade")
    st.pyplot(fig1)

    # City type vs Sick Days
    st.subheader("🏙️ City Type vs Sick Days")
    city_sick = df.groupby('city_type')['total_sick_days_per_year'].mean().reset_index()
    st.dataframe(city_sick)

    fig2, ax2 = plt.subplots()
    sns.barplot(data=city_sick, x='city_type', y='total_sick_days_per_year', ax=ax2)
    ax2.set_title("Avg Sick Days by City Type")
    st.pyplot(fig2)

    # FTE vs Sick on Monday
    if 'sick_on_mon_more_six' in df.columns:
        st.subheader("🛌 FTE vs Sick on Monday > 6 Times")
        sick_ftes = df.groupby(['fte', 'sick_on_mon_more_six']).size().unstack(fill_value=0)
        st.dataframe(sick_ftes)

        fig3, ax3 = plt.subplots()
        sick_ftes.plot(kind='bar', stacked=True, ax=ax3)
        ax3.set_ylabel("Count")
        ax3.set_title("FTE and Monday Sickness")
        st.pyplot(fig3)

    # Education hours vs Amount
    st.subheader("🎓 Education Hours vs Compensation")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df, x='education_hours_per_year', y='amount', hue='grade', ax=ax4)
    ax4.set_title("Compensation vs Education Hours")
    st.pyplot(fig4)

else:
    st.info("👆 Upload a CSV file to begin analysis.")


