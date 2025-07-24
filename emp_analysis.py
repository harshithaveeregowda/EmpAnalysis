import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()
CLAUDE_URL   = os.getenv("CLAUDE_ENDPOINT_URL")
CLAUDE_TOKEN = os.getenv("CLAUDE_BEARER_TOKEN")
DATABRICKS_SERVER = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_PATH   = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN  = os.getenv("DATABRICKS_ACCESS_TOKEN")

def format_insights(raw: str) -> str:
    """
    - Finds the first line starting with '#' and uses it as the markdown H2 title.
    - Converts the remaining lines into bullet points.
    - Falls back to "🔍 Insights" if no heading is present.
    """
    lines = [l.rstrip() for l in raw.splitlines() if l.strip()]
    title = "🔍 Insights"
    bullets = []

    for line in lines:
        if line.startswith("#"):
            # strip leading '#' and whitespace
            title = line.lstrip("#").strip()
        else:
            # clean up numbering and extra spaces
            txt = line.lstrip("0123456789. ").strip()
            bullets.append(f"- {txt}")

    # build the final markdown
    md = [f"## {title}", ""]
    md += bullets
    return "\n".join(md)
    
# Title
st.title("Employee Sick Leave Risk Prediction + Behavior Clustering")

tabs = st.tabs(["Insights", "Ask your Data"])

# File Upload
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)

    # --- Tab 1: Insights ---
    with tabs[0]:
    
        

        # AI tips in an expander
        with st.expander("🔍 Automated Tips", expanded=True):
            if st.button("Generate General Tips", key="gen_tips_btn"):
                prompt = (
                f"Our KPIs are:\n"
                f"- €{df}\n"
                "Please provide 3 concise, prioritized tips to decrease sick leaves and improve employee engagement."
                )
                headers = {
                    "Authorization": f"Bearer {CLAUDE_TOKEN}",
                    "Content-Type": "application/json"
                }
                body = {"messages": [{"role": "user", "content": prompt}]}
    
                with st.spinner("Generating tips…"):
                    r = requests.post(CLAUDE_URL, json=body, headers=headers, timeout=120)
                    if r.status_code != 200:
                        st.error(f"Invocation failed with status {r.status_code}")
                        st.code(r.text, language="json")
                        st.stop()
                    raw = r.json()["choices"][0]["message"]["content"]
    
            # Format and render
                pretty = format_insights(raw)
                st.markdown(pretty)
    
        #----------------------------------------------------------------
        # Encode target
        df['sick_on_mon_more_six'] = df['sick_on_mon_more_six'].map({'yes': 1, 'no': 0})
    
        # Backup original for filtering
        original_df = df.copy()
    
        # Encode categorical features
        df_model = pd.get_dummies(df, drop_first=True)
    
        # Prepare features
        X = df_model.drop('sick_on_mon_more_six', axis=1)
        y = df_model['sick_on_mon_more_six']
    
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
        # Random Forest Classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
    
        # Predict probabilities
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
    
        # Add predictions to original data
        original_df['Predicted Sick'] = preds
        original_df['Risk Score (0-1)'] = probs
    
        # --- K-Means Clustering ---
        st.subheader("🔍 Employee Behavior Clusters (K-Means)")
    
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
        # Use PCA to reduce dimensions for plotting
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
    
        # KMeans Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_scaled)
        original_df['Cluster'] = clusters
    
        # Plot clusters
        fig_cluster, ax_cluster = plt.subplots()
        scatter = ax_cluster.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='Set2', s=60)
        ax_cluster.set_title("K-Means Clusters of Employees")
        ax_cluster.set_xlabel("PCA Component 1")
        ax_cluster.set_ylabel("PCA Component 2")
        legend1 = ax_cluster.legend(*scatter.legend_elements(), title="Cluster")
        ax_cluster.add_artist(legend1)
        st.pyplot(fig_cluster)
    
        # --- UI Filters ---
        st.sidebar.header("Filter Options")
    
        # Risk slider
        risk_threshold = st.sidebar.slider("Minimum Risk Score", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
        # Dropdowns
        selected_grade = st.sidebar.selectbox("Select Grade", ["All"] + sorted(original_df['grade'].unique().tolist()))
        selected_city = st.sidebar.selectbox("Select City Type", ["All"] + sorted(original_df['city_type'].unique().tolist()))
    
        # Apply filters
        filtered_df = original_df[original_df['Risk Score (0-1)'] >= risk_threshold]
        if selected_grade != "All":
            filtered_df = filtered_df[filtered_df['grade'] == selected_grade]
        if selected_city != "All":
            filtered_df = filtered_df[filtered_df['city_type'] == selected_city]
    
        # Display filtered data
        st.subheader("📋 Filtered Employees by Risk and Cluster")
        st.dataframe(
            filtered_df[['name', 'grade', 'age', 'amount', 'year', 'city_type', 'Risk Score (0-1)', 'Predicted Sick', 'Cluster']]
            .sort_values(by='Risk Score (0-1)', ascending=False)
            .style.apply(lambda row: ['background-color: salmon' if row['Risk Score (0-1)'] > 0.5 else '' for _ in row], axis=1)
        )
    
        # --- Feature Importance ---
        st.subheader("🎯 Feature Importance for Sick Leave Prediction")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=importances.index, palette="magma", ax=ax)
        ax.set_title("Top 10 Important Features")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)
    
        # Show average profile per cluster
        cluster_summary = original_df.groupby('Cluster')[['age', 'amount', 'Risk Score (0-1)']].mean()
        st.write("🧾 Average Characteristics by Cluster:")
        #st.dataframe(cluster_summary)
    
        # Cluster Profile Summary
        #st.subheader("🧾 Cluster Profile Summary")
        cluster_summary = original_df.groupby('Cluster')[['age', 'amount', 'Risk Score (0-1)']].mean()
        cluster_summary['Count'] = original_df.groupby('Cluster').size()
        st.dataframe(cluster_summary.style.highlight_max(axis=0))
    
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
        # Grade vs Amount
        st.subheader("💼 Grade vs Compensation")
        grade_amt = df.groupby('grade')['amount'].mean().reset_index()
        st.dataframe(grade_amt)
        # City type vs Sick Days
        st.subheader("🏙️ City Type vs Sick Days")
        city_sick = df.groupby('city_type')['total_sick_days_per_year'].mean().reset_index()
        st.dataframe(city_sick)
        # FTE vs Sick on Monday
        if 'sick_on_mon_more_six' in df.columns:
            st.subheader("🛌 FTE vs Sick on Monday > 6 Times")
            sick_ftes = df.groupby(['fte', 'sick_on_mon_more_six']).size().unstack(fill_value=0)
            st.dataframe(sick_ftes)
    
        #st.subheader("🔍 Top 5 Employees with Highest Education Hours")
        #st.dataframe(df.sort_values(by="education_hours_per_year", ascending=False).head(5))
    
        #st.subheader("🔍 Top 5 Employees with Lowest Education Hours")
        #st.dataframe(df.sort_values(by="education_hours_per_year", ascending=True).head(5))
    
        #st.subheader("🏙️ Grade Distribution by City Type")
        city_grade_counts = df.groupby(['city_type', 'grade']).size().unstack(fill_value=0)
        #st.dataframe(city_grade_counts)
    
        #st.subheader("👶👴 Youngest and Oldest Employee in Each Grade")
        youngest = df.loc[df.groupby("grade")["age"].idxmin()]
        oldest = df.loc[df.groupby("grade")["age"].idxmax()]
    
        #st.markdown("**Youngest by Grade**")
        #st.dataframe(youngest)
    
        #st.markdown("**Oldest by Grade**")
        #st.dataframe(oldest)
    
        #st.subheader("📊 Average Age by Grade")
        avg_age = df.groupby("grade")["age"].mean().reset_index().rename(columns={"age": "average_age"})
        #st.dataframe(avg_age)
    
        #st.subheader("⏳ Top 5 Employees with Highest Time Employed")
        #st.dataframe(df.sort_values(by="time_employed", ascending=False).head(5))
    
        #st.subheader("⏱️ Top 5 Employees with Lowest Time Employed")
        #st.dataframe(df.sort_values(by="time_employed", ascending=True).head(5))
    
        #st.subheader("💰 Top 5 Employees with Highest Amount")
        #st.dataframe(df.sort_values(by="amount", ascending=False).head(5))
    
        #st.subheader("💸 Top 5 Employees with Lowest Amount")
        #st.dataframe(df.sort_values(by="amount", ascending=True).head(5))
    
        #st.subheader("🤒 Employee with Maximum Total Sick Days Per Year")
        max_sick = df[df["total_sick_days_per_year"] == df["total_sick_days_per_year"].max()]
        #st.dataframe(max_sick)
    
        #st.subheader("💪 Employee with Minimum Total Sick Days Per Year")
        min_sick = df[df["total_sick_days_per_year"] == df["total_sick_days_per_year"].min()]
        #st.dataframe(min_sick)
    
        ### EDUCATION HOURS ###
        st.header("🎓 Education Hours Analysis")
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("Top 5 Highest")
            top_edu = df.sort_values(by="education_hours_per_year", ascending=False).head(5)
            st.dataframe(top_edu)
    
        with col2:
            st.subheader("Top 5 Lowest")
            low_edu = df.sort_values(by="education_hours_per_year", ascending=True).head(5)
            st.dataframe(low_edu)
    
        # Chart
        st.subheader("📊 Education Hours - Max & Min")
        fig, ax = plt.subplots()
        sns.barplot(data=pd.concat([top_edu, low_edu]), x="name", y="education_hours_per_year", hue="grade", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
        ### GRADE COUNT PER CITY ###
        st.header("🏙️ Grade Distribution by City Type")
    
        # Create a pivot table with grades as columns and city_type as rows
        grade_city = df.groupby(['city_type', 'grade']).size().unstack(fill_value=0)
    
        # Display the table
        st.dataframe(grade_city)
    
        # Plot: each grade in a separate bar (grouped bar chart)
        fig2, ax2 = plt.subplots()
        grade_city.plot(kind='bar', stacked=False, ax=ax2)  # Set stacked=False
        ax2.set_ylabel("Count")
        ax2.set_title("Grade Distribution by City Type")
        ax2.legend(title='Grade')
        st.pyplot(fig2)
    
        ### AGE ANALYSIS ###
        st.header("👶👴 Age Analysis by Grade")
    
        youngest = df.loc[df.groupby("grade")["age"].idxmin()]
        oldest = df.loc[df.groupby("grade")["age"].idxmax()]
        avg_age = df.groupby("grade")["age"].mean().reset_index().rename(columns={"age": "average_age"})
    
        st.subheader("Youngest in Each Grade")
        st.dataframe(youngest)
    
        st.subheader("Oldest in Each Grade")
        st.dataframe(oldest)
    
        st.subheader("Average Age by Grade")
        st.dataframe(avg_age)
    
        fig3, ax3 = plt.subplots()
        sns.barplot(data=avg_age, x="grade", y="average_age", ax=ax3)
        ax3.set_title("Average Age by Grade")
        st.pyplot(fig3)
    
        ### TIME EMPLOYED ###
        st.header("⏳ Time Employed Analysis")
    
        top_time = df.sort_values(by="time_employed", ascending=False).head(5)
        low_time = df.sort_values(by="time_employed", ascending=True).head(5)
    
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Top 5 Longest Employed")
            st.dataframe(top_time)
        with col4:
            st.subheader("Top 5 Shortest Employed")
            st.dataframe(low_time)
    
        fig4, ax4 = plt.subplots()
        sns.barplot(data=pd.concat([top_time, low_time]), x="name", y="time_employed", hue="grade", ax=ax4)
        ax4.set_title("Time Employed - Max & Min")
        plt.xticks(rotation=45)
        st.pyplot(fig4)
    
        ### AMOUNT ###
        st.header("💰 Compensation Analysis")
    
        top_amt = df.sort_values(by="amount", ascending=False).head(5)
        low_amt = df.sort_values(by="amount", ascending=True).head(5)
    
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("Top 5 Highest Amount")
            st.dataframe(top_amt)
        with col6:
            st.subheader("Top 5 Lowest Amount")
            st.dataframe(low_amt)
    
        fig5, ax5 = plt.subplots()
        sns.barplot(data=pd.concat([top_amt, low_amt]), x="name", y="amount", hue="grade", ax=ax5)
        ax5.set_title("Amount - Max & Min")
        plt.xticks(rotation=45)
        st.pyplot(fig5)
    
        ### SICK DAYS ###
        st.header("🤒 Total Sick Days Per Year")
    
        max_sick = df[df["total_sick_days_per_year"] == df["total_sick_days_per_year"].max()]
        min_sick = df[df["total_sick_days_per_year"] == df["total_sick_days_per_year"].min()]
    
        st.subheader("Max Sick Days")
        st.dataframe(max_sick)
    
        st.subheader("Min Sick Days")
        st.dataframe(min_sick)
    
        # Select top 5 employees with highest sick days
        top_sick = df.sort_values(by="total_sick_days_per_year", ascending=False).head(5)
    
        st.header("🤒 Top 5 Employees with Highest Total Sick Days Per Year")
        st.dataframe(top_sick)
    
        # Plot the chart
        fig6, ax6 = plt.subplots()
        sns.barplot(data=top_sick, x="name", y="total_sick_days_per_year", hue="grade", ax=ax6)
        ax6.set_title("Top 5 Employees by Sick Days")
        ax6.set_ylabel("Total Sick Days Per Year")
        plt.xticks(rotation=45)
        st.pyplot(fig6)
    
        # Lowest sick days
        low_sick = df.sort_values(by="total_sick_days_per_year", ascending=True).head(5)
    
        st.subheader("💪 Top 5 Employees with Lowest Total Sick Days Per Year")
        st.dataframe(low_sick)

    # --- Tab 2: Ask your data ---
    with tabs[1]:
        # Create two columns: left panel (width=1) & main chat area (width=4)
        panel_col, chat_col = st.columns([1, 4], gap="small")
    
        # ---- LEFT PANEL in panel_col ----
        with panel_col:
            st.markdown(
                """
                <div style="
                  background:#f0dfce;
                  padding:1rem;
                  height:calc(80vh - 2rem);
                  box-sizing:border-box;
                  border-radius:8px;
                ">
                  <h2 style="margin:0 0 .5rem 0; color:#333;">💬 AI Assistant</h2>
                  <p style="color:#222; line-height:1.4; font-size:0.9rem;">
                    Analyze employee sick leave data<br/>
                    and get actionable insights for decision-making.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
        # ---- CHAT AREA in chat_col ----
        with chat_col:
            st.markdown("## 💬 Ask the Data")
    
            # Initialize history
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Hi there! 👋\n\nI can help you explore your data. What would you like to ask?"
                    }
                ]
    
            # Render chat history
            chat_container = st.container()
            for msg in st.session_state.messages:
                chat_container.chat_message(msg["role"]).write(msg["content"])
    
            # Inject CSS for a fixed-footer input
            st.markdown(
                """
                <style>
                  .footer-input {
                    position: fixed;
                    bottom: 0;
                    left: 20%;   /* 1/(1+4) of the width */
                    width: 80%;  /* 4/(1+4) of the width */
                    padding: 1rem;
                    background: #f7f7f7;
                    box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
                    z-index: 1000;
                  }
                  /* ensure messages don't get hidden under the footer */
                  .block-container {
                    padding-bottom: 6rem;
                  }
                </style>
                """,
                unsafe_allow_html=True,
            )
    
            # Build a form that clears on submit
            footer = st.empty()
            with footer.container():
                st.markdown('<div class="footer-input">', unsafe_allow_html=True)
                with st.form(key="chat_form", clear_on_submit=True):
                    user_question = st.text_input(
                        "", 
                        placeholder="Ask me about employee sick leave…",
                        label_visibility="hidden"
                    )
                    submitted = st.form_submit_button("➤")
                st.markdown('</div>', unsafe_allow_html=True)
    
            # Handle new question
            if submitted and user_question:
                # Add & render the user’s message
                st.session_state.messages.append({"role": "user", "content": user_question})
                chat_container.chat_message("user").write(user_question)
    
                # Call your LLM exactly as before
                body = {
                    "messages": [
                        {"role": "system", "content": (
                            "You are an expert data analyst assistant. "
                            "Please format the answer as a Markdown bullet list, using '- ' at the start of each line, with one item per line."
                        )},
                        {"role": "assistant", "content": "Understood, here’s my answer:"},
                        {"role": "user", "content": (
                            f"Context:\n{df}\n\n"
                            f"Question: {user_question}\n\n"
                            "Please format the answer using clean bullet points, with each bullet on its own line. Start each bullet with '• '."
                        )},
                    ]
                }
                headers = {"Authorization": f"Bearer {CLAUDE_TOKEN}", "Content-Type": "application/json"}
                with st.spinner("Thinking…"):
                    r = requests.post(CLAUDE_URL, json=body, headers=headers, timeout=120)
    
                if r.status_code != 200:
                    st.error(f"Error {r.status_code}")
                    st.code(r.text, language="json")
                else:
                    reply = r.json()["choices"][0]["message"]["content"]
                    cleaned = "\n".join(
                        line for line in reply.splitlines()
                        if not (line.startswith("<<") and line.endswith(">>"))
                    )
    
                    # ✨ Format bullet points to render nicely
                    if "•" in cleaned:
                        bullet_points = cleaned.split("•")
                        formatted = "\n".join(f"• {bp.strip()}" for bp in bullet_points if bp.strip())
                    else:
                        formatted = cleaned
    
                    # Show response
                    st.session_state.messages.append({"role": "assistant", "content": formatted})
                    chat_container.chat_message("assistant").write(formatted)
