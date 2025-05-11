import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import numpy as np

# -----------------------------------
# Step 1: Data Creation
# -----------------------------------
data = {
    'name': ['Person_0', 'Person_1', 'Person_2', 'Person_3', 'Person_4', 'Person_5', 'Person_6'],
    'grade': ['Senior', 'Director', 'Senior', 'Senior', 'Director', 'Director', 'Senior'],
    'age': [20, 48, 57, 54, 53, 41, 48],
    'time_employed': [28, 20, 47, 7, 11, 20, 16],
    'party_hometown_on_sun': ['no'] * 7,
    'bridge_days_on_tue': ['no'] * 7,
    'amount': [2684, 4950, 4550, 5792, 3471, 4941, 3599],
    'education_hours_per_year': [21, 20, 25, 39, 20, 24, 40],
    'total_sick_days_per_year': [17, 28, 21, 17, 13, 29, 23],
    'years_in_department': [5, 4, 5, 6, 6, 8, 7],
    'city_type': ['city', 'suburb', 'suburb', 'suburb', 'suburb', 'city', 'suburb'],
    'homeoffice_days_per_week': [3, 2, 4, 3, 3, 3, 3],
    'fte': [1, 0.5, 0.5, 1, 1, 1, 0.5],
    'year': [2024] * 7,
    'sick_on_mon_more_six': ['no', 'yes', 'no', 'no', 'no', 'no', 'no']
}
df = pd.DataFrame(data)

# -----------------------------------
# Step 2: Preprocessing - Encode Categorical Variables
# -----------------------------------
# We encode all categorical fields (excluding 'name' which is an identifier)
label_cols = df.select_dtypes(include='object').columns.drop('name')
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Optionally drop 'name' if it's not needed for the analysis
df = df.drop(columns=['name'])

# -----------------------------------
# Step 3: Exploratory Data Analysis (EDA)
# -----------------------------------
# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of All Features")
plt.show()

# Pair Plot by 'sick_on_mon_more_six'
sns.pairplot(df, hue="sick_on_mon_more_six")
plt.suptitle("Pairwise Plot Grouped by Sick on Monday", y=1.02)
plt.show()

# -----------------------------------
# Step 4: Predictive Modeling
# A. Classification: Predicting 'sick_on_mon_more_six'
# -----------------------------------
X_cls = df.drop(columns=['sick_on_mon_more_six'])
y_cls = df['sick_on_mon_more_six']

# Using RandomForestClassifier for this demonstration
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_cls, y_train_cls)
y_pred_cls = clf.predict(X_test_cls)

print("=== Classification Report ===")
print(classification_report(y_test_cls, y_pred_cls))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test_cls, y_pred_cls))

# Feature Importance from Classification
importances_cls = pd.Series(clf.feature_importances_, index=X_cls.columns)
importances_cls.sort_values().plot(kind='barh', title="Feature Importance for 'sick_on_mon_more_six'")
plt.show()

# -----------------------------------
# B. Regression: Predicting 'amount'
# -----------------------------------
X_reg = df.drop(columns=['amount'])
y_reg = df['amount']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

print("=== Regression Metrics ===")
print("R²:", r2_score(y_test_reg, y_pred_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))

# Feature Importance from Regression
importances_reg = pd.Series(reg.feature_importances_, index=X_reg.columns)
importances_reg.sort_values().plot(kind='barh', title="Feature Importance for 'amount'")
plt.show()

# -----------------------------------
# Step 5: Summarize Key Findings
# -----------------------------------
print("\nKey Insights:")
top_sick_features = importances_cls.sort_values(ascending=False).head(3)
print("Top predictors for 'sick_on_mon_more_six':\n", top_sick_features)

top_amount_features = importances_reg.sort_values(ascending=False).head(3)
print("\nTop predictors for 'amount':\n", top_amount_features)
