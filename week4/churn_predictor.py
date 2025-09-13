# --- Imports ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("Libraries imported successfully.")

# --- TASK 1: Load and Prepare the Data ---
# A) Load the 'telecom_churn.csv' file
df = pd.read_csv("./data/telecom_churn.csv")

# B) Clean 'TotalCharges' column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
print("\nCleaned 'TotalCharges' column.")

# C) Convert the target variable 'Churn' to numbers
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# --- THE FIX ---
# D) CRITICAL: Drop any rows where the 'Churn' value might be missing after the mapping.
# If a value in the original 'Churn' column was not 'Yes' or 'No', the .map() function
# would have turned it into NaN. We must remove these rows.
df.dropna(subset=['Churn'], inplace=True)
# We also need to convert the Churn column to an integer type now that NaNs are gone.
df['Churn'] = df['Churn'].astype(int)
print("Cleaned 'Churn' column and removed any rows with missing target values.")


# E) Identify and one-hot encode categorical features
categorical_features = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

print("Data has been cleaned and encoded.")
print("\n--- Data Head After Cleaning ---")
print(df.head())


# --- TASK 2: Define Features (X) and Target (y) ---
X = df.drop('Churn', axis=1)
y = df['Churn']


# --- TASK 3: Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\nData split into {len(X_train)} training and {len(X_test)} testing samples.")


# --- TASK 4: Train and Evaluate the Random Forest Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # This line will now work
y_pred = model.predict(X_test)

print("\n--- Churn Prediction Model Evaluation ---")
print(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))


# --- TASK 5: Find the Most Important Features ---
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print("\n--- Top 5 Most Important Features for Predicting Churn ---")
print(feature_importances.nlargest(5))