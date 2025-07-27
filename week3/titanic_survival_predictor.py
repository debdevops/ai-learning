# --- Step 0: Imports ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report # <-- New report import

# --- Step 1: Load the Data ---
print("Loading the Titanic dataset from Seaborn...")
# Seaborn has a built-in function to load this classic dataset
df = sns.load_dataset('titanic')

# --- Step 2: Exploratory Data Analysis & Data Cleaning ---
print("\n--- Original Data Info ---")
df.info()

# NEW STEP: We need to handle non-numerical data and missing values.
# Let's select the features we want to use.
# Pclass (Passenger Class), Sex, Age, Fare
# Our target is 'survived' (0 = No, 1 = Yes)

features_to_use = ['pclass', 'sex', 'age', 'fare']
target = 'survived'

# Create a new DataFrame with only the data we need
df_clean = df[features_to_use + [target]].copy()

# A) Handle Categorical Data: Convert 'sex' to numbers
# We map 'male' to 0 and 'female' to 1.
df_clean['sex'] = df_clean['sex'].map({'male': 0, 'female': 1})
print("\nConverted 'sex' column to numbers (0 for male, 1 for female).")

# B) Handle Missing Data: Fill missing 'age' values
# A simple strategy is to fill missing ages with the average age of all passengers.
average_age = df_clean['age'].mean()
df_clean['age'].fillna(average_age, inplace=True)
print(f"Filled missing 'age' values with the average age: {average_age:.2f}")

# C) Drop any other rows that might still have missing data
df_clean.dropna(inplace=True)

print("\n--- Cleaned Data Info ---")
df_clean.info()


# --- Step 3: Define Final Features (X) and Target (y) ---
X = df_clean[features_to_use]
y = df_clean[target]

# --- Step 4: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Step 5: Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 6: Train the Logistic Regression Model ---
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print("\nTitanic survival prediction model has been trained.")

# --- Step 7: Evaluate the Model ---
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# NEW STEP: Use classification_report for a detailed breakdown.
# This automatically calculates Precision, Recall, and F1-Score for each class!
print("\n--- Classification Report ---")
# 'target_names' makes the report more readable.
print(classification_report(y_test, y_pred, target_names=['Did not Survive (0)', 'Survived (1)']))
print("-----------------------------")


# --- Step 8: Interpret the Coefficients ---
# Let's see what features the model found most important for survival.
coefficients = pd.DataFrame(model.coef_[0], X.columns, columns=['Coefficient'])
print("\n--- Model Coefficients (Importance of each feature) ---")
print(coefficients.sort_values(by='Coefficient', ascending=False))