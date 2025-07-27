# --- Step 0: Imports ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # <-- New Model
from sklearn.metrics import accuracy_score, confusion_matrix # <-- New Metrics

# --- Step 1: Load and Prepare Data ---
file_path = './data/social_network_ads.csv'
df = pd.read_csv(file_path)

# Define Features (X) and Target (y)
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased'] # Target is 0 (Not Purchased) or 1 (Purchased)

# --- Step 2: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Step 3: Feature Scaling ---
# Scaling is just as important for Logistic Regression.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 4: Train the Logistic Regression Model ---
# We instantiate the LogisticRegression model instead of the LinearRegression model.
model = LogisticRegression()
print("Training the classification model...")
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- Step 5: Make Predictions ---
# .predict() gives the final class (0 or 1) based on the 0.5 threshold.
y_pred = model.predict(X_test_scaled)

# .predict_proba() shows the raw probability for each class.
# This helps us understand the model's confidence.
y_pred_proba = model.predict_proba(X_test_scaled)

# --- Step 6: Evaluate the Model ---
# We use our new classification metrics.

# Metric 1: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Metric 2: Confusion Matrix
# This table shows us where the model got confused.
# [[True Negatives, False Positives],
#  [False Negatives, True Positives]]
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("------------------------")

# --- Step 7: Inspect a Prediction ---
# Let's look at the first person in our test set.
print("\n--- Example Prediction ---")
actual_result = y_test.iloc[0]
predicted_result = y_pred[0]
probabilities = y_pred_proba[0]

print(f"Test Person's Actual Result: {'Purchased' if actual_result == 1 else 'Did Not Purchase'}")
print(f"Model's Prediction: {'Purchased' if predicted_result == 1 else 'Did Not Purchase'}")
print(f"Model's Raw Probabilities: [Prob(Not Purchase)={probabilities[0]:.4f}, Prob(Purchase)={probabilities[1]:.4f}]")