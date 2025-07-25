# --- Step 0: Import necessary libraries ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Libraries imported successfully.")

# --- Step 1: Load the Data ---
# We must load the data FIRST to create the 'df' DataFrame.
file_path = './data/housing_prices.csv'
df = pd.read_csv(file_path)
print("\n--- Original Data (First 5 Rows) ---")
print(df.head())

# --- Step 2: Define Features (X) and Target (y) ---
# Now that 'df' exists, we can use it.
# We are preparing the data for the y = mx + c formula.
X = df[['SquareFeet']] # This is our 'x'
y = df['Price']        # This is our 'y'

print("\nDefining features (X) and target (y)...")

# --- Step 3: Split Data into Training and Testing Sets ---
# We create flashcards. The training set is for learning, the test set is for the exam.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nData split into training and testing sets.")
print(f"Training set size: {len(X_train)} houses")
print(f"Testing set size: {len(X_test)} houses")

# --- Step 4: Create and Train the Linear Regression Model ---
# 1. Create a blank model instance.
model = LinearRegression()

# 2. 'fit' is the training step. The model learns the relationship.
print("\nTraining the model on the training data...")
model.fit(X_train, y_train)
print("Model training complete.")

# Let's inspect what the model learned.
# .coef_ holds the slope 'm' and .intercept_ holds the intercept 'c'.
print("\n--- Model Learned Parameters ---")
print(f"Slope (m): {model.coef_[0]:.2f}")
print(f"Intercept (c): {model.intercept_:.2f}")
print("This means the learned formula is: Price = 144.35 * SquareFeet + 29339.29")
print("--------------------------------")

# --- Step 5: Make a Prediction on New Data ---
# Now we use the learned formula: y = mx + c
new_house_size = 2500
new_data_point = [[new_house_size]] 

predicted_price = model.predict(new_data_point)

print(f"\n--- Prediction Result ---")
print(f"Predicted price for a {new_house_size} sq ft house is: ${predicted_price[0]:.2f}")
print("-------------------------")