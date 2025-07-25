# --- Step 0: Import libraries ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler # <-- New Import for Feature Scaling

# --- Step 1: Load Data ---
file_path = './data/housing_prices_multi.csv'
df = pd.read_csv(file_path)
print("--- Data with Multiple Features ---")
print(df.head())

# --- Step 2: Define Features (X) and Target (y) ---
# X now includes all the columns we want to use for prediction.
features = ['SquareFeet', 'Bedrooms', 'Age']
X = df[features]
y = df['Price']

# --- Step 3: Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Feature Scaling (The New, Critical Step) ---
# We create a scaler object.
scaler = StandardScaler()

# We 'fit' the scaler on the TRAINING data only. This learns the mean and standard deviation.
# Then we 'transform' the training data to be scaled.
X_train_scaled = scaler.fit_transform(X_train)

# We use the SAME scaler (already fitted) to transform the TESTING data.
# We do NOT fit it again on the test data. This prevents "data leakage".
X_test_scaled = scaler.transform(X_test)

print("\nData has been scaled. The model will now train on this scaled data.")

# --- Step 5: Create and Train the Model ---
# The model creation is the same, but we train it on the SCALED data.
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Multivariate model training complete.")

# --- Step 6: Make a Prediction on a New House ---
# Let's predict the price for a house with:
# SquareFeet: 2500, Bedrooms: 4, Age: 5 years
new_house_data = [[2500, 4, 5]]

# CRITICAL: We must scale our new data point using the same scaler we used for training.
new_house_scaled = scaler.transform(new_house_data)

# Make the prediction using the scaled data
predicted_price = model.predict(new_house_scaled)

print(f"\n--- Prediction Result ---")
print(f"Features of the new house: {new_house_data[0]}")
print(f"Predicted price is: ${predicted_price[0]:.2f}")
print("-------------------------")

# Optional: You can inspect the coefficients to see what the model learned.
# Each coefficient corresponds to one of our features in the 'features' list.
print("\nModel Coefficients (Importance of each feature):")
for i, feature in enumerate(features):
    print(f" - {feature}: {model.coef_[i]:.2f}")