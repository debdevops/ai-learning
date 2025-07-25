# --- Step 0: Import necessary libraries ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Steps 1-4: Load, Prepare, Split, and Train the Model (Same as before) ---

# 1. Load Data
file_path = './data/housing_prices.csv'
df = pd.read_csv(file_path)

# 2. Define Features (X) and Target (y)
X = df[['SquareFeet']]
y = df['Price']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and Train Model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model has been trained. Now evaluating on the unseen test data...")

# --- Step 5: Evaluate the Model (Quantitative) ---
# We use our trained model to make predictions on the 'X_test' data.
y_pred = model.predict(X_test)

# Now we compare the model's predictions (y_pred) with the actual prices (y_test).

# Metric 1: R-squared (R²) Score
# This tells us what percentage of the change in price is explained by the change in square feet.
# A score of 1.0 is a perfect fit. Our score is very high, which is great!
r2_score = model.score(X_test, y_test)
print(f"\n--- Quantitative Evaluation ---")
print(f"R-squared (R²) Score: {r2_score:.4f}")

# Metric 2: Mean Squared Error (MSE)
# This measures the average of the squares of the errors.
# In simple terms, a smaller number is better. It gives us an idea of how 'off' our predictions are.
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:,.2f}") # The :,.2f formats it with commas and 2 decimals.
print("-----------------------------")


# --- Step 6: Visualize the Results ---
print("\nVisualizing the model's fit on the test data...")

# Create a figure to plot on
plt.figure(figsize=(10, 6))

# 1. Plot the actual data points from the test set (the ground truth)
plt.scatter(X_test, y_test, color='blue', label='Actual Prices (Test Set)')

# 2. Plot the regression line that our model learned
# The X values are our test set square footage.
# The Y values are the prices our model PREDICTED for that footage.
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices (Regression Line)')

# Add labels and a title for clarity
plt.title('Linear Regression Model Fit', fontsize=16)
plt.xlabel('Size (Square Feet)', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend() # Shows the legend with the labels we defined
plt.grid(True)

# Display the plot in a pop-up window
plt.show()

print("\nEvaluation complete.")