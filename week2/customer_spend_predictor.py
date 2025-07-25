# --- Step 0: Imports ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- Step 1: Load and Explore the Data ---
file_path = './data/ecommerce_customers.csv'
df = pd.read_csv(file_path)

print("--- E-commerce Customer Data ---")
print(df.head())
print("\n--- Data Info ---")
df.info()

# Let's visualize the relationships in the data. A pairplot is perfect for this.
# It creates scatter plots for every pair of numerical columns.
print("\nCreating a pairplot to visualize data relationships... (Close the plot window to continue)")
# Look for the variable that has the clearest linear relationship with 'Yearly_Amount_Spent'.
sns.pairplot(df)
plt.show()

# --- Step 2: Define Features (X) and Target (y) ---
# Our target is what we want to predict.
target = 'Yearly_Amount_Spent'
# Our features are all the numerical columns except the target.
features = ['Avg_Session_Length', 'Time_on_App', 'Time_on_Website', 'Length_of_Membership']

X = df[features]
y = df[target]

# --- Step 3: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 5: Train the Linear Regression Model ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("\nModel training complete.")

# --- Step 6: Evaluate the Model ---
# Make predictions on the unseen test data
y_pred = model.predict(X_test_scaled)

# Calculate R-squared and MSE
r2_score = model.score(X_test_scaled, y_test)
mse = mean_squared_error(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"R-squared (R²) Score: {r2_score:.4f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")

# --- Step 7: Interpret the Coefficients ---
# Let's see what the model thinks is most important.
coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print("\n--- Model Coefficients ---")
print(coefficients.sort_values(by='Coefficient', ascending=False))
print("------------------------")

# --- Step 8: Visualize Predictions vs. Actuals ---
# A great way to see how well the model performed is to plot its predictions against the true values.
# If the model is perfect, all the dots will be on a straight 45-degree line.
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Yearly Spend (y_test)")
plt.ylabel("Predicted Yearly Spend (y_pred)")
plt.title("Actual vs. Predicted Yearly Spend")
# Add the 'perfect prediction' line for reference
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.grid(True)
plt.show()