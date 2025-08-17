# --- Step 0: Import the Pandas Library ---
# We always start by importing the tools we need. For now, it's just pandas.
import pandas as pd

print("Pandas library imported successfully.")

# --- Step 1: Load the Dataset ---
# We define the path to our data and use pd.read_csv() to load it.
# This creates our DataFrame, which we'll call 'df'.
file_path = './data/bank_marketing.csv'
df = pd.read_csv(file_path)

print("\n--- Original Data (First 5 Rows) ---")
print(df.head())

print("\n--- Initial Data Info ---")
# Let's inspect the data types to see which columns are not numerical.
# 'object' means it's text.
df.info()


# --- Step 2: Prepare the Data for the Model ---
# Our model only understands numbers. We need to convert all text columns.

# A) Convert the target variable 'deposit'.
# This is a binary column ('yes'/'no'), which is perfect for mapping.
# We'll change 'yes' to 1 and 'no' to 0.
print("\nConverting 'deposit' column to 0s and 1s...")
df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

# B) Convert the categorical feature columns.
# Columns like 'job', 'marital', and 'education' have multiple text categories.
# The best way to handle this is 'One-Hot Encoding'.
# Pandas has a perfect function for this: pd.get_dummies().
# It creates new columns for each category and puts a 1 or 0 to indicate its presence.
# For example, a 'job_management' column will be 1 if the job is 'management', and 0 otherwise.
print("Converting categorical text columns into numerical format using one-hot encoding...")
df = pd.get_dummies(df, columns=['job', 'marital', 'education'], drop_first=True)


# --- Final Check ---
# Let's look at our DataFrame now. It should be completely numerical.
print("\n--- Data After Cleaning and Encoding (First 5 Rows) ---")
print(df.head())

print("\n--- Final Data Info ---")
# The .info() method should now show that all columns are numerical (int64, float64, or uint8).
df.info()

# --- Step 3: Define Features (X) and Target (y) ---
# The target is the single column we want to predict.
y = df['deposit']

# The features are all the other columns that we will use to make the prediction.
# We can get this by dropping the target column from our DataFrame.
X = df.drop('deposit', axis=1)

print("\n--- Features (X) ---")
print(X.head())
print("\n--- Target (y) ---")
print(y.head())


# --- Step 4: Split the Data into Training and Testing Sets ---
# We need to import the function from scikit-learn to do this.
from sklearn.model_selection import train_test_split

# We'll hold back 30% of the data for testing our model.
# 'random_state=42' ensures that we get the exact same split every time
# we run the code, which makes our results reproducible.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# It's good practice to check the size of our new sets.
print(f"\nData successfully split.")
print(f"Number of rows in the training set: {len(X_train)}")
print(f"Number of rows in the testing set: {len(X_test)}")



# A) Import the necessary tools from scikit-learn
from sklearn.tree import DecisionTreeClassifier # <-- This is our new model
from sklearn.metrics import classification_report # <-- Our tool for evaluation

# B) Create an instance of the model
# We create a "blank" Decision Tree.
# 'random_state=42' ensures that if there are any random elements in the tree-building
# process, they will be the same every time we run the code.
dt_model = DecisionTreeClassifier(random_state=42)
print("\nDecision Tree model created.")

# C) Train the model
# This is the "learning" step. The model will analyze the training data (X_train, y_train)
# to find the best series of "if/then" questions to separate the 'yes' (1) and 'no' (0) deposits.
print("Training the Decision Tree model...")
dt_model.fit(X_train, y_train)
print("Model training complete.")

# D) Make predictions on the unseen test data
# We ask our trained model to predict the outcome for the features in our test set.
dt_preds = dt_model.predict(X_test)

# E) Evaluate the model's performance
# We compare the model's predictions (dt_preds) with the actual answers (y_test)
# to see how well it did. The classification report gives us a detailed breakdown.
print("\n--- Decision Tree Model Evaluation ---")
print(classification_report(y_test, dt_preds, target_names=['No Deposit (0)', 'Deposit (1)']))
print("------------------------------------")

# --- Step 6: Train and Evaluate the Random Forest Model ---

# A) Import the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# B) Create an instance of the model
# 'n_estimators=100' means we will build a forest of 100 decision trees.
# 'random_state=42' ensures reproducibility.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
print("\nRandom Forest model created.")

# C) Train the model
# This process is more intensive. It's building 100 trees on random samples of the data.
print("Training the Random Forest model...")
rf_model.fit(X_train, y_train)
print("Model training complete.")

# D) Make predictions on the test data
rf_preds = rf_model.predict(X_test)

# E) Evaluate the Random Forest model
print("\n--- Random Forest Model Evaluation ---")
print(classification_report(y_test, rf_preds, target_names=['No Deposit (0)', 'Deposit (1)']))
print("--------------------------------------")
