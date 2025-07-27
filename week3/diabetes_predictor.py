# --- Step 0: Imports ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --- Step 1: Load and Prepare the Data ---
# We will use a famous dataset about diabetes. We'll only use two features for easy visualization.
print("Loading the PIMA Indians Diabetes Dataset...")
# This dataset is available online, pandas can load it directly from a URL.
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, header=None, names=column_names)

print("\n--- Dataset Head ---")
print(df.head())

# For this example, we'll use 'Glucose' and 'BMI' to predict the 'Outcome'.
# Outcome: 1 for has diabetes, 0 for does not.
features = ['Glucose', 'BMI']
target = 'Outcome'

X = df[features]
y = df[target]

# --- Step 2: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Step 3: Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 4: Train the Logistic Regression Model ---
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print("\nDiabetes prediction model has been trained.")

# --- Step 5: Evaluate the Model ---
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\n--- Confusion Matrix ---")
print("The model correctly identified:")
print(f" - {cm[1, 1]} people WITH diabetes (True Positives)")
print(f" - {cm[0, 0]} people WITHOUT diabetes (True Negatives)")
print("The model made mistakes on:")
print(f" - {cm[0, 1]} people, predicting they HAVE diabetes when they DON'T (False Positives)")
print(f" - {cm[1, 0]} people, predicting they DON'T have diabetes when they DO (False Negatives)")
print("------------------------")

# A nicer way to visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title("Confusion Matrix for Diabetes Prediction")
plt.show()

# --- Step 6: Visualize the Decision Boundary ---
# This is a powerful way to see how our model is making decisions.
print("\nVisualizing the model's decision boundary...")

# We need to create a "mesh grid" of points to plot the boundary
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# We use the model to predict on every single point in the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the actual data points from the training set
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Glucose (scaled)')
plt.ylabel('BMI (scaled)')
plt.show()