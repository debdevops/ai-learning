# --- Imports ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # <-- New import for text processing
from sklearn.naive_bayes import MultinomialNB # <-- A new, simple model great for text
from sklearn.metrics import classification_report

print("Libraries imported successfully.")

# --- TASK 1: Load and Prepare the Data ---
# A) Load the 'spam_dataset.csv' file into a DataFrame called 'df'.
# The file is in the './data/' directory.
# Your code here
df = pd.read_csv("./data/spam_dataset.csv")

# B) The 'label' column is our target. Map 'ham' to 0 and 'spam' to 1.
# Your code here
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("\n--- Data Head ---")
print(df.head())


# --- TASK 2: Define Features (X) and Target (y) ---
# The target 'y' is the numerical 'label' column.
# The features 'X' is the 'message' column containing the text.
# Your code here
X = df['message']
y = df['label']


# --- TASK 3: Split the Data ---
# Split X and y into training and testing sets.
# Use a test_size of 0.25 and a random_state of 42.
# Your code here
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\nData split into {len(X_train)} training and {len(X_test)} testing messages.")


# --- TASK 4: Vectorize the Text Data ---
# This is the most important new step. We convert text to numbers.
# A) Create an instance of CountVectorizer. Call it 'vectorizer'.
# Your code here
vectorizer = CountVectorizer()

# B) 'fit' the vectorizer on the TRAINING text to build its vocabulary,
# then 'transform' the training text into a numerical matrix.
# Use the .fit_transform() method. Store the result in 'X_train_vectorized'.
# Your code here
X_train_vectorized = vectorizer.fit_transform(X_train)

# C) Transform the TESTING text using the SAME fitted vectorizer.
# Use the .transform() method. Store the result in 'X_test_vectorized'.
# Your code here
X_test_vectorized = vectorizer.transform(X_test)

print("\nText data has been vectorized into a numerical format.")


# --- TASK 5: Train and Evaluate the Model ---
# We'll use 'Multinomial Naive Bayes', a classifier that's fast and effective for text.

# A) Create an instance of the MultinomialNB classifier. Call it 'model'.
# Your code here
model = MultinomialNB()

# B) Train the model on the VECTORIZED training data.
# Your code here
model.fit(X_train_vectorized, y_train)

# C) Make predictions on the VECTORIZED test data.
# Your code here
y_pred = model.predict(X_test_vectorized)

print("\n--- Spam Detection Model Evaluation ---")
# D) Print the classification report.
# Your code here
print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))


# --- TASK 6: Predict on New, Unseen Messages ---
print("\n--- Testing on New Messages ---")
new_messages = [
    "Hey, are you coming to the meeting this afternoon?",
    "URGENT! You have won a 1,000,000 prize. CALL 090909090 to claim NOW!"
]

# A) Vectorize our new messages using the SAME fitted vectorizer.
# Your code here
new_messages_vectorized = vectorizer.transform(new_messages)

# B) Predict on the vectorized new messages.
# Your code here
new_predictions = model.predict(new_messages_vectorized)

for message, prediction in zip(new_messages, new_predictions):
    label = "Spam" if prediction == 1 else "Ham"
    print(f"Message: '{message}'\nPredicted: {label}\n")