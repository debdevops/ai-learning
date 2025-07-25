import pandas as pd

# --- 1. LOADING DATA ---
# We specify the relative path to our data file.
# The './' means 'start in the current folder' (which is 'week1').
file_path = './data/housing_prices.csv'
print(f"Loading data from: {file_path}")

# pd.read_csv() is a powerful function to read tabular data into a DataFrame.
# A DataFrame is the primary Pandas data structure, like a spreadsheet or SQL table.
df = pd.read_csv(file_path)


# --- 2. INSPECTING THE DATA ---
print("\n--- 2.1 First 5 Rows (.head()) ---")
# .head() is the best way to quickly see the first few rows of your data.
print(df.head())

print("\n--- 2.2 Data Types and Info (.info()) ---")
# .info() gives a technical summary: column names, number of non-null values, and data types.
# This is great for spotting missing data.
df.info()

print("\n--- 2.3 Descriptive Statistics (.describe()) ---")
# .describe() provides key statistical data for the numerical columns (mean, std dev, min, max, etc.).
# This gives you a great high-level understanding of the data's distribution.
print(df.describe())


# --- 3. SELECTING DATA (COLUMNS) ---
print("\n--- 3. Selecting a single column (a 'Series') ---")
# You can select a single column by using its name in square brackets.
# A single column in Pandas is called a 'Series'.
prices = df['Price']
print(prices.head())


# --- 4. FILTERING DATA ---
print("\n--- 4. Filtering for houses with more than 4 bedrooms ---")
# This is one of the most powerful features of Pandas.
# 1. We create a boolean condition: df['Bedrooms'] > 4
# 2. We pass this condition back into the DataFrame.
# The result is a new DataFrame containing only the rows where the condition is True.
large_houses = df[df['Bedrooms'] > 4]
print(large_houses)

print("\n--- 4.1 Filtering for houses that cost less than $300,000 ---")
affordable_houses = df[df['Price'] < 300000]
print(affordable_houses)