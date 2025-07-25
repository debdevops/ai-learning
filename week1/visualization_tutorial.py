# Import both pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. LOAD THE DATA (Same as before) ---
# We use the same skill to load our data into a DataFrame.
file_path = './data/housing_prices.csv'
df = pd.read_csv(file_path)

print("Data loaded successfully. Preparing to create a plot...")

# --- 2. PREPARE THE DATA FOR PLOTTING ---
# We'll put our X and Y axes into their own variables for clarity.
# The X-axis will be the size of the house.
x_axis_data = df['SquareFeet']
# The Y-axis will be the price of the house.
y_axis_data = df['Price']


# --- 3. CREATE THE PLOT ---
# plt.figure() creates a figure object, which is like the canvas for our plot.
# figsize=(10, 6) makes the plot 10 inches wide and 6 inches tall for better readability.
plt.figure(figsize=(10, 6))

# plt.scatter() is the function to create a scatter plot.
# We pass our X and Y data to it.
plt.scatter(x_axis_data, y_axis_data)

# --- 4. ADD LABELS AND A TITLE ---
# It's crucial to label your plots so others (and you!) can understand them.
plt.title('Relationship between House Size and Price', fontsize=16)
plt.xlabel('Size (Square Feet)', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.grid(True) # Adds a grid for easier reading

# --- 5. SHOW THE PLOT ---
# plt.show() displays the plot in a new pop-up window.
print("Displaying plot...")
plt.show()

print("Plot window closed.")