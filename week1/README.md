# Week 1: The Data Scientist's Toolkit

### Objective
Before we can build any machine learning models, we must first learn how to speak the language of data. This week is dedicated to mastering the foundational Python libraries that allow us to load, manipulate, and visualize datasets. This is the absolute prerequisite for all subsequent weeks.

### Key Concepts

1.  **Data Structures for Machine Learning:**
    *   **NumPy Arrays:** The bedrock of numerical computing in Python. They are fast, memory-efficient arrays that form the basis for all other data science libraries.
    *   **Pandas DataFrame:** A 2-dimensional labeled data structure, like a spreadsheet or a SQL table. It is the single most important tool for handling real-world, structured data. Each column in a DataFrame is a **Series**.

2.  **Exploratory Data Analysis (EDA):**
    *   This is the critical first step in any data project. It involves inspecting the data to understand its structure, find patterns, spot anomalies, and check assumptions.
    *   **Inspection:** Using functions like `.head()`, `.info()`, and `.describe()` to get a high-level summary of the dataset.
    *   **Visualization:** Using plots to "see" the data. A visual representation can reveal patterns that are impossible to see in a table of numbers. The most fundamental plot for regression is the **scatter plot**, which helps us visualize the relationship between two variables.

### Key Libraries
- **Pandas**: For loading and manipulating tabular data (`pd.read_csv`, `df['column']`).
- **Matplotlib**: The foundational library for creating static, animated, and interactive visualizations in Python (`plt.scatter`, `plt.show`).

### Files in this folder
- `README.md`: This file, outlining the core concepts.
- `data/housing_prices.csv`: A simple dataset of house prices for our practice.
- `pandas_tutorial.py`: A script demonstrating how to load and inspect the data with Pandas.
- `visualization_tutorial.py`: A script showing how to create a scatter plot from the data using Matplotlib to visualize the relationship between house size and price.