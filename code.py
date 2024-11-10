# First, import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Convert to pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column (species)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display basic information about the dataset
print("Dataset Shape:", iris_df.shape)
print("\nFirst few rows of the dataset:")
print(iris_df.head())

# Display basic statistics
print("\nBasic statistics of the dataset:")
print(iris_df.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(iris_df.isnull().sum())

# Create a simple visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='species')
plt.title('Iris Dataset: Sepal Length vs Sepal Width')
plt.show()

# Display unique species
print("\nUnique species in the dataset:")
print(iris_df['species'].unique())

# Show distribution of features
plt.figure(figsize=(12, 8))
iris_df.boxplot(by='species', figsize=(12, 8))
plt.title('Feature Distribution by Species')
plt.suptitle('') # This removes the automatic suptitle
plt.show()
