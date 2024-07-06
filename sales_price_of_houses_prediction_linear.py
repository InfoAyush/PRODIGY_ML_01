import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Create a synthetic dataset
np.random.seed(42)
n_samples = 100

square_footage = np.random.randint(500, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
prices = (square_footage * 300) + (bedrooms * 50000) + (bathrooms * 30000) + np.random.randint(-50000, 50000, n_samples)

data = pd.DataFrame({
    'SquareFootage': square_footage,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Price': prices
})

# Display the first few rows of the dataset
print(data.head())


# Check for missing values
print(data.isnull().sum())

# Visualize the data
sns.pairplot(data)
plt.show()

# Check the correlation matrix
print(data.corr())


# Split the data into features (X) and target (y)
X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")


# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Print the coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")


# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save results to a CSV file
results = pd.DataFrame({
    'Actual Prices': y_test,
    'Predicted Prices': y_pred
})
results.to_csv('results.csv', index=False)

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
