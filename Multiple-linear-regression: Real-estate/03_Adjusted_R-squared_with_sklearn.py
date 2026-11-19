"""
In this exercise, you will evaluate the performance of a multiple linear regression model using Adjusted R-Squared. The dataset real_estate_price_size_year.csv contains the following columns:
    - size: The size of the property in square feet.
    - year: The year the property was built.
    - price: The price of the property in USD.

Your task is to:
    1. Build a multiple linear regression model using the size and year columns as input features to predict the price.
    2. Calculate the Adjusted R-Squared value for the model based on the number of observations (
), number of predictors (
), and the R-Squared score (
 
) of the model.
        - The formula for Adjusted R-Squared is:
 \text{Adjusted R-squared}=1-\frac{(1-R^2)(n-1)}{n-p-1}
 
    3. Finally, print the Adjusted R-Squared value in the format:
Adjusted R-squared: ...

"""

# 1. Data Import
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the real estate dataset into a pandas DataFrame
data = pd.read_csv('real_estate_price_size_year.csv')

# 2. Feature Selection
# Define the independent variables (features) as a 2D DataFrame
x = data[['size', 'year']]
# Define the dependent variable (target) as a 1D Series
y = data['price']

# 3. Model Creation and Fitting
# Initialize the Ordinary Least Squares (OLS) linear regression model
reg = LinearRegression()
# Train (fit) the model using the features (x) and target (y)
reg.fit(x, y)

# 4. R-Squared Calculation
# Calculate the standard R-squared score (proportion of variance explained by the model)
r2 = reg.score(x, y)

# 5. Adjusted R-Squared Computation
# Extract the number of observations (rows) and predictors (columns) from x
n = x.shape[0]  # Number of samples (observations)
p = x.shape[1]  # Number of independent variables (features)

# Apply the Adjusted R-squared formula to penalize for unnecessary predictors:
# Adjusted R^2 = 1 - [(1 - R^2) * (n - 1) / (n - p - 1)]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# 6. Output
# Print the final Adjusted R-squared value (~0.772)
print("Adjusted R-squared: ", adjusted_r2)