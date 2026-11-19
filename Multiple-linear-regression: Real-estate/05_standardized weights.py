"""
In this exercise, you will analyze the real_estate_price_size_year.csv dataset, containing 2 features (size, year) and 1 target variable (price)

Your task is to: 
    1. Create a linear regression
    2. Standardize the features using sklearn
    3. For each feature and the intercept find the coefficient (weight)
    4. Arrange the values in a summary table (DataFrame) with 2 columns - features, weights. The order of the features is intercept, size, year.
    5. Print the summary table. The final output should look like this:
    Features        Weights 
0  Intercept  xxxxxx.xxxxxx 
1       Size   xxxxx.xxxxxx 
2       Year   xxxxx.xxxxxx

"""

# 1. Import Necessary Libraries
# pandas: Used for data loading and organizing the final summary table
import pandas as pd
# LinearRegression: Used to build and fit the Ordinary Least Squares (OLS) regression model
from sklearn.linear_model import LinearRegression
# StandardScaler: Used to standardize features to zero mean and unit variance
from sklearn.preprocessing import StandardScaler

# 2. Load the Dataset and Define Variables
# Read the real estate CSV dataset into a DataFrame
data = pd.read_csv('real_estate_price_size_year.csv')
# Select 'size' and 'year' as the 2D feature matrix (x)
x = data[['size', 'year']]
# Select 'price' as the 1D target vector (y)
y = data['price']

# 3. Standardize the Features
# Create an instance of StandardScaler
scaler = StandardScaler()
# Compute the mean (μ) and standard deviation (σ) of each feature column in x
scaler.fit(x)
# Transform x into standardized z-scores: z = (x - μ) / σ
x_scaled = scaler.transform(x)

# 4. Create and Fit the Linear Regression Model
# Initialize the LinearRegression model
reg = LinearRegression()
# Train the model using the standardized features (x_scaled) and target (y)
reg.fit(x_scaled, y)

# 5. Build and Print the Summary Table
# Create a DataFrame listing 'Intercept', 'Size', and 'Year' under the 'Features' column
reg_summary = pd.DataFrame(['Intercept', 'Size', 'Year'], columns=['Features'])
# Populate the 'Weights' column with the model's intercept and the two standardized coefficients
reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]

# Print the final summary DataFrame
print(reg_summary)