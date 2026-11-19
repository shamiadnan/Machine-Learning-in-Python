"""
In this exercise, you will analyze the real_estate_price_size_year_view.csv dataset, containing 3 features (size, year, view) and 1 target variable (price)

Your task is to:
    1. Create a linear regression
    2. For each feature find:
        - the coefficient of the linear regression
        - the p-value, rounded to 3 decimal places
    3. Arrange the values in a summary table (DataFrame) with 3 columns - features, coefficients, p-values. The order of the features is size, year, view.
    4. Print the summary table. The final output should look like this:
  Features  Coefficients  p-values 
0     size    xxx.xxxxxx     x.xxx 
1     year   xxxx.xxxxxx     x.xxx 
2     view  xxxxx.xxxxxx     x.xxx

"""

# 1. Import Necessary Libraries
# pandas: Used for loading, manipulating, and organizing data in DataFrames
import pandas as pd
# LinearRegression: Used to build and fit the Ordinary Least Squares (OLS) regression model
from sklearn.linear_model import LinearRegression
# f_regression: Used to perform univariate linear regression tests to calculate F-statistics and p-values
from sklearn.feature_selection import f_regression

# 2. Load and Prepare the Dataset
# Read the CSV dataset into a raw DataFrame
raw_data = pd.read_csv('real_estate_price_size_year_view.csv')
# Create a working copy so the original raw_data remains intact
data = raw_data.copy()
# Encode the categorical 'view' column into binary numerical values (dummy variable)
# 'Sea view' becomes 1 and 'No sea view' becomes 0 so the regression model can process it
data['view'] = data['view'].map({'Sea view': 1, 'No sea view': 0})

# 3. Define Features (Independent Variables) and Target (Dependent Variable)
# Select 'size', 'year', and 'view' as the 2D input feature matrix (x)
x = data[['size', 'year', 'view']]
# Select 'price' as the 1D target vector (y) we want to predict
y = data['price']

# 4. Create and Fit the Linear Regression Model
# Instantiate the LinearRegression model object
reg = LinearRegression()
# Train (fit) the model on the features (x) and target (y) to compute optimal weights/coefficients
reg.fit(x, y)

# 5. Calculate p-values for Statistical Significance
# f_regression(x, y) returns a tuple of two arrays: (F-statistics, p-values)
# Indexing with [1] extracts just the array of p-values corresponding to each feature
p_values = f_regression(x, y)[1]

# 6. Build the Regression Summary Table
# Initialize a DataFrame with the feature names in their exact order ('size', 'year', 'view')
reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features'])
# Add a 'Coefficients' column containing the learned slope weights from the fitted model
reg_summary['Coefficients'] = reg.coef_
# Add a 'p-values' column, rounding each p-value to 3 decimal places for readability
reg_summary['p-values'] = p_values.round(3)

# 7. Output the Final Table
# Print the formatted summary DataFrame
print(reg_summary)