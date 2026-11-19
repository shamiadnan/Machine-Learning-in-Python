"""
Using the same dataset as in the previous exercise (real_estate_price_size.csv), perform the following steps:
    1. Fit a linear regression model using the statsmodels library, predicting the price based on the size.
    2. Check the R-squared value from the model's summary.
    3. Complete the provided print statement by including the R-squared value exactly as it appears in the summary (without adding extra spaces or modifying the format).
    
"""
import pandas as pd
import statsmodels.api as sm

# Load the data
data = pd.read_csv('real_estate_price_size.csv')

# Define regression variables
y = data['price']
x1 = data['size']

# Fit the regression model
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()

# Check the summary and R-squared value, and the comment the code so it doesn't execute
#print(results.summary())

# Print the R-squared value
print("R-squared: ", 0.745)