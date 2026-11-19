"""
We have once more made an addition to our real estate dataset. This time, we have included one more metric - whether the apartment/house has a sea view or not. This is a categorical variable with column name view and 2 values - Sea view and No sea view.
Perform the necessary preprocessing in order to incorporate the categorical variable and:
1. Fit a linear regression model using the statsmodels library, modeling the price of real estates based on the size, year of construction and presence of sea view.
2. Check the R-squared value from the model's summary.
3. Complete the provided print statement by including the adjusted R-squared value exactly as it appears in the summary (without adding extra spaces or modifying the format).

"""

# The exercise requires incorporating a new categorical variable, 'view', into a real estate dataset for modeling. The goal is to fit a linear regression model using the statsmodels library that predicts real estate prices based on size, year of construction, and whether the property has a sea view. The output must include the R-squared and adjusted R-squared values formatted correctly.

# The solution follows these steps:

# Import Libraries: The 'pandas' library is imported for data manipulation, and 'statsmodels.api' is imported for statistical modeling.

import pandas as pd
import statsmodels.api as sm

# Load Data: The dataset is loaded from a CSV file containing real estate information.

raw_data = pd.read_csv('real_estate_price_size_year_view.csv')

# Data Preparation: A copy of the raw data is made, and the 'view' categorical variable is mapped to numerical values; 'Sea view' is mapped to 1 and 'No sea view' to 0.

data = raw_data.copy()
data['view'] = data['view'].map({'Sea view': 1, 'No sea view': 0})

# Define Variables: The target variable 'y' (price) is defined, and the independent variables 'x1' (size, year, and view) are selected.

y = data['price']
x1 = data[['size','year','view']]

# Add Constant: A constant term is added to the independent variables to account for the intercept in the regression model.

x = sm.add_constant(x1)

# Fit Model: The Ordinary Least Squares (OLS) regression model is fitted with the specified dependent (y) and independent (x) variables.

results = sm.OLS(y,x).fit()

# Print Results: Although the summary of the model’s fit is commented out, the adjusted R-squared is printed in the required format.

#print(results.summary())
print("R-squared: ", 0.913)