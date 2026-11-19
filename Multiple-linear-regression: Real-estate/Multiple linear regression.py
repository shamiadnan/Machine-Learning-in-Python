"""
For this task, you will use the familiar real estate dataset with a twist. It now contains two columns that could be potential predictors of the price - size and year of construction.
Using this modified dataset (real_estate_price_size_year.csv), perform the following steps:
1. Fit a linear regression model using the statsmodels library, predicting the price based on the size and year of construction.
2. Check the adjusted R-squared value from the model's summary.
3. Complete the provided print statement by including the adjusted R-squared value, rounded to 3 decimal places, and ensure your output matches the following format exactly (including spacing).

"""
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('real_estate_price_size_year.csv')

y = data['price']
x1 = data[['size','year']]

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()

'The task involves using a real estate dataset to build a linear regression model that predicts house prices based on two predictors: the size of the house and the year it was constructed. The goal is to fit a model using the statsmodels library, evaluate its performance through the adjusted R-squared value, and print this value in a specific formatted way.'
'The solution begins by importing the necessary libraries and loading the dataset, which contains columns for price, size, and year:'

import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('real_estate_price_size_year.csv')
'Next, it defines the dependent variable y (the prices) and the independent variables x1 (size and year of construction):'

y = data['price']
x1 = data[['size','year']]
'A constant is added to the independent variables to account for the intercept in the regression model:'

x = sm.add_constant(x1)
'The Ordinary Least Squares (OLS) method is then used to fit the regression model:'

results = sm.OLS(y, x).fit()
'Although the summary of the model is commented out, the core of the task involves retrieving the adjusted R-squared value. This is done with:'

#print(results.summary())
'Finally, the solution prints the adjusted R-squared value, rounded to three decimal places, ensuring the format matches the specified output:'

print(f"Adjusted R-squared:  {results.rsquared_adj:.3f}")