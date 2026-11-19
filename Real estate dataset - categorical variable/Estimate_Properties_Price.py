"""
In this exercise, you will work with the final version of the real estate dataset - including the features size, year and view, along with the target price. Your goal is to build two regression models to predict property prices and analyze the influence of the "sea view" feature.

Steps to follow:
1. Regression model 1: Build a regression model using only the features size and year.
2. Regression model 2: Build another regression model that incorporates the view feature in addition to size and year.
3. Predict property price: Use both models to estimate the price of the following property:
  - size: 823
  - year: 2012
  - view: No sea view
  - Note: The first model does not require the view parameter
4. Output Predictions:
  - Print the predictions from both models.
  - Make sure you round each predicted price to the nearest whole number.
  - Important: Use the provided print functions without modifying them, or follow the exact same format.
"""

# The exercise requires building two regression models to predict property prices based on a dataset that includes the features 'size', 'year', and 'view', along with the target variable 'price'. The first model should use only 'size' and 'year', while the second model should additionally include the 'view' feature. After building the models, predictions for a specific property are to be made and displayed.

# The solution is structured as follows:

# Import Libraries: The necessary libraries are imported: pandas for data manipulation and statsmodels for regression analysis.

import pandas as pd
import statsmodels.api as sm

# Load Data: The real estate dataset is loaded into a DataFrame using pd.read_csv(). A copy of the data is made for further processing.

raw_data = pd.read_csv('real_estate_price_size_year_view.csv')
data = raw_data.copy()

# Handle Dummy Variable: The 'view' categorical variable is mapped to binary numerical values (1 for 'Sea view' and 0 for 'No sea view') to prepare it for regression analysis.

data['view'] = data['view'].map({'Sea view': 1, 'No sea view': 0})

# Define and Fit Regression Model 1: The first regression model is created using 'size' and 'year' as predictors. The target variable 'price' is assigned to y, and the predictors are assigned to x1. A constant is added to x1 using sm.add_constant(), and the model is fitted using sm.OLS().

y = data['price']
x1 = data[['size','year']]
x = sm.add_constant(x1)
regression1 = sm.OLS(y,x).fit()

# Define and Fit Regression Model 2: The second regression model includes 'size', 'year', and 'view'. Similar to the first model, 'price' is assigned to y, and all predictors are included in x1. The model is fitted in the same way.

x1 = data[['size','year','view']]
x = sm.add_constant(x1)
regression2 = sm.OLS(y,x).fit()

# Make Predictions: Predictions are made for a specific property by substituting the values (823 for size, 2012 for year, and 0 for view in model 2). The predictions are made using the predict() method.

prediction_1 = regression1.predict([1,823,2012])
prediction_2 = regression2.predict([1,823,2012,0])

# Print Predictions: Finally, the predicted prices from both models are printed, rounded to the nearest whole number.

print('Price without considering view: ', round(prediction_1[0]))
print('Price when considering the view: ', round(prediction_2[0]))