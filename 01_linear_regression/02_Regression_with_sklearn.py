"""
For this exercise, you will use the simplest version of the real estate dataset (real_estate_price_size.csv). As a reminder, it aims to predict property prices based on their size. Thus, it contains just one feature (size) and one target variable (price).

Your task is to use sklearn in order to build a linear regression based on the data. Then, following the provided format print these numbers:
    1. Print the R-squared value of the regression model
    2. Print the coefficient of the linear model (as is returned by sklearn)
    3. Print the intercept of the linear model
    4. Make predictions about the price of two properties with sizes 1023 sq. ft. and 690 sq. ft., respectively.
        - Print these predictions as an array of two numbers

The final output should have the following format (the numbers here are just example numbers, not the correct ones):
R-squared: 0.6996391865847586
Coefficient: [227.32874259]
Intercept: 426832.60180122897
Predictions: [82572.45547551 130340.93419151]
 
Note: When making predictions with sklearn, ensure that your new data matches the format of the training data. If your training data included named features (i.e., column names in a DataFrame), your new data must also be provided as a DataFrame with the exact same column names and order.

"""

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('real_estate_price_size.csv')

# Define model variables
x = data[['size']]
y = data['price']

# Build the model
reg = LinearRegression()
reg.fit(x,y)

# Print out model metrics
print("R-squared: ", reg.score(x,y))
print("Coefficient: ", reg.coef_)
print("Intercept: ", reg.intercept_)

# Make predisctions on new data
# Note that you can define test_data as an array, not a DataFrame...
# BUT! sklearn will give a warning since we used named data to create the model
test_data = pd.DataFrame(data=[1023, 690], columns = ['size'])
predictions = reg.predict(test_data)

# Print the predictions
print("Predictions: ", predictions)