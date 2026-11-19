"""
You are provided with the real_estate_price_size_year.csv dataset, containing two features (size, year) and one target (price).
Your goal is to build a multiple linear regression model to predict price using size and year as input features. After training the model, extract and print the coefficient associated with the year feature.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('real_estate_price_size_year.csv')
x = data[['size', 'year']]
y = data['price']

reg = LinearRegression()
reg.fit(x, y)

print("Year coefficient: ", reg.coef_[1])