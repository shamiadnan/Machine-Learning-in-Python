"""
In this exercise, you will analyze the real_estate_price_size_year.csv dataset, containing 2 features (size, year) and 1 target variable (price)

Your task is to: 
1. Create a linear regression
2. Standardize the features using sklearn
3. Obtain the estimated prices for 2 new properties:
  - Property 1: size = 528; year = 2020
  - Property 2: size = 843; year = 2012
4. Print the predicted price. The format should be: [prediction_1 prediction_2]

Note: When making predictions with sklearn, ensure that your new data matches the format of the training data. If your training data included named features (i.e., column names in a DataFrame), your new data must also be provided as a DataFrame with the exact same column names and order. If your training data was scaled, the new data must be scaled in the same manner.

"""

# 1. Import Necessary Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 2. Load the Dataset and Define Features/Target
data = pd.read_csv('real_estate_price_size_year.csv')
x = data[['size', 'year']]  # 2D DataFrame of features
y = data['price']           # 1D Series of target prices

# 3. Standardize the Features
scaler = StandardScaler()
scaler.fit(x)                  # Compute mean and standard deviation for 'size' and 'year'
x_scaled = scaler.transform(x) # Standardize training features (mean = 0, std = 1)

# 4. Create and Fit the Linear Regression Model
reg = LinearRegression()
reg.fit(x_scaled, y)           # Train model on standardized features

# 5. Prepare, Scale, and Predict on New Properties
# Match the exact DataFrame format and column names ('size', 'year') used in training
new_data = pd.DataFrame(data=[[528, 2020], [843, 2012]], columns=['size', 'year'])

# Scale the new data using the SAME scaler fitted on the training data
new_data_scaled = scaler.transform(new_data)

# Print predicted prices in [prediction_1 prediction_2] format
print(reg.predict(new_data_scaled))