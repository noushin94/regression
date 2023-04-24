import pandas as pd
import numpy as np

#reading file
df = pd.read_csv(r'/content/sample_data/california_housing_train.csv')
df.head()

# allocating x and y
X = df.drop('median_house_value', axis=1)
Y = df['median_house_value']