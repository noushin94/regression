import pandas as pd
import numpy as np

#reading file
df = pd.read_csv(r'/content/sample_data/california_housing_train.csv')
df.head()

# allocating x and y
X = df.drop('median_house_value', axis=1)
Y = df['median_house_value']

#packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

#spliting test and train
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=22)