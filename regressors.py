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

#KNN resgressor
KNN = KNeighborsRegressor(n_neighbors=5, weights='uniform', n_jobs=3)
KNN.fit(Xtrain, Ytrain)
pred = KNN.predict(Xtest)
pred

#MSE
KNN_R = KNeighborsRegressor(n_neighbors=5, weights='uniform', n_jobs=3)
KNN_R.fit(Xtrain, Ytrain)
pred = KNN_R.predict(Xtest)

print(mean_squared_error(Ytest,pred))

#Decision TRee
DT_R = DecisionTreeRegressor(criterion='absolute_error', max_depth=5, min_samples_split=7, min_samples_leaf=2, random_state=2)
DT_R.fit(Xtrain, Ytrain)
pred = DT_R.predict(Xtest)

print(mean_squared_error(Ytest,pred))

#neural network
MLP_R = MLPRegressor(max_iter=200, random_state=2020, activation='relu', hidden_layer_sizes=(10,20),
                    learning_rate_init=0.01, learning_rate='invscaling')
MLP_R.fit(Xtrain, Ytrain)
pred = MLP_R.predict(Xtest)

print(mean_squared_error(Ytest,pred))

#SVM
SV_R = SVR(kernel='rbf', C=1.2)
SV_R.fit(Xtrain, Ytrain)
pred = SV_R.predict(Xtest)

print(mean_squared_error(Ytest,pred))