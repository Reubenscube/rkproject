# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:57:12 2015

@author: reubenkarchem
"""
#YELP DATA HOMEWORK

#TASK 1
import pandas as pd
yelp_data = pd.read_csv('yelp.csv', index_col=False)
yelp_data.head()
yelp_data.shape
#TASK 1 (Bonus)
# read the data from yelp.json into a list of rows
# each row is decoded into a dictionary using using json.loads()
import json
ydj = []
with open('yelp.json', 'rU') as f:
    for row in f:
        row = json.loads(row)
        ydj.append(row)

# show the first review
ydj[0]

# convert the list of dictionaries to a DataFrame
ydj_df = pd.DataFrame(ydj)
ydj_df.head()

# add DataFrame columns for cool, useful, and funny
df2 = ydj_df['votes'].str.join(sep='*').str.get_dummies(sep='*')
#df2 = pd.get_dummies(ydj_df['votes']) this would be used for distinct cats
#df2.head()
ydj_df['votes'][0]['funny'] #this is the first item in the first row
#iterate through each row and append the value to the new list
ydj_df['votes_funny'] = [row['funny'] for row in ydj_df['votes']]
ydj_df.head()
#Do this for the two other keys in the dictionary

ydj_df['vote'] = pd.to_datetime(df['df2'])
ydj_df2 = pd.merge(ydj_df, df2, how='inner', on=None, left_on=None, right_index=True,
      left_index=True, sort=False, copy=True)


#TASK 2
#Explore the relationship between each of the vote types (cool/useful/funny)
#and the number of stars.
# treat stars as a categorical variable and look for differences between groups
yelp_data.groupby('stars').describe()
#Higher starred reviews tended to be more cool, but less useful or funny
#1-2 stars were viewed as much more useful

# correlation matrix
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
%matplotlib inline
scatter_matrix(yelp_data, alpha=0.2, figsize=(10, 10), diagonal='kde')


#TASK 3
#Define cool/useful/funny as the features, and stars as the response.
feature_cols = ['cool','useful','funny']
response = ['stars']


#TASK 4
#Fit a linear regression model and interpret the coefficients.
#Do the coefficients make intuitive sense to you?
#Explore the Yelp website to see if you detect similar trends.
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
X = yelp_data[feature_cols]
y = yelp_data[response]
linreg.fit(X,y)

print feature_cols
print linreg.coef_ #[ 0.27435947 -0.14745239 -0.13567449]]
#This matches the general pattern I observed early when looking at
#the stars groupby means.  Higher starred reviews received more
#cool votes, but fewer useful and funny votes.  Low star reviews
#tend to be viewed as useful.  That's similar to what I saw on Yelp
#It's as though negative reviews serve as powerful warnings.


#TASK 5
#Evaluate the model by splitting it into training and testing sets
#and computing the RMSE. Does the RMSE make intuitive sense to you?
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
print feature_cols
print linreg.coef_

from sklearn import metrics
import numpy as np
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))
#RMSE = 1.18661152827
#The RMSE doesn't seem terrible - the average error is 1-ish stars.

# Define a function that accepts a list of features and returns testing RMSE
def rmse_calc(features):
    # modules already imported    
    X = yelp_data[features]
    y = yelp_data['stars']
    #instatiate linreg
    linreg = LinearRegression()
    linreg.fit(X, y)
    y_pred = linreg.predict(X)
    return np.sqrt(metrics.mean_squared_error(y, y_pred))


#TASK 6
print rmse_calc(['cool','useful','funny'])
#RMSE = 1.18738481933
print rmse_calc(['cool'])
print rmse_calc(['useful'])
print rmse_calc(['funny'])
print rmse_calc(['cool','useful']) #1.198788
print rmse_calc(['funny','useful'])#1.21173
print rmse_calc(['cool','funny']) #1.97741
#RMSE looks to be worse when just using one feature,
#but they all perform about equal ~ 1.21


#TASK 7
#add a new feature, review length yelp_data[rev_len] =
rev_len = []
for row in yelp_data['text']:
    rev_len.append(len(row))
yelp_data['rev_len'] = rev_len

love = []
for row in yelp_data['text']:
    if 'love' in row:
        love.append(1)
    else:
        love.append(0)
yelp_data['love'] = love
hate = []
for row in yelp_data['text']:
    if 'hate' in row:
        hate.append(1)
    else:
        hate.append(0)
yelp_data['hate'] = hate

#Model with the new features and MRSE
feature_cols2 = ['cool','useful','funny','rev_len','love','hate']
X = yelp_data[feature_cols2]
y = yelp_data['stars']

linreg2 = LinearRegression()
linreg2.fit(X, y)
y_pred2 = linreg2.predict(X)
print feature_cols2
print linreg2.coef_
np.sqrt(metrics.mean_squared_error(y, y_pred2))
#RMSE is smaller 1.16562. old features have the same pos/neg sign
#but their absolute values are smaller


#TASK 8
#Compare your best RMSE on the testing set with the RMSE for the
# "null model", which is the model that ignores all features and
#simply predicts the mean response value in the testing set.
# compute null accuracy
y_test.value_counts().head(1) / len(y_test)
#y_test has 2500 rows

# split the data (outside of the function)
    #What does this mean?

# create a NumPy array with the same shape as y_test
# fill the array with the mean of y_test
y_test.mean()
y_null = np.full(2500, y_test.mean())
y_null

# calculate null RMSE
np.sqrt(metrics.mean_squared_error(y_test, y_null))
#RMSE using the null model is 1.1816091401135995

yelp_data.head()


#TASK 9
#Try this as a classification problem with a KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
knn.fit(X_train, y_train) #includes the additional features

y_pred3 = knn.predict(X_test)
metrics.accuracy_score(y_test, y_pred3)
#.2792

#Finding the best number of neighbors
range_k = range(1,201)
training_error = []
testing_error = []
for k in range_k:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred4 = knn.predict(X)
    training_accuracy = metrics.accuracy_score(y,y_pred4)
    training_error.append(1 - training_accuracy)
    
    knn.fit(X_train,y_train)
    y_pred4 = knn.predict(X_test)
    testing_accuracy = metrics.accuracy_score(y_test,y_pred4)
    testing_error.append(1 - testing_accuracy)

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
column_dict = {'K': range_k, 'training_error':training_error, 'testing_error':testing_error}
df = pd.DataFrame(column_dict).set_index('K').sort_index(ascending=True)
df.plot(y='testing_error')
plt.xlabel('Value of K for KNN')
plt.ylabel('Error (lower is better)')
#You want to look at the graph of the testing error, and stop K= when
#there is marginal improvement in error.
#think about the number in context of the data set
df.sort('testing_error').head(10)
#K =152 is the best testing_error, 0.6196


#TASK 10
# use linear regression to make continuous predictions


# round its predictions to the nearest integer


# calculate classification accuracy of the rounded predictions