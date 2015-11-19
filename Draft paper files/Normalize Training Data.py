# -*- coding: utf-8 -*-
"""
@author: reubenkarchem
Moving iPython Notebook code to Spyder
"""

import pandas as pd
import csv
#Import Data
training_data = pd.read_table('TrainingData.csv', sep=',', header=0, index_col='Unnamed: 0')

#Rename columns
training_data.rename(columns={'Months since Last Donation':'Last_Donation', 'Number of Donations': 'Number_of_Donations', 'Total Volume Donated (c.c.)':'Volume_Donated', 'Made Donation in March 2007':'March_Donation', 'Months since First Donation':'First_Donation'}, inplace=True)
training_data.shape

#Descriptive statistics
training_data.Last_Donation.describe()
training_data.First_Donation.describe()
training_data.Number_of_Donations.describe()
training_data.Volume_Donated.describe()
#Last Donation, Number of Donations/Volume appear to have outliers at the high end.
#For example, # of donations 75% are below 15 and then a cluser over 30.
training_data.boxplot(column='Number_of_Donations', vert=False)
training_data.boxplot(column='Last_Donation', vert=False)

#Box Plots grouped by March_Donation
import matplotlib.pyplot as plt
training_data.boxplot(column='Last_Donation', by='March_Donation', vert=False)
training_data.boxplot(column='First_Donation', by='March_Donation', vert=False)
training_data.boxplot(column='Number_of_Donations', by='March_Donation', vert=False)
training_data.boxplot(column='Volume_Donated', by='March_Donation', vert=False)

#Histograms
plt.hist(training_data.Last_Donation, 25)
plt.hist(training_data.First_Donation, 25)
plt.hist(training_data.Number_of_Donations, 25)
plt.hist(training_data.Volume_Donated, 25)

#Add standardized features to the dataframe
from sklearn import preprocessing
import numpy as np

#Standardized to the mean and standard deviations
std_scale = preprocessing.StandardScaler().fit(training_data[['First_Donation','Last_Donation','Number_of_Donations','Volume_Donated']])
training_data_std = std_scale.transform(training_data[['First_Donation','Last_Donation','Number_of_Donations','Volume_Donated']])

#First and Last Donations
plt.scatter(training_data['First_Donation'], training_data['Last_Donation'], color='green', label='input scale', alpha=0.5)
plt.scatter(training_data_std[:,0], training_data_std[:,1], color='blue', label='std scaled', alpha=0.3)
plt.title('First and Last Donations re-scaled')
plt.xlabel('first donation')
plt.ylabel('last donation')


#Volume and Number of Donations, should be a line
plt.scatter(training_data['Number_of_Donations'], training_data['Volume_Donated'], color='green', label='input scale', alpha=0.5)
plt.scatter(training_data_std[:,2], training_data_std[:,3], color='blue', label='std scaled', alpha=0.3)
plt.title('Number and Volume re-scaled')
plt.xlabel('Number_of_Donations')
plt.ylabel('Volume_Donated')

#Dataframe with Standardized Values
df_std = pd.DataFrame(data=training_data_std, index=training_data.index, columns=['First_Donation','Last_Donation','Number_of_Donations','Volume_Donated'])
df_std.head()
df_std['March_Donation'] = training_data['March_Donation']

#Distribution keeps the same relative distances, but condensed
training_data.boxplot(column='Last_Donation', vert=False)
df_std.boxplot(column='Last_Donation', vert=False)


#LOGISTIC REGRESSION COMPARISON
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
from sklearn.cross_validation import train_test_split
feature_cols = ['First_Donation', 'Last_Donation','Number_of_Donations','Volume_Donated']
X = training_data[feature_cols]
y = training_data.March_Donation
X_train,X_test, y_train, y_test = train_test_split(X, y, random_state=3)

#Logreg on the training data features with no change
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred)

#Logreg on the dataframe where features have been standardized
logreg2 = LogisticRegression(C=1e9)
X2 = df_std[feature_cols]
y = df_std.March_Donation
X_train2,X_test2, y_train2, y_test2 = train_test_split(X2, y, random_state=3)
logreg2.fit(X_train2, y_train2)
y_pred2 = logreg2.predict(X_test2)
print metrics.accuracy_score(y_test2, y_pred2)
#No change in the metrics by using this standardization change


#MINMAX
#MinMax didn't work on integers, convert original df to floats
training_data.First_Donation = training_data.First_Donation.astype('float64') 
training_data.Last_Donation = training_data.Last_Donation.astype('float64') 
training_data.Number_of_Donations = training_data.Number_of_Donations.astype('float64') 
training_data.Volume_Donated = training_data.Volume_Donated.astype('float64') 

def sci_minmax(X):
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    return minmax_scale.fit_transform(X)
sci_minmax(training_data.First_Donation)
sci_minmax(training_data.Last_Donation)
sci_minmax(training_data.Number_of_Donations)
sci_minmax(training_data.Volume_Donated)


#Create MinMax DF
df_mm = pd.DataFrame(index=training_data.index)
df_mm['First_Donation'] = sci_minmax(training_data.First_Donation)
df_mm['Last_Donation'] = sci_minmax(training_data.Last_Donation)
df_mm['Number_of_Donations'] = sci_minmax(training_data.Number_of_Donations)
df_mm['Volume_Donated'] = sci_minmax(training_data.Volume_Donated)
df_mm['March_Donation'] = training_data['March_Donation']

df_mm.describe()
training_data.describe()

#Distribution keeps the same relative distances, but condensed
training_data.boxplot(column='Last_Donation', vert=False)
df_mm.boxplot(column='Last_Donation', vert=False)


#Logreg on the MinMax dataframe
logreg3 = LogisticRegression(C=1e9)
X3 = df_mm[feature_cols]
y = df_mm.March_Donation
X_train3,X_test3, y_train3, y_test3 = train_test_split(X3, y, random_state=3)
logreg3.fit(X_train3, y_train3)
y_pred3 = logreg3.predict(X_test3)
print metrics.accuracy_score(y_test3, y_pred3)
#Again, this shows the same level of accuracy.

#For now, it looks like transforming features only made the dataviz
#look better, but did not impact the model performance

#K-Nearest Neighbors

#KNN with 5 neighbors on standardized dataframe
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train2, y_train2)
y_pred22 = knn.predict(X_test2)
print metrics.accuracy_score(y_test2, y_pred22)
#0.7847, KNN performs worse than the logistic regression at 5 neighbors

#KNN with 5 neighbors on the MinMax dataframe
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train3, y_train3)
y_pred32 = knn.predict(X_test3)
print metrics.accuracy_score(y_test3, y_pred32)
#0.80555, same as the original dataframe, but better than the previous KNN model


#CROSS-VALIDATION
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
kf = KFold(25, n_folds=5, shuffle=False)

#CV on the original dataframe
knn11 = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn11, X_train, y_train, cv=5, scoring='accuracy')
print scores
print scores.mean() #0.747607, training_data

knn11 = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn11, X_train2, y_train2, cv=5, scoring='accuracy')
print scores
print scores.mean() #0.761587810746, standardized dataframe

scores = cross_val_score(knn11, X_train3, y_train3, cv=5, scoring='accuracy')
print scores
print scores.mean() #0.743090082866, MinMax dataframe

#Checking for the best # of Neighbors based on Cross_val Score, Standardized DF
#Std better than MM
k_range = range(1, 51)
k_scores = []
for k in k_range:
    knn2 = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn2, X_train2, y_train2, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

#print k_scores
plt.plot(k_range, k_scores)
plt.xlabel('KNN K-value')
plt.ylabel('Cross_val_score')
#The maximum cross_val_score occurs at around K=22, but still not better than
#the logistic regression model on the original dataframe.

#Checking for best Neighbors from MinMax
k_range = range(1, 51)
k_scores2 = []
for k in k_range:
    knn3 = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn3, X_train3, y_train3, cv=5, scoring='accuracy')
    k_scores2.append(scores.mean())

#print k_scores2
plt.plot(k_range, k_scores)
plt.xlabel('KNN K-value')
plt.ylabel('Cross_val_score')
print max(k_scores)
print max(k_scores2)