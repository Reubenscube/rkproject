# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:40:09 2015

@author: reubenkarchem
"""
import pandas as pd
#1. Read yelp.csv into a DataFrame
yelp_data = pd.read_csv('yelp.csv', index_col=False)
yelp_data.head()

#2. Create a new DataFrame that only contains the 5-star and 1-star reviews.
ydata_15 = yelp_data[(yelp_data.stars ==1) | (yelp_data.stars ==5)]
#adjusting all 5-values to 0, so that ROC AUC step will work
ydata_15.stars.replace('5', 0, inplace=True)

#3. Split the new DataFrame into training and testing sets, using the review text as the only feature and the star rating as the response.
from sklearn.cross_validation import train_test_split
X = ydata_15.text
y = ydata_15.stars
X_train, X_test, y_train, y_test = train_test_split(X, y)


#4. Use CountVectorizer to create document-term matrices from X_train and X_test. 
#    Hint: If you run into a decoding error, instantiate the vectorizer with the argument decode_error='ignore'.
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer() #instantiate
vect.fit(X_train)
simple_X_train_dtm = vect.transform(X_train)
simple_X_train_dtm
# Make the doc-term matrix into a df: X_train_dtm_df = pd.DataFrame(simple_X_train_dtm.toarray(), columns=vect.get_feature_names())

vect_text = CountVectorizer() #instantiate
vect_text.fit(X_test)
simple_X_test_dtm = vect.transform(X_test)
# Make the doc-term matrix into a df: X_test_dtm_df = pd.DataFrame(simple_X_test_dtm.toarray(), columns=vect.get_feature_names())


#5. Use Naive Bayes to predict the star rating for reviews in the testing set, and calculate the accuracy.
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB() #Instantiate
nb.fit(simple_X_train_dtm, y_train) #Fit the NB model to the train doc-term matrix, and y_train outcomes

y_pred = nb.predict(simple_X_test_dtm) #Predict based on X_test

from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred) #Evaluate the NB model


#6. Calculate the AUC
y_pred_prob = nb.predict_proba(simple_X_test_dtm)[:, 1]
#y_test values have to be 0 and 1, so going back to adjust star feature to set 5 == 0
metrics.roc_auc_score(y_test, y_pred_prob)
#0.94820418698765185


#7. Plot the ROC curve.
import matplotlib.pyplot as plt
# %matplotlib inline
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

#8. Print the confusion matrix, and calculate the sensitivity and specificity
print metrics.confusion_matrix(y_test, y_pred)
'''
these change every time I re-read the data
[[835  24]   True Positive, False Positive
 [ 50 113]]  False Negative, True Negative
sensitivity = 835/(835+24) = 0.9720605355
specificity = 113/(113+50) = 0.6932515337
I have low specificity compared to sensitivity. My model has too many false
negatives, so I'm not catching certain 1-star reviews
'''
#9. Browse through the review text for some of the false positives and
#false negatives. Based on your knowledge of how Naive Bayes works, do
#you have any theories about why the model is incorrectly classifying
#these reviews?
rev_len = 0
for row in X_test[y_test > y_pred]:  #False Pos
    rev_len += len(row)
for row in X_test[y_test < y_pred]:  #False Neg
    rev_len += len(row)
#On average, the False Neg reviews had a fraction of the characters 300 versus 1400.

#10. Let's pretend that you want to balance sensitivity and specificity.
# You can achieve this by changing the threshold for predicting a 5-star review.
# What threshold approximately balances sensitivity and specificity?
#try passing in a series of different thresholds for Positive
metrics.precision_recall_curve(y_test, y_pred_prob, pos_label=None, sample_weight=None)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
print metrics.confusion_matrix(y_test, y_pred)
import numpy as np
y_pred = np.where(y_pred_prob > 0.04, 1, 0)
print metrics.confusion_matrix(y_test, y_pred)
print metrics.roc_auc_score(y_test, y_pred_prob) #Went down slightly to .94352
