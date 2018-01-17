#Project Name: YouTube Personality Detection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import *
stemmer = PorterStemmer()
transformer = TfidfTransformer()

#This function is to plot a hist graph to show the first ten feature importances
def plot_hist(rf):
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(10), importances[indices[0:10]], color="r", align="center")
    aft_indices = []
    for ii in df2.columns[5 + indices[0:10]]:
        if type(ii) == int:
            aft_indices.append(ngramCount.get_feature_names()[ii])
        else:
            aft_indices.append(ii)
    plt.xticks(range(10), aft_indices, rotation=45)
    plt.xlim([-1, 10])
    plt.show()
#read the label as well as all the non-text features of the training and testing set
filepath = "dataset/"
personality_impression_scores = pd.read_csv(filepath + "YouTube-Personality-Personality_impression_scores.csv", sep=' ', index_col=0)
audiovisual_features = pd.read_csv(filepath + "YouTube-Personality-audiovisual_features.csv", sep=' ', index_col=0)
gender = pd.read_csv(filepath + "YouTube-Personality-gender.csv", sep=' ', index_col=0)

#save the label as well as all the non-text features in a Pandas dataframe
df = pd.concat([personality_impression_scores, audiovisual_features, gender], axis=1)

#encode the gender feature
encoder = LabelEncoder()
df.gender = encoder.fit_transform(df.gender)

#read the transcripts
result = []
for i in df.index:
    with open(filepath + 'transcripts/{}.txt'.format(i)) as f:
        text = f.read()
    result.append(text)

#use Potter Stemmer for preprocessing
stemmed_result = []
for article in result:
    stemmed_result.append(" ".join([stemmer.stem(ii.decode('utf-8')) for ii in article.split()]))

#tokenize the transcripts and generate a collection of unigrams and bigrams
ngramCount = CountVectorizer(ngram_range=(1,2), lowercase=True, min_df=3)

#encode the stemmed transcripts
text = ngramCount.fit_transform(stemmed_result)

#use tf-idf method to adjust the values of the sparse matrix of encoded transcripts and fit it into a Pandas dataframe
text_tf = transformer.fit_transform(text)
text_df = pd.DataFrame(text_tf.todense())
text_df = text_df.set_index(df.index)

#concatenate the dataframe of text features and non-text features
df2 = pd.concat([df, text_df], axis = 1)


#use all features
print "--------------------------------------------------------------"
print "All Features"
error1_lasso = []
error1_svm = []
error1_rf = []
for personality in ['Extr', 'Agr', 'Cons', 'Emot', 'Open']:
    clf = svm.SVR()
    lr = Lasso(alpha=0.1)
    rf = RandomForestRegressor()
    #split the data into training set and test set, and the ratio is 75:25 for training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(df2.iloc[:, 5:], df2[personality], random_state=1)

    lr.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    rf.fit(X_train, y_train)
   # plot_hist(rf)      #plot a hist graph of feature importance of random forest if you wish

    y_pred = lr.predict(X_test)
    y_pred_svm = clf.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    error1_lasso.append(mean_absolute_error(y_pred, y_test))
    error1_svm.append(mean_absolute_error(y_pred_svm, y_test))
    error1_rf.append(mean_absolute_error(y_pred_rf, y_test))
    print (
    "The mean absolute error for " + personality + " using Lasso is: " + str(mean_absolute_error(y_pred, y_test)))
    print ("The mean absolute error for " + personality + " using SVM is: " + str(mean_absolute_error(y_pred_svm, y_test)))
    print ("The mean absolute error for " + personality + " using random forest is: " + str(
        mean_absolute_error(y_pred_rf, y_test)))
    print ('\n')

#use non-text features
print "--------------------------------------------------------------"
print "Non-text Features"
error2_lasso = []
error2_svm = []
error2_rf = []
for personality in ['Extr', 'Agr', 'Cons', 'Emot', 'Open']:
    clf = svm.SVR()
    lr = Lasso(alpha=0.1)
    rf = RandomForestRegressor()
    # split the data into training set and test set, and the ratio is 75:25 for training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(df2.loc[:, list(audiovisual_features.columns.values) + ['gender']], df2[personality], random_state=1)

    lr.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    #plot_hist(rf)      #plot a hist graph of feature importance of random forest if you wish

    y_pred = lr.predict(X_test)
    y_pred_svm = clf.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    error2_lasso.append(mean_absolute_error(y_pred, y_test))
    error2_svm.append(mean_absolute_error(y_pred_svm, y_test))
    error2_rf.append(mean_absolute_error(y_pred_rf, y_test))
    print ("The mean absolute error for " + personality + " is: " + str(mean_absolute_error(y_pred, y_test)))
    print ("The mean absolute error for " + personality + " using SVM is: " + str(mean_absolute_error(y_pred_svm, y_test)))
    print ("The mean absolute error for " + personality + " using random forest is: " + str(mean_absolute_error(y_pred_rf, y_test)))
    print ('\n')

#use text features
print "--------------------------------------------------------------"
print "Text Features"
error3_lasso = []
error3_svm = []
error3_rf = []
for personality in ['Extr', 'Agr', 'Cons', 'Emot', 'Open']:
    clf = svm.SVR()
    lr = Lasso(alpha=0.1)
    rf = RandomForestRegressor()

    # split the data into training set and test set, and the ratio is 75:25 for training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(df2.loc[:, range(0, text.shape[1])], df2[personality], random_state=1)

    lr.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    #plot_hist(rf)          #plot a hist graph of feature importance of random forest if you wish

    y_pred = lr.predict(X_test)
    y_pred_svm = clf.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    error3_lasso.append(mean_absolute_error(y_pred, y_test))
    error3_svm.append(mean_absolute_error(y_pred_svm, y_test))
    error3_rf.append(mean_absolute_error(y_pred_rf, y_test))
    print ("The mean absolute error for " + personality + " is: " + str(mean_absolute_error(y_pred, y_test)))
    print ("The mean absolute error for " + personality + " using SVM is: " + str(mean_absolute_error(y_pred_svm, y_test)))
    print ("The mean absolute error for " + personality + " using random forest is: " + str(mean_absolute_error(y_pred_rf, y_test)))
    print ('\n')

print "--------------------------------------------------------------"
print "Errors of Lasso Algorithm"
errors_lasso = pd.DataFrame([error1_lasso,error2_lasso, error3_lasso], index=['all features', 'audio+gender', 'text only'],columns=personality_impression_scores.columns)
print errors_lasso
print '\n'

print "--------------------------------------------------------------"
print "Errors of SVM Algorithm"
errors_svm = pd.DataFrame([error1_svm,error2_svm, error3_svm], index=['all features', 'audio+gender', 'text only'],columns=personality_impression_scores.columns)
print errors_svm
print '\n'

print "--------------------------------------------------------------"
print "Errors of Random Forest Algorithm"
errors_rf = pd.DataFrame([error1_rf,error2_rf, error3_rf], index=['all features', 'audio+gender', 'text only'],columns=personality_impression_scores.columns)
print errors_rf
print '\n'

print "--------------------------------------------------------------"
print "Mean Errors of Lasso Algorithm"
print errors_lasso.mean(axis=1)
print '\n'

print "--------------------------------------------------------------"
print "Mean Errors of SVM Algorithm"
print errors_svm.mean(axis=1)
print '\n'

print "--------------------------------------------------------------"
print "Mean Errors of Random Forest Algorithm"
print errors_rf.mean(axis=1)
print '\n'

#this part is to use the text feature model to predict the Big Five score of Donald Trump's speech
print "--------------------------------------------------------------"
print "Application: using the text feature model to predict the Big Five score of Donald Trump's speech"
with open('trump_speech/speech.txt') as f:
    text_test = f.read()
stemmed_test = (" ".join([stemmer.stem(ii.decode('utf-8')) for ii in text_test.split()]))
encoded_text = ngramCount.transform([stemmed_test])

for personality in ['Extr', 'Agr', 'Cons', 'Emot', 'Open']:
    rf = RandomForestRegressor()
    X_train, X_test, y_train, y_test = train_test_split(df2.loc[:, range(0, text.shape[1])], df2[personality], random_state=1)
    rf.fit(X_train, y_train)
    test_pred_rf = rf.predict(encoded_text)
    print "The predicted",personality,"score for Donald Trump speech using Random Forest is", test_pred_rf[0]