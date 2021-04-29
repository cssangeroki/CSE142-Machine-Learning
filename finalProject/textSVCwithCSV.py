import time
import csv
import json
import sys
import random
import re
from ourStopWords import stopWordList2
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm

# Fills feature and label parameters with data
# features and labels must be lists.
def getFeaturesAndLabels(jsonFile, features, labels):
	with open(jsonFile) as f:
		rawList = json.load(f)
	for index in rawList:
		# features is a list of strings
		features.append(index["text"])
		# labels is a list of ints(1-5)
		labels.append(index["stars"])


def getFeatures(jsonFile, features):
        with open(jsonFile) as f:
            rawList = json.load(f)
        for index in rawList:
            # features is a list of strings
            features.append(index["text"]) 
#################################
#		Load Data
#################################

jsonFileTrain = sys.argv[1]
jsonFileTest = sys.argv[2]

# get features and labels for train data
train_x = []
train_y = []
getFeaturesAndLabels(jsonFileTrain, train_x, train_y)

# get features and labels for test data
test_x = []
getFeatures(jsonFileTest, test_x)

#################################
#	Modify Data
#	lowercase text
#   Remove punctuation, special characters and numbers
#	Stem
#################################

# Custom tokenizer that will get passed to tfidfVectorizer
# Takes String and returns processed list of words
def kevinTokenizer(strInput):
	cleanedWords = re.sub('\W+', ' ', strInput).lower().split()
	stemmer = PorterStemmer()
	stemmedWords = [stemmer.stem(word) for word in cleanedWords if not word in stopWordList2]
	return stemmedWords

#################################
#		Prep Data
#################################

# Create vectorizer
vizer = TfidfVectorizer(tokenizer=kevinTokenizer, min_df=20)

# Bag of words quantifier for test_x
# -- Transform data to fit training model & create feature vector
train_x_vector = vizer.fit_transform(train_x)
test_x_vector = vizer.transform(test_x)

#################################
#	Train and Classify -- SVM
#################################

svmClf = svm.SVC(kernel='linear')
svmClf.fit(train_x_vector, train_y)


#################################
#        CSV Output File
#################################
test_y_svm = ['Predictions']
for i in svmClf.predict(test_x_vector):
    test_y_svm.append(float(i))

#Writes values from svm prediction
wtr = csv.writer(open ('predictions.csv', 'w'), delimiter=';', lineterminator='\n')
for x in test_y_svm : 
    wtr.writerow ([x]) 

print('CSV done printing')

'''
#################################
#		Evaluate
#################################

# Predict on unseen data test_x
print(classification_report(test_y, svmClf.predict(test_x_vector)))
'''
