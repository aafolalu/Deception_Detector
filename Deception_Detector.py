# coding: utf-8

import csv                               # csv reader
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import average_precision_score, recall_score, accuracy_score, f1_score




# load data from a file and append it to the rawData

def loadData(path, Text=None):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            (Id, Text, Label) = parseReview(line)
            rawData.append((Id, Text, Label))
            preprocessedData.append((Id, preProcess(Text), Label))


def splitData(percentage):
    dataSamples = len(rawData)
    halfOfData = int(len(rawData)/2)
    trainingSamples = int((percentage*dataSamples)/2)
    for (_, Text, Label) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
        trainData.append((toFeatureVector(preProcess(Text)),Label))
    for (_, Text, Label) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
        testData.append((toFeatureVector(preProcess(Text)),Label))



################
## QUESTION 1 ##
################

# Convert line from input file into an id/text/label tuple; index:0/8/1
def parseReview(reviewLine):
    # Should return a triple of an integer, a string containing the review, and a string indicating the label
    list = []
    Id, Text, Label = reviewLine[0], reviewLine[8], reviewLine[1]
    list.append(Id)
    list.append(Text)
    list.append(Label)
    tup = tuple(list)
    return tup


# TEXT PREPROCESSING AND FEATURE VECTORIZATION


# Input: a string of one review
def preProcess(text):
    # Should return a list of tokens
    vectorizer = CountVectorizer( min_df=1 )  # instance CountVectorizer class
    X = vectorizer.fit_transform([text]) #extract bag of words representation
    Y = vectorizer.get_feature_names() #output list of tokens
    return Y



################
## QUESTION 2 ##
################
featureDict = {} # A global dictionary of features

def toFeatureVector(tokens):
    # Should return a dictionary containing features as keys, and weights as values
    d = {x: tokens.count(x) for x in tokens}
    featureDict.update(d)
    return d


# TRAINING AND VALIDATING OUR CLASSIFIER
def trainClassifier(trainData):
    print("Training Classifier...")
    pipeline =  Pipeline([('svc', LinearSVC())])
    return SklearnClassifier(pipeline).train(trainData)



################
## QUESTION 3 ##
################

def crossValidate(dataset, folds):
    shuffle(dataset)
    cv_results = []
    foldSize = len(dataset)/folds
    for i in range(0,len(dataset),int(foldSize)):
        #trains and tests on the 10 folds of data in the dataset
        val_data = dataset[i:i+int(foldSize)]
        left_over = dataset[0:i] + dataset[i+int(foldSize):]
        class1 = trainClassifier(left_over)
        print(val_data)
        val, y = zip(*val_data)
        pred = predictLabels(val_data, class1)
        print(class1)
        print(pred)

        #model performance metrics
        acc = accuracy_score(y, pred)
        #avg_pre = average_precision_score(y, pred)
        #recall_1 = recall_score(y, pred, average=None)
        #f1 = f1_score(y, pred, average=None)
        cv_results.append(acc)
        #cv_results.append(recall_1)
        #cv_results.append(avg_pre)
        #cv_results.append(f1)
    return cv_results


# PREDICTING LABELS GIVEN A CLASSIFIER

def predictLabels(reviewSamples, classifier):
    return classifier.classify_many(map(lambda t: t[0], reviewSamples))


def predictLabel(reviewSample, classifier):
    return classifier.classify(toFeatureVector(preProcess(reviewSample)))




# MAIN

# loading reviews
rawData = []          # the filtered data from the dataset file (should be 21000 samples)
preprocessedData = [] # the preprocessed reviews (just to see how your preprocessing is doing)
trainData = []        # the training data as a percentage of the total dataset (currently 80%, or 16800 samples)
testData = []         # the test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# the output classes
fakeLabel = 'fake'
realLabel = 'real'

# references to the data files
reviewPath = '/Users/adebisiafolalu/Desktop/OneDrive/Documents/MSc_Assignments/Semester2/NLP/assignment1/Deception_Detector/amazon_reviews.txt'

## Do the actual stuff
# We parse the dataset and put it in a raw data list
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing the dataset...",sep='\n')
loadData(reviewPath)
# We split the raw dataset into a set of training data and a set of test data (80/20)
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing training and test data...",sep='\n')
splitData(0.8)
# We print the number of training samples and the number of features
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Training Samples: ", len(trainData), "Features: ", len(featureDict), sep='\n')

folds = 10
cv_results = crossValidate(trainData, folds)
print(cv_results)

