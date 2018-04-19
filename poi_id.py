#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import GridSearchCV, train_test_split, KFold, \
                                    cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import svm

from tester import dump_classifier_and_data
from collections import Counter
from time import time

### Task 1: Select what features you'll use.
features_list = ['poi','person_to_poi_ratio',
    'shared_receipt_with_poi','salary', 'expenses', 'total_stock_value']

    ### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

## Deciding to set any NaN and values under 0 to 0 since they are assumed
## to have 0 in total_stock_value and salary
def NANO(feature):
    for person in data_dict:
        if data_dict[person][feature] == 'NaN' or \
        data_dict[person][feature] < 0:
            data_dict[person][feature] = 0
NANO('salary')
NANO('expenses')
NANO('total_stock_value')


for person in data_dict:
    if data_dict[person]['to_messages'] == 'NaN':
        for features in ('to_messages', 'from_messages',
        'from_poi_to_this_person', 'from_this_person_to_poi'):
            data_dict[person][features] = 0

### Task 3: Create new feature(s)
for i in data_dict:
    if data_dict[i]['to_messages'] == 0:
        data_dict[i].update({'person_to_poi_ratio': 0})
        data_dict[i].update({'poi_to_person_ratio': 0})
    else:
        sent_ratio = float(data_dict[i]['from_this_person_to_poi'])/float(data_dict[i]['from_messages'])
        received_ratio = float(data_dict[i]['from_poi_to_this_person'])/float(data_dict[i]['to_messages'])
        data_dict[i].update({'person_to_poi_ratio': sent_ratio})
        data_dict[i].update({'poi_to_person_ratio': received_ratio})

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#clf = GaussianNB()
#clf = DecisionTreeClassifier()
#clf = svm.SVC()

#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)

#acc = accuracy_score(labels_test, pred)
#print("The accuracy is: {}").format(acc)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation.

### Grid Search SVC
#C_Range = np.logspace(-3, 10, 10)
#Gamma_Range = np.logspace(-3, 10, 10)
#parameters = {'C':C_Range, 'gamma':Gamma_Range}
#svector = svm.SVC(kernel = 'rbf')

#clf = GridSearchCV(svector, parameters)
#clf.fit(featuers_train, labels_train)
#pred = clf.predict(features_test)
#print clf.best_score_
#print clf.best_params_

### Grid Search Decision Tree
#parameter = {'min_samples_split':range(2,11)}
#decisiontree = DecisionTreeClassifier()

#clf = GridSearchCV(decisiontree, parameter)
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#print clf.best_score_
#print clf.best_params_

### Testing different classifiers with different parameters
#clf = GridSearchCV(svector, parameters)
#clf = svm.SVC(kernel = 'rbf', C = 10)
clf = DecisionTreeClassifier(min_samples_split = 5)
#clf = KMeans(n_clusters = 2)

### Testing each precision/recall for each fold
#kf=StratifiedKFold(n_splits = 3, random_state=14841, shuffle = False)
#for train_index, test_index in kf.split(features, labels):
#    features_train= [features[x] for x in train_index]
#    features_test= [features[x] for x in test_index]
#    labels_train= [labels[x] for x in train_index]
#    labels_test= [labels[x] for x in test_index]
#    clf.fit(features_trian, labels_train)
#    pred = clf.predict(features_test)

#    acc = accuracy_score(labels_test, pred)
#    prec = precision_score(labels_test, pred)
#    rec = recall_score(labels_test, pred)
#    print('Accuracy: {} -- Precision: {} -- Recall:{}').format(acc, prec, rec)


clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
prec = precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)

print('Accuracy: {} -- Precision: {} -- Recall:{}').format(acc, prec, rec)
#print cross_val_score(clf, features, labels, cv=kf, n_jobs=1)

importances = clf.feature_importances_
k = len(features_list)-1
top_k = importances.argsort()[-k:][::-1]
print("Features Importance List:")
for itm in top_k:
    print("{}: {}").format(features_list[itm+1], importances[itm])

#print clf.best_score_
#print clf.best_params_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
