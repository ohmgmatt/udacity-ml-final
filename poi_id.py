#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm

from tester import dump_classifier_and_data
from collections import Counter
from time import time

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_stock_value', 'person_to_poi_ratio',
'poi_to_person_ratio']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Figure out the length of the my_dataset
#print(len(data_dict.keys())) --146



### Task 2: Remove outliers
#for x in range(0,len(data_dict.keys())):
#    if data_dict[data_dict.keys()[x]]['total_stock_value'] == 'NaN':
#        print(data_dict.keys()[x],':',data_dict[data_dict.keys()[x]]['poi'])
data_dict.pop('TOTAL')

## Deciding to set any NaN and values under 0 to 0 since they are assumed
## to have 0 in total_stock_value
for x in range(0, len(data_dict.keys())):
    if data_dict[data_dict.keys()[x]]['total_stock_value'] == 'NaN' or \
    data_dict[data_dict.keys()[x]]['total_stock_value'] < 0:
        data_dict[data_dict.keys()[x]]['total_stock_value'] = 0
    if data_dict[data_dict.keys()[x]]['to_messages'] == 'NaN':
        data_dict[data_dict.keys()[x]]['to_messages'] = 0
        data_dict[data_dict.keys()[x]]['from_messages'] = 0
        data_dict[data_dict.keys()[x]]['from_poi_to_this_person'] = 0
        data_dict[data_dict.keys()[x]]['from_this_person_to_poi'] = 0



### Task 3: Create new feature(s)
###https://stackoverflow.com/questions/1024847/add-new-keys-to-a-dictionary
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
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


parameter = {'min_samples_split':range(2,11)}
decisiontree = DecisionTreeClassifier()

#clf = GridSearchCV(decisiontree, parameter, cv = 2)
#t0 = time()
#clf.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
#pred = clf.predict(features_test)
#print "predict time:", round(time()-t0, 3), "s"
#print clf.best_score_
#print clf.best_params_

C_Range = np.logspace(-3, 10, 10)
Gamma_Range = np.logspace(-3, 10, 10)
parameters = {'C':C_Range, 'gamma':Gamma_Range}
svector = svm.SVC(kernel = 'rbf')

clf = GridSearchCV(svector, parameters)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"


t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"

print(accuracy_score(labels_test, pred))
print clf.best_score_
print clf.best_params_



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
