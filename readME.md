# Udacity Data Analyst Nanodegree
## Machine Learning project

This is a project for the Udacity Data Analyst Nanodegree in which we
utilized machine learning techniques to determine whether a person
is a person on interest in the Enron case.

## Contents

poi_id.py - The model for determining POIs.

Responses.pdf - Answers to the series of questions

Other files are necessary to make the identifier run. 

## Works Cited
https://stackoverflow.com/questions/18265935/python-create-list-with-numbers-between-2-values

https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html

https://docs.scipy.org/doc/numpy-1.8.1/reference/generated/numpy.logspace.html#numpy.logspace

http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0

https://stackoverflow.com/questions/28501072/how-to-check-which-version-of-nltk-scikit-learn-installed

https://stackoverflow.com/questions/31155813/getting-feature-names

https://stats.stackexchange.com/questions/269300/why-does-sklearn-grid-search-gridsearchcv-return-random-results-on-every-executi

https://stackoverflow.com/questions/40159161/feature-importance-vector-in-decision-trees-in-scikit-learn-along-with-feature-n

https://en.wikipedia.org/wiki/Precision_and_recall

https://stackoverflow.com/questions/1024847/add-new-keys-to-a-dictionary

## Working Notes
Data Exploration:
* Total Number of Data Points:
```python
print(len(data_dict.keys()))M
```
  - 146
* Allocation Across Classes:
``` python
poi_yes = 0
poi_no = 0
for x in range(0, len(data_dict)):
    if data_dict[data_dict.keys()[x]]['poi'] == True:
        poi_yes += 1
    else:
        poi_no += 1
print poi_yes
print poi_no
```
  - POI: 18
  - nonPOI: 128
* Number of Features Used (initial)
```python
len(data_dict[data_dict.keys()[0]].keys())
```
  - 21 Features initially
* Features with missing values
```python
nan_dump = []
for x in range(0, len(data_dict)):
    for i in range(0, len(data_dict[data_dict.keys()[x]])):
        if data_dict[data_dict.keys()[x]][data_dict[data_dict.keys()[x]].keys()[i]] == 'NaN':
            nan_dump.append(data_dict[data_dict.keys()[x]].keys()[i])
print(Counter(nan_dump))
```
  - ```Counter({'loan_advances': 142, 'director_fees': 129, 'restricted_stock_deferred': 128, 'deferral_payments': 107, 'deferred_income': 97, 'long_term_incentive': 80, 'bonus': 64, 'to_messages': 60, 'shared_receipt_with_poi': 60, 'from_poi_to_this_person': 60, 'from_messages': 60, 'from_this_person_to_poi': 60, 'other': 53, 'salary': 51, 'expenses': 51, 'exercised_stock_options': 44, 'restricted_stock': 36, 'email_address': 35, 'total_payments': 21, 'total_stock_value': 20})```


```python
averaging = 0
counter = 0
lowest = 6000000
max = 0
for x in range(0, len(data_dict.keys())):
    if data_dict[data_dict.keys()[x]]['total_stock_value'] != 'NaN':
        averaging += data_dict[data_dict.keys()[x]]['total_stock_value']
        counter += 1
    if data_dict[data_dict.keys()[x]]['total_stock_value'] <= lowest and \
    data_dict.keys()[x] != 'BELFER ROBERT':
        lowest = data_dict[data_dict.keys()[x]]['total_stock_value']
    if data_dict[data_dict.keys()[x]]['total_stock_value'] >= max and data_dict[data_dict.keys()[x]]['total_stock_value'] != 'NaN':
        max = data_dict[data_dict.keys()[x]]['total_stock_value']
print averaging/counterM
print lowestM
print maxM

print '~~~~'

for x in range(0, len(data_dict.keys())):
    if data_dict[data_dict.keys()[x]]['total_stock_value'] != 'NaN' and \
    data_dict[data_dict.keys()[x]]['total_stock_value'] < 0:
        print data_dict.keys()[x]
```

I also noticed that there are a set of 60 missing information and that
all relates to the persons email habits. I ran a quick check
```python
words = ('poi','to_messages',
'shared_receipt_with_poi',
'from_poi_to_this_person',
'from_messages',
'from_this_person_to_poi')

for x in range(0, len(data_dict)):
    for feature in words:
        if data_dict[data_dict.keys()[x]][feature] == 'NaN':
            data_dict[data_dict.keys()[x]][feature] = 0
    print(data_dict[data_dict.keys()[x]][words[0]],
              data_dict[data_dict.keys()[x]][words[1]],
              data_dict[data_dict.keys()[x]][words[2]],
              data_dict[data_dict.keys()[x]][words[3]],
              data_dict[data_dict.keys()[x]][words[4]]
             )
```

```python
TrueNum = 0
FalseNum = 0
TrueNaN = 0
FalseNaN = 0
for i in data_dict:
    if data_dict[i]['poi'] == True:
        if data_dict[i]['to_messages'] == 'NaN':
            TrueNaN +=1
        else:
            TrueNum += 1
    else:
        if data_dict[i]['to_messages'] == 'NaN':
            FalseNaN += 1
        else:
            FalseNum += 1
print(TrueNum, FalseNum, TrueNaN, FalseNaN)
```
From this, we get a result of
* POIs with values : 14
* POIs with NaNs: 4
* nonPOIs with values: 72
* nonPOIs with NaNs: 56
