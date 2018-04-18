# Udacity Data Analyst Nanodegree
## Machine Learning project

1. Summarize for us the goal of this project and how machine learning is useful
in trying to accomplish it. As part of your answer, give some background on the
dataset and how it can be used to answer the project question. Were there any
outliers in the data when you got it, and how did you handle those?  [relevant
rubric items: “data exploration”, “outlier investigation”]
    - The goal of this project is to discover who were the persons of Interest
in the Enron dataset. Machine learning is useful in this instance because we
are able to create a model that can iterate over many variables to most
accurately label someone as a Person of Interest(POI) or not. The dataset
contains 146 points, 18 of which are POIs while 128 are non-POIs. For each
datapoint, there exists 21 features (initially).
    - However, not all are good values, meaning that some 'features' actually
have no value or 'NaN' values.These will mess up with our model. To address
this issue, we want to tackle outliers. The first obvious outlier was the
'Total' column which summed all the salaries and other financial information
together. Another outlier was the 'TRAVELING AGENCY' which introduced
unnecessary information. Other outliers included missing values for all the
features initially included. It ranged from 21 missing values for the
'total_stock_value' to 142 missing values for 'loan_advances'. Since I did not
use all the features, I did not need to handle all the NaNs. However for most
NaNs for features that were used, they were simply turned to zeroes. This was
done because our data set of 146 points is so small and removing points would
drastically affect our results. Another way would have been to turn it to
averages but setting POI results and NonPOI results to the average would also
muddle the results. A finer way would have been to set the NaNs as averages of
each POI or NonPOI. That way, POIs and NonPOIs do not have the save average.
However, the simplest way was to set each to 0.

2. What features did you end up using in your POI identifier, and what
selection process did you use to pick them? Did you have to do any scaling? Why
or why not? As part of the assignment, you should attempt to engineer your own
feature that does not come ready-made in the dataset -- explain what feature
you tried to make, and the rationale behind it. (You do not necessarily have to
use it in the final analysis, only engineer and test it.) In your feature
selection step, if you used an algorithm like a decision tree, please also give
the feature importances of the features that you use, and if you used an
automated feature selection function like SelectKBest, please report the
feature scores and reasons for your choice of parameter values.  [relevant
rubric items: “create new features”, “intelligently select features”, “properly
scale features”]
    - In the end of the process, I used 5 features, 1 of which was newly
created. Since I was using a decision tree to classify, I did not need to
feature scale since they are not affected by scales. I chose my features based
on how many missing values they had. I started off with only 'total_stock_value'
which of course was not enough. I then worked my way up the list, hand selecting
features to add. A very noticeable group was 'to_messages', 'from_messages' and  
the like. It was interesting that they all had the same amount of points so I
decided to work with that. The values did not end up being what I wanted so I
decided to work around them and create a ratio of their sent/received messages
to/from POIs instead of taking a flat number. People could have sent a small
amount but only been talking to POIs or sent a large number but also have a
large amount of communications with other people.
    - With just the 'total_stock_value', 'shared_receipt_with_poi', and my two
ratios, I was not getting the precision and recall in tester.py so I wanted to
select more features. I again looked for a group of features that could work
well together. Similar to the email messages, 'salary' and 'expenses' both had
51 missing values so they could easily be worked with. Using these 6, I only
sometimes got the target I wanted (since decision trees split in random places).
I wanted to consistently hit above my target so I used the D.T. Classifier's
attribute of feature_importances_ which showed me that one of my ratios was
not performing as well as I wanted to, going as far as showing 0.0 for
importance. When I removed it, my precision and recall was well above my target
and my accuracy was hitting 84%. Albeit, this is lower than I would like for
accuracy, I think it is pretty good for the amount of training data we had.
  - Feature Importances: (one of the many results)
```
     total_stock_value: 0.361970853574
     person_to_poi_ratio: 0.232689761239
     expenses: 0.168854961832
     shared_receipt_with_poi: 0.133949173833
     salary: 0.102535249521
```
3. What algorithm did you end up using? What other one(s) did you try? How did
model performance differ between algorithms?  [relevant rubric item: “pick an
algorithm”]
   - I ended up using the DecisionTreeClassifier to create a model. I started
off with using GaussianNB and compared it with the DecisionTreeClassifier,
SVC, and KMeans. Initially, I ended up choosing the SVC since it had several
parameters I could tune and also had a high initial accuracy. However, I came
to realize that even with a high accuracy, my precision and recall were not
doing well. With a dataset heavily skewed towards one data point (specifically
non POIs), it would classify them all as non POIs and get an accuracy of 90%+.
I then reverted back to a decision tree classifier when I started to look for
precision and recall in the initial models. Looking back, I could have applied
feature scaling to the SVC to see if I could have improved it but the
DecisionTreeClassifier got me what I needed.

4. What does it mean to tune the parameters of an algorithm, and what can happen
if you don’t do this well?  How did you tune the parameters of your particular 
algorithm? What parameters did you tune? (Some algorithms do not have parameters
that you need to tune -- if this is the case for the one you picked, identify
and briefly explain how you would have done it for the model that was not your
final choice or a different model that does utilize parameter tuning, e.g. a
decision tree classifier).  [relevant rubric items: “discuss parameter tuning”,
“tune the algorithm”]
   - Tuning parameters are an important part of machine learning as these, along
with proper features, are keys to improving the algorithm. Tuning parameters
alters the ways, for example, a classifier classifies each point. Tuning changes
the thresholds for the model. If you did not tune and just stuck with the base
algorithm, while you could get a decent score, it would not be the most
optimzied solution.
  - For the DecisionTreeClassifier, I stuck with tuning their minimum samples
split. I stuck with the gini criterion since we want to minimize
misclassifications. We did not want to limit max_depth since we want all
all our leaves to be pure, especially with a small dataset.
