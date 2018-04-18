# Udacity Data Analyst Nanodegree
## Machine Learning project

1. Summarize for us the goal of this project and how machine learning is useful
in trying to accomplish it. As part of your answer, give some background on the
dataset and how it can be used to answer the project question. Were there any
outliers in the data when you got it, and how did you handle those?  [relevant
rubric items: “data exploration”, “outlier investigation”]
  * The goal of this project is to discover who were the persons of Interest
in the Enron dataset. Machine learning is useful in this instance because we
are able to create a model that can iterate over many variables to most
accurately label someone as a Person of Interest(POI) or not. The dataset
contains 146 points, 18 of which are POIs while 128 are non-POIs. For each
datapoint, there exists 21 features (initially).
  * However, not all are good values, meaning that some 'features' actually
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
  * In the end of the process, I used 3 features, 2 of which were newly created.
