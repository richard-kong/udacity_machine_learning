#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from tester import test_classifier, dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit



def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    if poi_messages == "NaN" or all_messages == "Nan":
        fraction = 0
    else:
        fraction = float(poi_messages) / float(all_messages)



    return fraction
    

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list =  ['poi', 'salary', 'to_messages',  'total_payments', 'bonus',  'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'restricted_stock']


#final feature list
features_list =  ['poi',  'expenses', 'exercised_stock_options', 'other', 'shared_receipt_with_poi']



### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

#univariate feature selection


### Task 2: Remove outliers
data_dict.pop("TOTAL",0)
#to do, only remove outliers for features used

#has_salary_and_bonus = 0 = 81
### Task 3: Create new feature(s)
for name in data_dict:
    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )


    data_point["from_poi_to_this_person"] = fraction_from_poi
    data_point["from_this_person_to_poi"] = fraction_to_poi
    
        




### Store to my_dataset for easy export below.
my_dataset = data_dict
#features_list.append('fraction_bonus_to_salary' )

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
#from sklearn.naive_bayes import GaussianNB
from sklearn import tree
 
#clf = GaussianNB()    #not as accurate os decision tree
#
clf = tree.DecisionTreeClassifier()



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#decisiontree
#clf = tree.DecisionTreeClassifier(min_samples_split = 2,criterion = "entropy")
#clf = tree.DecisionTreeClassifier(criterion = "entropy")
clf = tree.DecisionTreeClassifier(min_samples_split = 4)

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)