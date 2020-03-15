import pandas as pd

masses_data = pd.read_csv('mammographic_masses.data.txt')
# print(masses_data.head())

##########DATA PREPROCESSING#############

## as we can see in the data there are some ? values
# and there are no column labels

##Adding column labels and changing ? values to NaN
masses_data = pd.read_csv('mammographic_masses.data.txt', na_values=['?'], names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
# print(masses_data.head())

## as we can see density column doesent have all the values


# print(masses_data.describe())

# print(masses_data.loc[(masses_data['age'].isnull()) |
#               (masses_data['shape'].isnull()) |
#               (masses_data['margin'].isnull()) |
#               (masses_data['density'].isnull())])

## as we can see the Nan are randomly distributed so drop them

masses_data.dropna(inplace=True)
# print(masses_data.describe())


##converting the pandas dataframe to numpy so that scikit learn
all_features = masses_data[['age', 'shape',
                             'margin', 'density']].values


all_classes = masses_data['severity'].values

feature_names = ['age', 'shape', 'margin', 'density']
# print(all_features)
# print(all_classes)
# print(feature_names)

##normalising the data as some of our models require normalised data
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)
# print(all_features_scaled)


import numpy
from sklearn.model_selection import train_test_split

#using train test with train block_size as 75%
# numpy.random.seed(1234)

(training_inputs,testing_inputs,training_classes,testing_classes) = train_test_split(all_features_scaled, all_classes, train_size=0.75, random_state=1)

#############DECISION TREES###################

from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier(random_state=1)

# Train the classifier on the training set
clf.fit(training_inputs, training_classes)

# print(clf.score(testing_inputs, testing_classes))

from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(random_state=1)
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
# print(cv_scores.mean())


########RANDOM FOREST CLASSIFIER#########
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, random_state=1)
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
# print(cv_scores.mean())


##############SUPPORT VECTOR MACHINES###############

from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear', C=C)
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)

# print(cv_scores.mean())



##############K NEAREST NEIGHBOURS#########
from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=10)
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)

# print(cv_scores.mean())


## for choosing the number of neighbours of the classifier try a few values for k
## find the corresponding cv_score and choose the best fit but do not overfit your data.

# for n in range(1, 50):
#     clf = neighbors.KNeighborsClassifier(n_neighbors=n)
#     cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
#     print (n, cv_scores.mean())

###########NAIVE BAYES##############

from sklearn.naive_bayes import MultinomialNB

scaler = preprocessing.MinMaxScaler()
all_features_minmax = scaler.fit_transform(all_features)

clf = MultinomialNB()
cv_scores = cross_val_score(clf, all_features_minmax, all_classes, cv=10)

# print(cv_scores.mean())



###########LOGISTIC REGRESSION############

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
# print(cv_scores.mean())
