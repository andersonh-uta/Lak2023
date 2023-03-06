"""
Version of the notebook `01a Bag of Words with scikit-learn.ipynb`,
converted to a .py script.
"""

import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

# load the data
train = pd.read_csv("../../data/train.csv")
test = pd.read_csv("../../data/test.csv")

def fit_and_score(clf, train, test):
    """fit the model `clf` to the `train` dataset and evaluate its
    performance on the `test` dataset."""
    clf.fit(train["review_body"], train["stars"])
    preds = clf.predict(test["review_body"])
    
    # calculate some classification metrics
    accuracy = metrics.accuracy_score(preds, test["stars"])
    f1 = metrics.f1_score(preds, test["stars"], average="macro")

    # and some regression metrics (since "predict the number of stars"
    # could reasonably be either kind of task).
    r2 = metrics.r2_score(preds, test["stars"])
    mae = metrics.mean_absolute_error(preds, test["stars"])
    
    return pd.Series({"Accuracy": accuracy, "F1": f1, "R2": r2, "MAE": mae})


# This is it--this is our pipeline.  CountVectorizer--dropping words
# that appears in >50% of our documents or <10 documents--followed
# by a Bernoulli Naive Bayes model.  Super simple, and super fast.
classifier = Pipeline([
    ("bag of words", CountVectorizer(max_df=0.5, min_df=10)),
    ("clf", BernoulliNB()),
])
res = fit_and_score(
    classifier,
    train,
    test,
).rename("Bag of Words + Linear kernel SVM")
print(res)
print()

# fit a dummy classifier to check how much better than a random guess
# we are.
classifier = GridSearchCV(
    DummyClassifier(),
    param_grid={
        "strategy": [
            "most_frequent",
            "prior",
            "stratified",
            "uniform"
        ]
    }
)
res = fit_and_score(
    classifier,
    train,
    test,
).rename("Dummy Classifier")
print(res)