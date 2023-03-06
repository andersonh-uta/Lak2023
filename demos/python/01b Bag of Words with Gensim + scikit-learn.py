"""
Version of the notebook `01b Bag of Words with Gensim + scikit-learn.ipynb`,
converted to a .py script.
"""

from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from gensim.parsing import preprocessing as pre
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from tqdm import tqdm

# register tqdm with pandas so we can get .progress_apply() method
# added to dataframes.  This is a version of pd.DataFrame.apply()
# but now it prints a progress bar!
tqdm.pandas(smoothing=0)

# load the data
train = pd.read_csv("../../data/train.csv")
test = pd.read_csv("../../data/test.csv")

def fit_and_score(clf, train_x, train_y, test_x, test_y):
    """fit the model `clf` to the `train` dataset and evaluate its
    performance on the `test` dataset."""
    clf.fit(train_x, train_y)
    preds = clf.predict(test_x)
    
    # calculate some classification metrics
    accuracy = metrics.accuracy_score(preds, test_y)
    f1 = metrics.f1_score(preds, test_y, average="macro")

    # and some regression metrics (since "predict the number of stars"
    # could reasonably be either kind of task).
    r2 = metrics.r2_score(preds, test_y)
    mae = metrics.mean_absolute_error(preds, test_y)
    
    return pd.Series({"Accuracy": accuracy, "F1": f1, "R2": r2, "MAE": mae})

train_bow = train["review_body"].progress_apply(pre.preprocess_string)
id2word = Dictionary(train_bow)
id2word.filter_extremes(no_above=0.5, no_below=10)
train_bow = [id2word.doc2bow(i) for i in train_bow]
train_bow = corpus2csc(train_bow).T

# repeat the above, but using Pandas method chaining syntax, since we
# don't have train the id2word object again.
test_bow = (
    test["review_body"]
    .progress_apply(pre.preprocess_string)
    .progress_apply(id2word.doc2bow)
    .pipe(corpus2csc, num_terms=len(id2word))
).T

res = fit_and_score(
    BernoulliNB(),
    train_bow,
    train["stars"],
    test_bow,
    test["stars"],
)
print(res)