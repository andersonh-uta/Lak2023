"""
Version of the notebook `01c Bag of Words with spaCy + scikit-learn.ipynb`,
converted to a .py script.
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
import spacy
from tqdm import tqdm

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

def spacy_preprocess(nlp, texts):
    """use spaCy for bag-of-words preprocessing"""
    docs = nlp.pipe(
        tqdm(texts),
        # disable not-needed steps for speed
        disable=["parser", "ner"],
        
        # multiprocess for extra speed
        n_process=8,
        batch_size=500,
    )

    # we could filter the tokens in a lot of ways, but I'm choosing
    # list comprehension today.
    docs = (
        [
            tok.lemma_.lower()
            for tok in doc
            if not (
                tok.is_stop     # no stopwords
                or tok.is_space # no space tokens
                or tok.is_punct # no punctuation tokens
                or tok.is_digit # no numbers
            )
        ]
        for doc in docs
    )

    # return documents as strings
    docs = [" ".join(i) for i in docs]
    
    return docs


# we need an `if __name__ == "__main__"` block to ensure that
# the spaCy multiprocessing doesn't run haywire--multiprocessing
# in Python is...tricky like that sometimes.
if __name__ == "__main__":
    # load the data
    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")
    # the small English model--optimized for speed and memory footprint,
    # but at the cost of (a little bit of) accuracy.
    nlp = spacy.load("en_core_web_sm")
    bow_train = spacy_preprocess(nlp, train["review_body"])
    bow_test = spacy_preprocess(nlp, test["review_body"])
    
    clf = Pipeline([
        ("bag of words", CountVectorizer(max_df=0.5, min_df=10)),
        ("clf", BernoulliNB())
    ])
    res = fit_and_score(
        clf,
        bow_train,
        train["stars"],
        bow_train,
        train["stars"],
    )
    print(res)