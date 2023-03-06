from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import spacy
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

tqdm.pandas(smoothing=0)

class Corpus:
    """a class that prints progress as you iterate through it.
    like tqdm, but it properly re-initializes the progress bar
    after each run through.  I like to use this in place of 
    once-per-epoch callbacks in Gensim since this provides
    more constant and real-time feedback."""
    def __init__(self, it):
        self.it = it
        self.n = 1
        
    def __iter__(self):
        yield from tqdm(
            self.it,
            unit_scale=True,
            desc=f"Pass {self.n}",
            smoothing=0,
        )
        self.n += 1
        
    def __len__(self):
        return len(self.it)

def spacy_vectorization(docs, nlp):
    # if all we need is vectors, we can use nlp.make_doc()
    # for a pretty big speedup.  This will basically just do
    # tokenization, case normalization, and vector lookups.
    return np.array([
        i.vector
        for i in map(nlp.make_doc, tqdm(docs, smoothing=0))
    ])

def gensim_vectorization(df, w2v, nlp):
    # we'll be fancy and do this with generators
    texts = df["review_body"].progress_apply(tokenize, nlp=nlp)
    
    # remove any words that didn't make it into the word2vec vocab
    texts = (
        [tok for tok in doc if tok in w2v.wv.key_to_index]
        for doc in texts
    )
    
    # get vectors and stack them into a single array
    texts = np.array([
        np.mean(w2v.wv[doc], axis=0)
        if len(doc) > 0
        else np.zeros(300)
        for doc in texts
    ])
    
    return texts

def prepare_dataset(x, y):
    x = torch.Tensor(x)
    y = torch.Tensor(pd.get_dummies(y["stars"]).values)
    return DataLoader(
        TensorDataset(x, y),
        # this dataset isn't very sensitive to batch size, so let's
        # pick a big one to let us iterate more quickly.
        batch_size=512,
        shuffle=True,
    )

def training_loop(training_dataset, testing_dataset, validation_dataset):
    """pretty standard PyTorch training loop to train a model"""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(300),
        torch.nn.Linear(300, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 5),
    )
    model.to(DEVICE)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # trackers for early stopping--5 rounds with no improvement
    # before stopping.
    best_val_loss = np.inf
    early_stopping_counter = 0
    
    for epoch in range(10):
        # validate at te top of each epoch
        model.eval()
        val_loss = 0
        preds = []
        ys = []
        for (x, y) in validation_dataset:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # loss-per-sample
            preds.append(model(x).to("cpu"))
            ys.append(y.to("cpu"))
        model.train()
        preds = torch.cat(preds)
        ys = torch.cat(ys)
        val_loss = loss_fn(preds, ys)
        
        preds = torch.argmax(preds, axis=1).numpy()
        ys = torch.argmax(ys, axis=1).numpy()
        val_acc = np.mean(preds == ys)
        val_f1 = f1_score(preds, ys, average="macro")
        print(f"{val_loss=:.6f} - {val_acc=:.3%} - {val_f1=:.6f}")
        
        # check early stopping criteria
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= 5:
            break
            
        # training epoch
        for (x, y) in tqdm(training_dataset, desc=f"Epoch {epoch}"):
            # DEVICE is read from outer/global scope; defined in
            # previous cell
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # learn from batch
            opt.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()
            
    # final performance metrics
    model.eval()
    preds = []
    ys = []
    for (x, y) in testing_dataset:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        preds.append(model(x).to("cpu").detach().numpy())
        ys.append(y.to("cpu"))
    preds = np.argmax(np.vstack(preds), axis=1)
    ys = np.argmax(np.vstack(ys), axis=1)

    acc = np.mean(preds == ys)
    f1 = f1_score(preds, ys, average="macro")
    
    return acc, f1

def tokenize(s, nlp):
    return [tok.lower_ for tok in nlp.make_doc(s)]

def train_model_with_spacy_vectors(train, test, val, nlp):
    # vectorize with spaCy's pre-trained GLoVe vectors
    train_docs = spacy_vectorization(train["review_body"], nlp)
    test_docs = spacy_vectorization(test["review_body"], nlp)
    val_docs = spacy_vectorization(val["review_body"], nlp)

    # convert to PyTorch DataLoaders
    train_torch = prepare_dataset(train_docs, train)
    test_torch = prepare_dataset(test_docs, test)
    val_torch = prepare_dataset(val_docs, val)

    # train the model
    acc, f1 = training_loop(train_torch, test_torch, val_torch)
    print(f"Final test set scores with spaCy vectors: accuracy={acc:.2%}, f1={f1:.4f}")
    
    return

def train_model_with_custom_vectors(train, test, val, nlp):
    """train a custom set of word2vec vectors with gensim, and use those
    to vectorize the documents before making predictions."""
    
    # train the word2vec model on all texts
    docs = pd.concat((
        train["review_body"],
        test["review_body"],
        val["review_body"]
    )).progress_apply(tokenize, nlp=nlp)
    w2v = Word2Vec(
        Corpus(docs),
        vector_size=300,
        workers=10,
        sg=1,
        hs=0,
        min_count=5,
        epochs=1,
    )
    
    train_docs = gensim_vectorization(train, w2v, nlp)
    test_docs = gensim_vectorization(test, w2v, nlp)
    val_docs = gensim_vectorization(val, w2v, nlp)
    
    train_torch = prepare_dataset(train_docs, train)
    test_torch = prepare_dataset(test_docs, test)
    val_torch = prepare_dataset(val_docs, val)
    
    acc, f1 = training_loop(train_torch, test_torch, val_torch)
    print(f"Final test set scores with custom Word2Vec vectors: accuracy={acc:.2%}, f1={f1:.4f}")
    return

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")

    # load the data
    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")
    val = pd.read_csv("../../data/validation.csv")
    
    train_model_with_spacy_vectors(train, test, val, nlp)
    train_model_with_custom_vectors(train, test, val, nlp)