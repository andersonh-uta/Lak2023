import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    get_scheduler
)
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, x, y, tokenizer):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def tokenize(texts, tokenizer):
    """Tokenize the texts, so the model can take them as inputs."""
    return tokenizer(
        # the texts must be a list--not an "array-like," but an actual
        # list--if we're passing multiple texts at once.
        list(texts),

        # pad short texts out to the max sequence length for the model.
        padding="max_length",

        # truncate long texts to the model's max length.  Truncation is
        # rarely an issue for texts that are just a bit longer than the
        # model's max length, but it can introduce errors for texts that
        # are very long.  (very long texts are a pathological problem for
        # transformers).
        truncation=True,

        # return PyTorch tensors, since we're going to use PyTorch models
        # and PyTorch training loops.
        # (other options are "tf" for Tensorflow tensors, or "np" for
        # Numpy ndarrays).
        return_tensors="pt",
    )

def make_dataset(df, tokenizer, batch_size=8):
    return DataLoader(
        TextDataset(df["review_body"], df["stars"], tokenizer),
        batch_size=batch_size,
        shuffle=True,
    )

def fine_tune(model, train_dataset, val_dataset):
    """fine-tune a model"""
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    n_epochs = 1
    n_train_steps = n_epochs * len(train_dataset)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_train_steps
    )

    # use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # training loop!
    # yes, the indents move about half a mile to the right, and I'm sorry
    # about that.
    model.to(device)
    batchnum = 0
    best_val_loss = 100
    early_stopping_patience = 0
    for (x, y) in tqdm(train_dataset, desc=f"Training"):

        # tokenize the strings and move them to the GPU, if a GPU is available.
        batch = {
            k: v.to(device)
            for (k,v) in tokenize(x, tokenizer).items()
        }
        batch["labels"] = y.to(device)

        preds = model(**batch)
        loss = preds.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        batchnum += 1

        # evaluate on the validation data every 250 batches
        if batchnum % 250 == 0:
            val_loss = 0
            with torch.no_grad():
                for (x, y) in val_dataset:
                    batch = {
                        k: v.to(device)
                        for (k,v) in tokenize(x, tokenizer).items()
                    }
                    batch["labels"] = y.to(device)
                    preds = model(**batch)
                    loss = preds.loss
                    val_loss += loss
            val_loss = val_loss / len(val_dataset)
            print(f"Batch {batchnum:,} - validation loss={val_loss}")

            # simple early stopping criteria--must see any improvement
            # in validation loss within 5 validation rounds, or we stop
            # training.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_patience = 0
            else:
                early_stopping_patience += 1

            if early_stopping_patience >= 5:
                break
                
    torch.cuda.empty_cache()
    with torch.no_grad():
        predicted = []
        ys = []
        model.cuda()
        for (x, y) in tqdm(test_dataset):
            batch = {k:v.to(device) for k,v in tokenize(x, tokenizer).items()}
            preds = model(**batch)
            predicted.append(preds["logits"].argmax(axis=1).detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())

    predicted = np.hstack(predicted)
    ys = np.hstack(ys)
        
    acc = np.mean(predicted == ys)
    f1 = metrics.f1_score(predicted, ys, average="macro")

    print(f"Accuracy: {acc:.2%}")
    print(f"Macro F1: {f1:.4f}")
    
    return

if __name__ == "__main__":
    # the model, which we'll fine-tune for our classification task.
    # it will be downloaded+cached if not already available locally.
    MODEL_NAME = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
        torch_dtype=torch.float32 if torch.cuda.is_available() else torch.float16,
    )

    # the tokenizer, which we'll use to manually preprocess our data.
    # it will be downloaded+cached if not already available locally.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # load the data
    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")
    val = pd.read_csv("../../data/validation.csv")
    
    train_dataset = make_dataset(train, tokenizer, batch_size=8)
    test_dataset = make_dataset(test, tokenizer, batch_size=32)
    val_dataset = make_dataset(val, tokenizer, batch_size=32)

    # the Transformer models will expect our labels to be numeric
    # indices starting from 0; just subtract 1 from our stars and
    # we're good.  (and add 1 to the final predicted number of stars
    # to convert back).
    train["stars"] -= 1
    test["stars"] -= 1
    val["stars"] -= 1
    
    model = fine_tune(model, train_dataset, val_dataset)

