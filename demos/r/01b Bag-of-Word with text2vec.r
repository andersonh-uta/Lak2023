# A version of the notebook `01b Bag-of-Words with text2vec.ipynb`,
# converted into a plain .r script and slighly rearranged.

# Requirements
# install.packages("dplyr")      # if you don't know what dplyr is I can't help you
# install.packages("magrittr")   # pipes!
# install.packages("naivebayes") # Naive Bayes implementation
# install.packages("SnowballC")  # stemming
# install.packages("text2vec")   # tokenization and document-term matrix creation
# install.packages("yardstick")  # model metrics; part of the tidymodels suite

library(dplyr)
library(magrittr)
library(naivebayes)
library(SnowballC)
library(text2vec)
library(yardstick)

preprocess <- function(s) {
    return (
        s$review_body
        %>% tolower()
        %>% gsub("[^a-z]+", " ", .)
        %>% word_tokenizer()
        %>% lapply(wordStem)
        %>% itoken()
    )
}

# Load data
train <- read.csv("../../data/train.csv", stringsAsFactors = FALSE)
test <- read.csv("../../data/test.csv", stringsAsFactors = FALSE)

# tokenize
cat("Preprocessing...")
train_tokens <- preprocess(train)
test_tokens <- preprocess(test)
cat("done.")

# create vectorizer from the training data
vectorizer <- (
    train_tokens
    %>% create_vocabulary()
    %>% filter(doc_count > 10)
    %>% filter(doc_count < (sum(.$doc_count) / 2))
    %>% vocab_vectorizer()
)

# convert to document-term matrix
train_bow <- create_dtm(train_tokens, vectorizer)
test_bow <- create_dtm(test_tokens, vectorizer)
train_y <- train$stars
test_y <- test$stars

# train and evaluate the naive bayes model
# Now fit the Naive Bayes model as before.
nb <- bernoulli_naive_bayes(train_bow, as.factor(train_y))
preds <- predict(nb, newdata = test_bow)

cat("Accuracy: ")
cat(mean(preds == test_y))
cat("\nF1 score: ")
cat(f_meas(
        data = data.frame(preds = preds, true = as.factor(test_y)),
        preds,
        true,
        beta = 1
    )$.estimate
)