# A version of the notebook `01a Bag-of-Words with tidytext.ipynb`,
# converted into a plain .r script and slighly rearranged.

# Dependencies---uncomment and run to install them
# install.packages("dplyr")      # if you don't know what dplyr is I can't help you
# install.packages("magrittr")   # pipes!
# install.packages("naivebayes") # Naive Bayes implementation
# install.packages("SnowballC")  # stemming
# install.packages("tidytext")   # text processing using Tidy data principles
# install.packages("yardstick")  # model metrics; part of the tidymodels suite

library(dplyr)
library(magrittr)
library(naivebayes)
library(SnowballC)
library(tidytext)
library(yardstick)

# convert a dataframe to the tokenized formt.
string2bow <- function (df) {
    tokenized <- (
        df[, c("Split", "review_body", "review_id")]
        %>% unnest_tokens(token, review_body)
        %>% filter(!(token %in% stop_words$word))
        %>% mutate(token = wordStem(token))
        %>% mutate(token = gsub("[^a-z]", "", token))
        %>% filter(nchar(token) > 2)
    )
    
    # remove rare + common terms; but base this determination
    # only on the training dataset.
    common_terms <- (
        tokenized
        %>% filter(Split == "Train")
        %>% group_by(token)
        %>% tally()
        %>% mutate(pct = n / sum(n))
        %>% filter(!(n < 10 | pct > 0.5))
    )
    tokenized <- filter(tokenized, token %in% common_terms$token)
    
    # we need a numeric value to populate the sparse matrix with;
    # this should be the word counts.
    tokenized$n = 1
    
    return(tokenized)
}

# load data
train <- read.csv("../../data/train.csv", stringsAsFactors = FALSE)
test <- read.csv("../../data/test.csv", stringsAsFactors = FALSE)

# add indicator columns so we can split the datasets apart again later.
train$Split <- "Train"
test$Split <- "Test"
data <- rbind(train, test)
cat("Tokenizing...")
tokens <- string2bow(data)
cat("done.")

# convert tokenized texts into sparse bag-of-word matrix
bow <- cast_sparse(tokens, review_id, token, n)

# extract the y values and the split labels
labels <- filter(data, review_id %in% rownames(bow))$stars
splits <- filter(data, review_id %in% rownames(bow))$Split

# break the data back out into train and test
train_bow <- bow[splits == "Train",]
train_y <- labels[splits == "Train"]

test_bow <- bow[splits == "Test",]
test_y <- labels[splits == "Test"]

# train + predict with naive bayes model
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