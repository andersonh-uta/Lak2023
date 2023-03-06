# A version of the notebook `01c Bag-of-Words with udpipe.ipynb`,
# converted into a plain .r script and slighly rearranged.

# Requirements
# install.packages("dplyr")      # if you don't know what dplyr is I can't help you
# install.packages("magrittr")   # pipes!
# install.packages("naivebayes") # Naive Bayes implementation
# install.packages("SnowballC")  # stemming
# install.packages("tidytext")   # only using the stop_words dataframe from tidytext
# install.packages("udpipe")     # linguistic annotation models
# install.packages("yardstick")  # model metrics; part of the tidymodels suite

library(dplyr)
library(magrittr)
library(naivebayes)
library(SnowballC)
library(tidytext)
library(udpipe)
library(yardstick)

string2bow <- function (df) {
    tokenized <- (
        # not parallelizing this--it seems to cause weird errors when
        # you parallelize and also specify a doc_id argument.
        udpipe(
            x = df$review_body,
            object = "english",
            trace = 10000,
            
            # disable part of speech tagging and syntactic parsing
            # for extra speed.
            tagger = "none",
            parser = "none",
            
            # specify document IDs.
            doc_id = c(df$id_and_split)
        )
        %>% as.data.frame()
        %>% filter(!(lemma %in% stop_words$word))
        %>% mutate(lemma = gsub("[^a-z]", "", token))
        %>% filter(nchar(lemma) > 2)
        %>% mutate(
            review_id = gsub(";[^;]+", "", doc_id),
            Split = gsub("[^;]+;", "", doc_id)
        )
    )
    
    # remove rare + common terms; but base this determination
    # only on the training dataset.
    common_terms <- (
        tokenized
        %>% filter(Split == "Train")
        %>% group_by(lemma)
        %>% tally()
        %>% mutate(pct = n / sum(n))
        %>% filter(!(n < 10 | pct > 0.5))
    )
    tokenized <- filter(tokenized, lemma %in% common_terms$lemma)
    
    tokenized$n = 1
    
    return(tokenized)
}

# load data
train <- read.csv("../../data/train.csv", stringsAsFactors = FALSE)
test <- read.csv("../../data/test.csv", stringsAsFactors = FALSE)

# only running on on a subset of our data for the sake of the demo
# and speed--feel free to run it over the whole dataset on your own,
# but be prepared to wait a while before seeing any output in Jupyter.
train$Split = "Train"
test$Split = "Test"
data <- rbind(train, test)

# we can specify a doc_id column with udpipe and use it to track document-
# level metadata.  We'll need to track the training and testing splits,
# as well as review IDs, but the document ID for udpipe has to be a character
# vector.  So we'll just paste together the fields we need and split them
# back apart later.
data$id_and_split <- paste(data$review_id, data$Split, sep=";")

# run udpipe analyses
tokens <- string2bow(data)

# Convert to sparse matrices
bow <- cast_sparse(tokens, review_id, lemma, n)

# extract the y values and the split labels
labels <- filter(data, review_id %in% rownames(bow))$stars
splits <- filter(data, review_id %in% rownames(bow))$Split

# break the data back out into train and test
train_bow <- bow[splits == "Train",]
train_y <- labels[splits == "Train"]

test_bow <- bow[splits == "Test",]
test_y <- labels[splits == "Test"]

# fit and evaluate the naive bayes model
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