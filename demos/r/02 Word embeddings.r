# A version of the notebook `01c Bag-of-Words with udpipe.ipynb`,
# converted into a plain .r script and slighly rearranged.

library(caret)
library(dplyr)
library(text2vec)
library(yardstick)

train <- read.csv("../../data/train.csv", stringsAsFactors = FALSE)
test <- read.csv("../../data/test.csv", stringsAsFactors = FALSE)

# tokenize the training data and generate the vocab vectorizer.
tokens <- (
    train$review_body
    %>% tolower()
    %>% gsub("[^a-z]+", " ", .)
    %>% word_tokenizer()
    %>% itoken()
)
vectorizer <- (
    tokens
    %>% create_vocabulary()
    %>% filter(term_count >= 5)
    %>% vocab_vectorizer()
)

# Convert the tokens to a co-occurrence matrix and train the GloVe
# vectors.
co_occurrence_matrix <- create_tcm(tokens, vectorizer, skip_grams_window = 5)
glove = GlobalVectors$new(rank = 300, x_max=25)
wv_main = glove$fit_transform(
    co_occurrence_matrix,
    n_iter = 25,
    convergence_tol = 0.01,
    # NOTE: SET THIS LOWER IF YOU HAVE FEWER THREADS AVAILABLE
    # ON YOUR SYSTEM!
    n_threads = 8
)
word_vectors = wv_main + t(glove$components)

preprocess <- function(df, vectorizer, word_vectors) {
    dtm <- (
        df$review_body
        %>% tolower()
        %>% gsub("[^a-z]+", " ", .)
        %>% word_tokenizer()
        %>% itoken(progressbar = FALSE)
        %>% create_dtm(vectorizer)
    )
    dtm <- dtm %*% word_vectors
    return(dtm)
}

# column names are required for training with caret; just use dummy ones
# since the columns/features in the matrix don't have any meaningful direct
# interpretation anyways.
colnames(train_vecs) <- c(1:dim(train_vecs)[2])
colnames(test_vecs) <- c(1:dim(test_vecs)[2])

# train the multi-layer perceptron
mlp <- train(
    # training an MLP like this requires a Matrix object
    # in order to do any automated preprocessing
    x = as.matrix(test_vecs),
    y = as.factor(test_y),
    preProcess = c("center", "scale"),
    method ="mlp",
    size = c(128, 64, 64),
    maxit = 10
)

preds <- predict(mlp, as.matrix(test_vecs))
cat("Accuracy: ")
cat(mean(preds == test_y))

cat("F1 score: ")
cat(
    f_meas(
        data = data.frame(preds = preds, true = as.factor(test_y)),
        preds,
        true,
        beta = 1
    )$.estimate
)