# R Demos

While Python has a much larger and a somewhat more mature ecosystem for doing NLP, R has plenty of great tools for doing most kinds of modeling.  This set of files shows how to use R for some basic predictive modeling tasks using language data.  (Note: while I consider myself a competent R programmer, I'm far less competent than I am in Python and Julia.  So it's entirely possible I'm just not aware of some great R library for working with text.  If you know about one that I don't, let me know about it!)

Each notebook has a cell with the required `install.packages()` statements to install all the dependencies.  Uncomment and run the respective cells to install all the needed goodies.  (you may need to make sure your version of R is relatively recent; these notebooks were run using R 4.1).

The files:
- `01a Bag-of-Words with tidytext`: shows how to do a bag-of-words analysis using the `tidytext` package, which is part of the Tidyverse package family.
- `01b Bag-of-Word with text2vec`: shows how to do a bag-of-words analysis using the `text2vec` package, which makes some things much easier than `tidytext` for predictive modeling.
- `01c Bag-of-Words with udpipe`: shows how to do a bag-of-words analysis using `udpipe`, which lets you work with a much richer set of linguistic annotations than either `tidytext` or `text2vec`, but at the cost of running slower.
- `02 Word embeddings`: shows how to use `text2vec` to build your own GloVe embeddings, and train a small neural network model on the resulting vectors.

Note that there is no dedicted notebook for using Transformer models.  This is because the gold-standard library for transformers is Huggingface's `transformers` library in Python, and as far as I can tell, there isn't anything approaching that library in R.  There are some guides for how to run `transformers` in R via the `reticulate` package, but I've never gotten `reticulate` to work properly, so I'm going to leave that as an exercise for you if you're interested.