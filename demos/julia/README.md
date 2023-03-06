# Julia Demos

This directory contains some demos of using Julia for basic NLP tasks.

## Huh?  Julia?

What is Julia?  Why include it?  I'm glad you asked!  Julia is a very cool, relatively young programming language (1.0 was released in 2018, but it had been in development since about 2012).  The pitch, in a nutshell: "writes like Python, runs like C."  Easy to write, and extremely fast to run.  And it has a lot of other cool bells and whistles that borrow and improve on good ideas from R (multiple dispatch; built-in package management), Haskell (type system; functional programming), Lisp (syntactic macros), and a few other languages.  Julia is super cool, especially if you find yourself needing you write really high-performance code, but you don't want to use the likes of C/C++/Fortran/Rust.

Sadly, Julia doesn't have a very mature NLP ecosystem as of right now.  Even a lot of its machine learning libaries are still fairly young and have a ways to go.  But, the language has already seen a huge growth in popularity, so it's just a matter of time before it becomes a mainstay of the data analysis world.

Mostly, though, I really like Julia, and want more people to use it.

## Files in this directory

- `01 Bag of words`: shows how to do a simple bag-of-words model in Julia.  Unlike R and Python, there's still some awkwardness using machine learning models with sparse matrices (like you get from bag-of-words), so this notebook also uses SVD to reduce the dimensionality to a more manageable size, and throws a decision tree at it.
- `02 Word Embeddings`: shows how to load pre-trained word embeddings (Julia doesn't seem to have a really good library for training your own yet) and use those to build a predictive model.

Note that, unlike R/Python, Julia doesn't have a huge plethora of libraries for things like tokenization, stemming, etc.  (Not yet, at least).

Each notebook is configured to activate a locak `Pkg` environment, and each notebook has a list of `Pkg.add()` calls to install the required dependencies.  Uncomment and run these cells to install the dependencies, after running the `Pke.activate(".")` call.  (note that this might take a while, since Julia pre-compiles parts of some libaries).

There's no notebook for transformers--the best place to use transformer models is the `transformers` library in Python, made by Huggingface.  You _can_ use this via Julia using the `PyCall.jl` package, but I don't have any experience using that package.  And if your workflow is going to be heavily dependent on the transformer models, you should probably just use Pyton and not worry as much about the inter-language communication.