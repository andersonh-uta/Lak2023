# LAK 2023 Workshop: Lowering the Technical Barriers to Trustworthy NLP

This repository contains information and materials from the Lowering the Technical Barriers to Trustworthy NLP workshop at LAK 2023.

# Contents

There are a few directories containing some simple demo code for doing NLP analyses in various programming languages.  Each one shows off different libraries/tools that exist in that language, and all use the same common dataset.

- `data/`: Contains the common dataset in .csv format.  This is the English language subset of the [`amazon_reviews_multi`](https://huggingface.co/datasets/amazon_reviews_multi) dataset from the Huggingface datasets hub, converted to CSV format, and with some extraneous columns removed.
- `demos/`: demos on a few different common NLP approaches using Python, Julia, and R.  Each set of demos has Jupyter notebooks (pre-run, so you can look at the code and outputs), plus plain `.py`/`.r`/`.jl` files with just the final code and less annotations.
- `demos/python/`: Python demos.
- `demos/r/`: R demos.
- `demos/julia/`: Julia demos.

## A note about the different languages

If you're going to be doing ny moderately intensive NLP work, you should probably be using Python. Python has, by far, the most mature and cohesive ecosystem for doing NLP as of right now (March 2023), and provides the best balance between high-level abstractions when you want them, and fairly easy (and relatively efficient) access to lower-level primitives when you need them.

R has some tools for NLP work, but they're usually not as feature-rich and have a few odd quirks.  Julia is in a similar position--there aren't a lot of very mature NLP libraries right now, but Julia makes it _much_ easier than Python or R to re-implement something yourself and not worry about the performance (because Julia is very fast).  I like Julia a lot for this reason; most of the stuff we do in NLP isn't particularly weird or esoteric, and the lack of pre-built tools means you have to build more yourself, and see how simple most of it really is.

But, to try to encourage you to use Python, the Python notebooks will contain more details about the more general "what" and "why" of a given approach, and you should go to them first if you need some conceptual grounding.  The R and Julia notebooks will be much more "just the code."