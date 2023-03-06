#=
A version of `01 Bag of words.ipynb` converted to a plain .jl script.
Julia really, really wants things to be wrapped up in functions when
writing plain scripts, so I've taken a lot of the stuff that was
previouly in cells in the notebook and encapsulated it into functions.
(I've also taken away all the excessive and unnecessary type annotations).
=#

using Pkg
Pkg.activate(".")

# requirements
# Pkg.add("CSV")            # parse CSV files
# Pkg.add("DataFrames")     # dataframes
# Pkg.add("MLJ")            # general machine learning framework
# Pkg.add("MLJScikitLearnInterface") # interface to scikit-learn, so we can use sklearn models
# Pkg.add("Pipe")           # macros for better piping syntax
# Pkg.add("ProgressMeter")  # progress bars
# Pkg.add("Snowball")       # interface to the Snowball stemming libary
# Pkg.add("WordTokenizers") # some common tokenization algorithms

using CategoricalArrays
using CSV
using DataFrames
using MLJ
using Pipe
using ProgressMeter
using Snowball
using SparseArrays

const STOPWORDS = map(
    x -> stem(Stemmer("english"), x),
    split("""i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t can will just don should now aren isn weren""")
)

function load_data()
     return (
        DataFrame(CSV.File("../../data/train.csv")),
        DataFrame(CSV.File("../../data/test.csv")),
    )
end

preprocess(s, stemmer) = @pipe (
    s
    |> lowercase(_)
    |> replace(_, r"[^a-z]+" => " ")
    |> split(_)
    |> map(x -> stem(stemmer, x), _)
    |> filter(x -> length(x) >= 3, _)
    |> filter(x -> !(x in STOPWORDS), _)
)

# Count tokens + the total number of tokens
function counter(iterable)
    counts = Dict()
    for i ∈ iterable
        counts[i] = get(counts, i, 0) + 1
    end
    return counts
end

function remove_extreme!(counts, threshold, comparison)
    if threshold < 1
        threshold = sum(values(counts)) * threshold
    end
    
    for (k, v) ∈ pairs(counts)
        if comparison(v, threshold)
            delete!(counts, k)
        end
    end
end

remove_frequent!(counts, thresh) = remove_extreme!(counts, thresh, >)
remove_rare!(counts, thresh) = remove_extreme!(counts, thresh, <)


function doc2bow(vocab, doc)
    return Dict(
        vocab[tok] => count
        for (tok, count) in counter(doc)
        if tok ∈ keys(vocab)
    )
end

function tokens2bow(vocab, docs)
    # convert each document into a dict of token_index => count pairs
    bow = [doc2bow(vocab, doc) for doc ∈ docs]

    # set up the "internal" arrays for the sparse matrix.
    # This is a pretty standard sparse matrix format, but you
    # may need to read some documentation for it to make sense
    # if you haven't seen it before.
    colptr = zeros(Int, length(bow) + 1)
    rowval = zeros(Int, sum(length.(bow)))
    nzval  = zeros(UInt16, sum(length.(bow)))
    colptr[1] = 1
    
     # indices that we'll advance through as we update the above arrays
    rowval_ptr = 1
    colptr_ptr = 2
    
    # update the colptr/rowval arrays
    for doc in bow
        for (row_idx, val) in doc
            rowval[rowval_ptr] = row_idx
            nzval[rowval_ptr] = val
            rowval_ptr += 1
            colptr[colptr_ptr] += 1
            # println("[$row_idx, $(colptr_ptr-1)]=$val")
        end
        colptr_ptr += 1
    end
    
    return SparseMatrixCSC(
        length(vocab),
        length(bow),
        cumsum(colptr),
        rowval,
        nzval,
    )
end

# load and preprocess
train, test = load_data()
train_tokens = @showprogress "Preprocessing training data" [
     preprocess(i, Stemmer("english")) for i in train[!, :review_body]
]
test_tokens = @showprogress "Preprocessing testing data" [
     preprocess(i, Stemmer("english")) for i in test[!, :review_body]
]

# caculate word counts and filter extreme frequencies
word_counts = counter(tok for doc ∈ train_tokens for tok ∈ Set(doc))
remove_frequent!(word_counts, 0.5)
remove_rare!(word_counts, 10)

# token-to-index mapping
vocab = Dict(j => i for (i, j) in enumerate(keys(word_counts)))

# convert to document-term matrices
train_bow = tokens2bow(vocab, train_tokens)
test_bow = tokens2bow(vocab, test_tokens)

# load + fit the model: SVD + decision tree
model = @load DecisionTreeClassifier pkg=DecisionTree
svd = @load TSVDTransformer pkg=TSVD
model = machine(
    Pipeline(svd(nvals=300), model()),
    coerce(transpose(train_bow), Continuous),
    coerce(train[!, :stars], Multiclass),
)
fit!(model)
preds = predict(model, coerce(transpose(test_bow), Continuous));

preds = mode.(preds)
println("""
    Accuracy: $(mean(preds .== test[!, :stars]))
    Macro F1: $(macro_f1score(preds, test[!, :stars]))
    R^2:      $(rsq(unwrap.(preds), test[!, :stars]))
    MAE:      $(mean(abs.(unwrap.(preds) .- test[!, :stars])))
""")