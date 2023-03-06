using Pkg
Pkg.activate(".")

# requirements
# Pkg.add("CSV")
# Pkg.add("CUDA")
# Pkg.add("DataFrames")
# Pkg.add("Embeddings")
# Pkg.add("Flux")
# Pkg.add("Pipe")
# Pkg.add("ProgressMeter")

using CSV
using CUDA
using DataFrames
using Embeddings
using Flux
using Pipe
using ProgressMeter

function load_data()
    return (
        DataFrame(CSV.File("../../data/train.csv")),
        DataFrame(CSV.File("../../data/test.csv")),
        DataFrame(CSV.File("../../data/validation.csv")),
    )
end

# very similar preprocessing to before, but without stemming
preprocess(s) = @pipe (
    s
    |> lowercase(_)
    |> replace(_, r"[^a-z]+" => " ")
    |> split(_)
    |> filter(x -> length(x) >= 3, _)
)

function counter(it)
    counts = Dict()
    for i ∈ it
        counts[i] = get(counts, i, 0) + 1
    end
    return counts
end

function get_vector(word, tok2id, embeddings)
    if !(word ∈ keys(tok2id))
        return zeros(size(embeddings.embeddings)[1])
    else
        return embeddings.embeddings[:, tok2id[word]]
    end
end

function doc2vec(doc, tok2id, embeddings)
    if length(doc) == 0
        return zeros(size(embeddings.embeddings)[1])
    end
    return @pipe (
        [get_vector(tok, tok2id, embeddings) for tok ∈ doc]
        |> reduce(hcat, _)
        |> sum(_, dims=2) ./ size(_)[2]
    )
end

function one_hot(labels)
    encoded = zeros(length(unique(labels)), size(labels)[1])
    for l ∈ 1:length(labels)
        encoded[labels[l], l] = 1
    end   
    return encoded
end

train, test, val = load_data()
train_tokens = preprocess.(train[!, :review_body])
test_tokens = preprocess.(test[!, :review_body])
val_tokens = preprocess.(val[!, :review_body])

# only keep words with >=5 total occurrences, based on the
# training dataset
whitelist = counter([tok for doc in train_tokens for tok in doc])
whitelist = Set(tok for (tok, count) ∈ whitelist if count > 5)

train_tokens = [[tok for tok ∈ doc if tok ∈ whitelist] for doc in train_tokens]
test_tokens = [[tok for tok ∈ doc if tok ∈ whitelist] for doc in test_tokens]
val_tokens = [[tok for tok ∈ doc if tok ∈ whitelist] for doc in val_tokens]

# downloads the vector files if needes.
# The "4" specifies which of the Glove embeddings to load--this loads
# the 300-dimensional ones.
vectors = load_embeddings(GloVe{:en}, 4)
WORD_TO_IDX = Dict(reverse.(enumerate(vectors.vocab)))

# vectorize the texts
train_vectors = reduce(
    hcat,
    @showprogress [doc2vec(i, WORD_TO_IDX, vectors) for i ∈ train_tokens]
)
test_vectors = reduce(
    hcat,
    @showprogress [doc2vec(i, WORD_TO_IDX, vectors) for i ∈ test_tokens]
)
val_vectors = reduce(
    hcat,
    @showprogress [doc2vec(i, WORD_TO_IDX, vectors) for i ∈ val_tokens]
)

# one-hot encode the labels
train_y = one_hot(train[!, :stars])
test_y = one_hot(test[!, :stars])
val_y = one_hot(val[!, :stars])

# now time for our Flux model!
training_data = Flux.DataLoader(
    (train_vectors, train_y) |> gpu,
    batchsize=256,
    shuffle=true,
);

function evaluate_model(model, x, y, loss_fn)
    preds = cpu(model(gpu(x)))
    hard_preds = [i.I[1] for i ∈ argmax(preds, dims=1)]
    y_ = [i.I[1] for i ∈ argmax(y, dims=1)]
    acc = sum(y_ .== hard_preds) / size(y)[2]
    return loss_fn(preds, y), acc
end

function train_model(train_vectors, train_y, test_vectors, test_y, val_vectors, val_y, vectors)
    # our network
    model = Chain(
        BatchNorm(size(vectors.embeddings)[1]),
        Dense(size(vectors.embeddings)[1] => 256, relu),
        Dense(256 => 256, relu),
        Dense(256 => 256, relu),
        Dense(256 => 5),
        softmax,
    )
    model = gpu(model)

    # our optimizer
    optim = Flux.setup(Flux.Adam(0.01), model)
    
    val_loss, acc = evaluate_model(model, val_vectors, val_y, Flux.crossentropy)
    println("Before training: val_loss=$val_loss acc=$acc")
    for epoch in 1:5
        @showprogress "Epoch $epoch training loop" for (x, y) in training_data
            loss, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                Flux.crossentropy(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
        end
        val_loss, acc = evaluate_model(model, val_vectors, val_y, Flux.crossentropy)
        println("After epoch $epoch: val_loss=$val_loss acc=$acc")
    end

    evaluate_model(model, test_vectors, test_y, Flux.crossentropy)
    
    return model
end

train_model(train_vectors, train_y, test_vectors, test_y, val_vectors, val_y, vectors)