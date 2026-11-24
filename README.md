<div align="center"> <img src="images/transformerslogo.png" alt="Transformers.jl" width="512"></img></div>

<!-- wardjm-fork-note -->
> [!NOTE]
> **This is a maintained fork of [chengchingwen/Transformers.jl](https://github.com/chengchingwen/Transformers.jl).** The upstream package is no longer
> actively updated, so I'm keeping this fork working with current Julia and current versions of its
> dependencies (Julia 1.11+, Flux 0.16, Zygote 0.7, CUDA 6, and JSON 1 in place of the deprecated JSON3). The fork keeps the original package name and UUID but is not
> registered in General, so install it by URL:
>
> ```julia
> using Pkg
> Pkg.add([
>     PackageSpec(url = "https://github.com/wardjm/PrimitiveOneHot.jl"),
>     PackageSpec(url = "https://github.com/wardjm/DoubleArrayTries.jl"),
>     PackageSpec(url = "https://github.com/wardjm/Pickle.jl"),
>     PackageSpec(url = "https://github.com/wardjm/NeuralAttentionlib.jl"),
>     PackageSpec(url = "https://github.com/JuliaText/WordTokenizers.jl"),
>     PackageSpec(url = "https://github.com/wardjm/TextEncodeBase.jl"),
>     PackageSpec(url = "https://github.com/wardjm/BytePairEncoding.jl"),
>     PackageSpec(url = "https://github.com/wardjm/Transformers.jl"),
> ])
> ```
>
> The other packages in that list are forks this one depends on (PrimitiveOneHot, DoubleArrayTries, Pickle, NeuralAttentionlib, TextEncodeBase, BytePairEncoding), plus the unreleased
> WordTokenizers 1.0 from upstream master. On Julia 1.13+ Pkg follows this repo's `[sources]`, so
> `Pkg.add(url = "https://github.com/wardjm/Transformers.jl")` alone is enough there.
>
> Issues and pull requests are welcome on this fork.


[![Build status](https://github.com/chengchingwen/Transformers.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/chengchingwen/Transformers.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/chengchingwen/Transformers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chengchingwen/Transformers.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chengchingwen.github.io/Transformers.jl/dev/)

Julia implementation of [transformer](https://arxiv.org/abs/1706.03762)-based models, with [Flux.jl](https://github.com/FluxML/Flux.jl).

*notice: The current version is almost completely different from the 0.1.x version. If you are using the old version, make sure to update the changes or stick to the old version.*

# Installation

In the Julia REPL:

    ]add Transformers


# Example

Using pretrained Bert with `Transformers.jl`.

```julia
using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace

textencoder, bert_model = hgf"bert-base-uncased"

text1 = "Peter Piper picked a peck of pickled peppers"
text2 = "Fuzzy Wuzzy was a bear"

text = [[ text1, text2 ]] # 1 batch of contiguous sentences
sample = encode(textencoder, text) # tokenize + pre-process (add special tokens + truncate / padding + one-hot encode)

@assert reshape(decode(textencoder, sample.token), :) == [
    "[CLS]", "peter", "piper", "picked", "a", "peck", "of", "pick", "##led", "peppers", "[SEP]",
    "fuzzy", "wu", "##zzy",  "was", "a", "bear", "[SEP]"
]

bert_features = bert_model(sample).hidden_state
```

See `example` folder for the complete example.


# For more information

If you want to know more about this package, see the [document](https://chengchingwen.github.io/Transformers.jl/dev/)
 and read code in the `example` folder. You can also tag me (@chengchingwen) on Julia's slack or discourse if
 you have any questions, or just create a new Issue on GitHub.
