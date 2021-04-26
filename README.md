# No more development; not even bug fixes in the foreseeable future

Due to life changes. I have 0 time now to handle this Open Source project. So this will be archived
until I can come back to it.

I will refocus my energy on only a couple of open source packages one of them being {disk.frame}.


# JLBoost.jl

This is a 100%-Julia implementation of Gradient Boosting Regresssion Trees (GBRT) based heavily on the algorithms published in the XGBoost, LightGBM and Catboost papers. GBRT is also referred to as Gradient Boosting Decision Tree (GBDT).

## Limitations for now
* Currently, `Union{T, Missing}` feature type is not supported, but is *planned*.
* Currently, only the single-valued models are supported. Multivariate-target models support is *planned*.
* Currently, only the numeric and boolean features are supported. Categorical support is *planned*.
* Currently, weights cannot be provided for each of the records. Support is *planned*.

## Objectives
* A full-featured & batteries included Gradient Boosting Regression Tree library
* Play nice with the Julia ecosystem e.g. Tables.jl, DataFrames.jl and CategoricalArrays.jl
* 100%-Julia
* Fit models on large data
* Easy to manipulate the tree after fitting; play with tree pruning and adjustments
* "Easy" to deploy
* Completely [hackable](https://docs.google.com/presentation/d/1xjhi8AbOpBzCxoLy9kGR_NuBqo0O2EWpqQFfXCAjCGY/edit?usp=sharing)

## Quick-start

### Fit model on `DataFrame`

#### Binary Classification
We fit the model by predicting one of the iris Species. To fit a model on a `DataFrame` you need to specify the column and the features default to all columns other than the target.

```julia
using JLBoost, RDatasets
iris = dataset("datasets", "iris")

iris[!, :is_setosa] = iris[!, :Species] .== "setosa"
target = :is_setosa

features = setdiff(names(iris), ["Species", "is_setosa"])

# fit one tree
# ?jlboost for more details
xgtreemodel = jlboost(iris, target)
```

```
1
150×6 DataFrameColumns
 Row │ SepalLength  SepalWidth  PetalLength  PetalWidth  Species     is_set
osa
     │ Float64      Float64     Float64      Float64     Cat…        Bool
─────┼─────────────────────────────────────────────────────────────────────
────
   1 │         5.1         3.5          1.4         0.2  setosa           t
rue
   2 │         4.9         3.0          1.4         0.2  setosa           t
rue
   3 │         4.7         3.2          1.3         0.2  setosa           t
rue
   4 │         4.6         3.1          1.5         0.2  setosa           t
rue
   5 │         5.0         3.6          1.4         0.2  setosa           t
rue
   6 │         5.4         3.9          1.7         0.4  setosa           t
rue
   7 │         4.6         3.4          1.4         0.3  setosa           t
rue
   8 │         5.0         3.4          1.5         0.2  setosa           t
rue
   9 │         4.4         2.9          1.4         0.2  setosa           t
rue
  10 │         4.9         3.1          1.5         0.1  setosa           t
rue
  11 │         5.4         3.7          1.5         0.2  setosa           t
rue
  12 │         4.8         3.4          1.6         0.2  setosa           t
rue
  13 │         4.8         3.0          1.4         0.1  setosa           t
rue
  14 │         4.3         3.0          1.1         0.1  setosa           t
rue
  15 │         5.8         4.0          1.2         0.2  setosa           t
rue
  16 │         5.7         4.4          1.5         0.4  setosa           t
rue
  17 │         5.4         3.9          1.3         0.4  setosa           t
rue
  18 │         5.1         3.5          1.4         0.3  setosa           t
rue
  19 │         5.7         3.8          1.7         0.3  setosa           t
rue
  20 │         5.1         3.8          1.5         0.3  setosa           t
rue
  21 │         5.4         3.4          1.7         0.2  setosa           t
rue
  22 │         5.1         3.7          1.5         0.4  setosa           t
rue
  23 │         4.6         3.6          1.0         0.2  setosa           t
rue
  24 │         5.1         3.3          1.7         0.5  setosa           t
rue
  25 │         4.8         3.4          1.9         0.2  setosa           t
rue
  26 │         5.0         3.0          1.6         0.2  setosa           t
rue
  27 │         5.0         3.4          1.6         0.4  setosa           t
rue
  28 │         5.2         3.5          1.5         0.2  setosa           t
rue
  29 │         5.2         3.4          1.4         0.2  setosa           t
rue
  30 │         4.7         3.2          1.6         0.2  setosa           t
rue
  31 │         4.8         3.1          1.6         0.2  setosa           t
rue
  32 │         5.4         3.4          1.5         0.4  setosa           t
rue
  33 │         5.2         4.1          1.5         0.1  setosa           t
rue
  34 │         5.5         4.2          1.4         0.2  setosa           t
rue
  35 │         4.9         3.1          1.5         0.2  setosa           t
rue
  36 │         5.0         3.2          1.2         0.2  setosa           t
rue
  37 │         5.5         3.5          1.3         0.2  setosa           t
rue
  38 │         4.9         3.6          1.4         0.1  setosa           t
rue
  39 │         4.4         3.0          1.3         0.2  setosa           t
rue
  40 │         5.1         3.4          1.5         0.2  setosa           t
rue
  41 │         5.0         3.5          1.3         0.3  setosa           t
rue
  42 │         4.5         2.3          1.3         0.3  setosa           t
rue
  43 │         4.4         3.2          1.3         0.2  setosa           t
rue
  44 │         5.0         3.5          1.6         0.6  setosa           t
rue
  45 │         5.1         3.8          1.9         0.4  setosa           t
rue
  46 │         4.8         3.0          1.4         0.3  setosa           t
rue
  47 │         5.1         3.8          1.6         0.2  setosa           t
rue
  48 │         4.6         3.2          1.4         0.2  setosa           t
rue
  49 │         5.3         3.7          1.5         0.2  setosa           t
rue
  50 │         5.0         3.3          1.4         0.2  setosa           t
rue
  51 │         7.0         3.2          4.7         1.4  versicolor      fa
lse
  52 │         6.4         3.2          4.5         1.5  versicolor      fa
lse
  53 │         6.9         3.1          4.9         1.5  versicolor      fa
lse
  54 │         5.5         2.3          4.0         1.3  versicolor      fa
lse
  55 │         6.5         2.8          4.6         1.5  versicolor      fa
lse
  56 │         5.7         2.8          4.5         1.3  versicolor      fa
lse
  57 │         6.3         3.3          4.7         1.6  versicolor      fa
lse
  58 │         4.9         2.4          3.3         1.0  versicolor      fa
lse
  59 │         6.6         2.9          4.6         1.3  versicolor      fa
lse
  60 │         5.2         2.7          3.9         1.4  versicolor      fa
lse
  61 │         5.0         2.0          3.5         1.0  versicolor      fa
lse
  62 │         5.9         3.0          4.2         1.5  versicolor      fa
lse
  63 │         6.0         2.2          4.0         1.0  versicolor      fa
lse
  64 │         6.1         2.9          4.7         1.4  versicolor      fa
lse
  65 │         5.6         2.9          3.6         1.3  versicolor      fa
lse
  66 │         6.7         3.1          4.4         1.4  versicolor      fa
lse
  67 │         5.6         3.0          4.5         1.5  versicolor      fa
lse
  68 │         5.8         2.7          4.1         1.0  versicolor      fa
lse
  69 │         6.2         2.2          4.5         1.5  versicolor      fa
lse
  70 │         5.6         2.5          3.9         1.1  versicolor      fa
lse
  71 │         5.9         3.2          4.8         1.8  versicolor      fa
lse
  72 │         6.1         2.8          4.0         1.3  versicolor      fa
lse
  73 │         6.3         2.5          4.9         1.5  versicolor      fa
lse
  74 │         6.1         2.8          4.7         1.2  versicolor      fa
lse
  75 │         6.4         2.9          4.3         1.3  versicolor      fa
lse
  76 │         6.6         3.0          4.4         1.4  versicolor      fa
lse
  77 │         6.8         2.8          4.8         1.4  versicolor      fa
lse
  78 │         6.7         3.0          5.0         1.7  versicolor      fa
lse
  79 │         6.0         2.9          4.5         1.5  versicolor      fa
lse
  80 │         5.7         2.6          3.5         1.0  versicolor      fa
lse
  81 │         5.5         2.4          3.8         1.1  versicolor      fa
lse
  82 │         5.5         2.4          3.7         1.0  versicolor      fa
lse
  83 │         5.8         2.7          3.9         1.2  versicolor      fa
lse
  84 │         6.0         2.7          5.1         1.6  versicolor      fa
lse
  85 │         5.4         3.0          4.5         1.5  versicolor      fa
lse
  86 │         6.0         3.4          4.5         1.6  versicolor      fa
lse
  87 │         6.7         3.1          4.7         1.5  versicolor      fa
lse
  88 │         6.3         2.3          4.4         1.3  versicolor      fa
lse
  89 │         5.6         3.0          4.1         1.3  versicolor      fa
lse
  90 │         5.5         2.5          4.0         1.3  versicolor      fa
lse
  91 │         5.5         2.6          4.4         1.2  versicolor      fa
lse
  92 │         6.1         3.0          4.6         1.4  versicolor      fa
lse
  93 │         5.8         2.6          4.0         1.2  versicolor      fa
lse
  94 │         5.0         2.3          3.3         1.0  versicolor      fa
lse
  95 │         5.6         2.7          4.2         1.3  versicolor      fa
lse
  96 │         5.7         3.0          4.2         1.2  versicolor      fa
lse
  97 │         5.7         2.9          4.2         1.3  versicolor      fa
lse
  98 │         6.2         2.9          4.3         1.3  versicolor      fa
lse
  99 │         5.1         2.5          3.0         1.1  versicolor      fa
lse
 100 │         5.7         2.8          4.1         1.3  versicolor      fa
lse
 101 │         6.3         3.3          6.0         2.5  virginica       fa
lse
 102 │         5.8         2.7          5.1         1.9  virginica       fa
lse
 103 │         7.1         3.0          5.9         2.1  virginica       fa
lse
 104 │         6.3         2.9          5.6         1.8  virginica       fa
lse
 105 │         6.5         3.0          5.8         2.2  virginica       fa
lse
 106 │         7.6         3.0          6.6         2.1  virginica       fa
lse
 107 │         4.9         2.5          4.5         1.7  virginica       fa
lse
 108 │         7.3         2.9          6.3         1.8  virginica       fa
lse
 109 │         6.7         2.5          5.8         1.8  virginica       fa
lse
 110 │         7.2         3.6          6.1         2.5  virginica       fa
lse
 111 │         6.5         3.2          5.1         2.0  virginica       fa
lse
 112 │         6.4         2.7          5.3         1.9  virginica       fa
lse
 113 │         6.8         3.0          5.5         2.1  virginica       fa
lse
 114 │         5.7         2.5          5.0         2.0  virginica       fa
lse
 115 │         5.8         2.8          5.1         2.4  virginica       fa
lse
 116 │         6.4         3.2          5.3         2.3  virginica       fa
lse
 117 │         6.5         3.0          5.5         1.8  virginica       fa
lse
 118 │         7.7         3.8          6.7         2.2  virginica       fa
lse
 119 │         7.7         2.6          6.9         2.3  virginica       fa
lse
 120 │         6.0         2.2          5.0         1.5  virginica       fa
lse
 121 │         6.9         3.2          5.7         2.3  virginica       fa
lse
 122 │         5.6         2.8          4.9         2.0  virginica       fa
lse
 123 │         7.7         2.8          6.7         2.0  virginica       fa
lse
 124 │         6.3         2.7          4.9         1.8  virginica       fa
lse
 125 │         6.7         3.3          5.7         2.1  virginica       fa
lse
 126 │         7.2         3.2          6.0         1.8  virginica       fa
lse
 127 │         6.2         2.8          4.8         1.8  virginica       fa
lse
 128 │         6.1         3.0          4.9         1.8  virginica       fa
lse
 129 │         6.4         2.8          5.6         2.1  virginica       fa
lse
 130 │         7.2         3.0          5.8         1.6  virginica       fa
lse
 131 │         7.4         2.8          6.1         1.9  virginica       fa
lse
 132 │         7.9         3.8          6.4         2.0  virginica       fa
lse
 133 │         6.4         2.8          5.6         2.2  virginica       fa
lse
 134 │         6.3         2.8          5.1         1.5  virginica       fa
lse
 135 │         6.1         2.6          5.6         1.4  virginica       fa
lse
 136 │         7.7         3.0          6.1         2.3  virginica       fa
lse
 137 │         6.3         3.4          5.6         2.4  virginica       fa
lse
 138 │         6.4         3.1          5.5         1.8  virginica       fa
lse
 139 │         6.0         3.0          4.8         1.8  virginica       fa
lse
 140 │         6.9         3.1          5.4         2.1  virginica       fa
lse
 141 │         6.7         3.1          5.6         2.4  virginica       fa
lse
 142 │         6.9         3.1          5.1         2.3  virginica       fa
lse
 143 │         5.8         2.7          5.1         1.9  virginica       fa
lse
 144 │         6.8         3.2          5.9         2.3  virginica       fa
lse
 145 │         6.7         3.3          5.7         2.5  virginica       fa
lse
 146 │         6.7         3.0          5.2         2.3  virginica       fa
lse
 147 │         6.3         2.5          5.0         1.9  virginica       fa
lse
 148 │         6.5         3.0          5.2         2.0  virginica       fa
lse
 149 │         6.2         3.4          5.4         2.3  virginica       fa
lse
 150 │         5.9         3.0          5.1         1.8  virginica       fa
lse
Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 133.33333333333334, lweight = 2.0, rweight = -2.0, further_split = true
))
node to split is next line

mehmehmeh
BitVector
Error: MethodError: no method matching getindex(::DataFrames.DataFrameColum
ns{DataFrames.DataFrame}, ::BitVector, ::Colon)
Closest candidates are:
  getindex(::DataFrames.DataFrameColumns, ::Union{Colon, Regex, AbstractVec
tor{T} where T, DataAPI.All, DataAPI.Between, DataAPI.Cols, InvertedIndices
.InvertedIndex}) at C:\Users\RTX2080\.julia\packages\DataFrames\JHf5N\src\a
bstractdataframe\iteration.jl:202
  getindex(::DataFrames.DataFrameColumns, !Matched::Union{AbstractString, S
igned, Symbol, Unsigned}) at C:\Users\RTX2080\.julia\packages\DataFrames\JH
f5N\src\abstractdataframe\iteration.jl:200
```





The returned model contains a vector of trees and the loss function and target

```julia
typeof(trees(xgtreemodel))
```

```
Error: UndefVarError: xgtreemodel not defined
```



```julia
typeof(xgtreemodel.loss)
```

```
Error: UndefVarError: xgtreemodel not defined
```



```julia
typeof(xgtreemodel.target)
```

```
Error: UndefVarError: xgtreemodel not defined
```





You can control parameters like  `max_depth` and `nrounds`
```julia
xgtreemodel2 = jlboost(iris, target; nrounds = 2, max_depth = 2)
```

```
1
150×6 DataFrameColumns
 Row │ SepalLength  SepalWidth  PetalLength  PetalWidth  Species     is_set
osa
     │ Float64      Float64     Float64      Float64     Cat…        Bool
─────┼─────────────────────────────────────────────────────────────────────
────
   1 │         5.1         3.5          1.4         0.2  setosa           t
rue
   2 │         4.9         3.0          1.4         0.2  setosa           t
rue
   3 │         4.7         3.2          1.3         0.2  setosa           t
rue
   4 │         4.6         3.1          1.5         0.2  setosa           t
rue
   5 │         5.0         3.6          1.4         0.2  setosa           t
rue
   6 │         5.4         3.9          1.7         0.4  setosa           t
rue
   7 │         4.6         3.4          1.4         0.3  setosa           t
rue
   8 │         5.0         3.4          1.5         0.2  setosa           t
rue
   9 │         4.4         2.9          1.4         0.2  setosa           t
rue
  10 │         4.9         3.1          1.5         0.1  setosa           t
rue
  11 │         5.4         3.7          1.5         0.2  setosa           t
rue
  12 │         4.8         3.4          1.6         0.2  setosa           t
rue
  13 │         4.8         3.0          1.4         0.1  setosa           t
rue
  14 │         4.3         3.0          1.1         0.1  setosa           t
rue
  15 │         5.8         4.0          1.2         0.2  setosa           t
rue
  16 │         5.7         4.4          1.5         0.4  setosa           t
rue
  17 │         5.4         3.9          1.3         0.4  setosa           t
rue
  18 │         5.1         3.5          1.4         0.3  setosa           t
rue
  19 │         5.7         3.8          1.7         0.3  setosa           t
rue
  20 │         5.1         3.8          1.5         0.3  setosa           t
rue
  21 │         5.4         3.4          1.7         0.2  setosa           t
rue
  22 │         5.1         3.7          1.5         0.4  setosa           t
rue
  23 │         4.6         3.6          1.0         0.2  setosa           t
rue
  24 │         5.1         3.3          1.7         0.5  setosa           t
rue
  25 │         4.8         3.4          1.9         0.2  setosa           t
rue
  26 │         5.0         3.0          1.6         0.2  setosa           t
rue
  27 │         5.0         3.4          1.6         0.4  setosa           t
rue
  28 │         5.2         3.5          1.5         0.2  setosa           t
rue
  29 │         5.2         3.4          1.4         0.2  setosa           t
rue
  30 │         4.7         3.2          1.6         0.2  setosa           t
rue
  31 │         4.8         3.1          1.6         0.2  setosa           t
rue
  32 │         5.4         3.4          1.5         0.4  setosa           t
rue
  33 │         5.2         4.1          1.5         0.1  setosa           t
rue
  34 │         5.5         4.2          1.4         0.2  setosa           t
rue
  35 │         4.9         3.1          1.5         0.2  setosa           t
rue
  36 │         5.0         3.2          1.2         0.2  setosa           t
rue
  37 │         5.5         3.5          1.3         0.2  setosa           t
rue
  38 │         4.9         3.6          1.4         0.1  setosa           t
rue
  39 │         4.4         3.0          1.3         0.2  setosa           t
rue
  40 │         5.1         3.4          1.5         0.2  setosa           t
rue
  41 │         5.0         3.5          1.3         0.3  setosa           t
rue
  42 │         4.5         2.3          1.3         0.3  setosa           t
rue
  43 │         4.4         3.2          1.3         0.2  setosa           t
rue
  44 │         5.0         3.5          1.6         0.6  setosa           t
rue
  45 │         5.1         3.8          1.9         0.4  setosa           t
rue
  46 │         4.8         3.0          1.4         0.3  setosa           t
rue
  47 │         5.1         3.8          1.6         0.2  setosa           t
rue
  48 │         4.6         3.2          1.4         0.2  setosa           t
rue
  49 │         5.3         3.7          1.5         0.2  setosa           t
rue
  50 │         5.0         3.3          1.4         0.2  setosa           t
rue
  51 │         7.0         3.2          4.7         1.4  versicolor      fa
lse
  52 │         6.4         3.2          4.5         1.5  versicolor      fa
lse
  53 │         6.9         3.1          4.9         1.5  versicolor      fa
lse
  54 │         5.5         2.3          4.0         1.3  versicolor      fa
lse
  55 │         6.5         2.8          4.6         1.5  versicolor      fa
lse
  56 │         5.7         2.8          4.5         1.3  versicolor      fa
lse
  57 │         6.3         3.3          4.7         1.6  versicolor      fa
lse
  58 │         4.9         2.4          3.3         1.0  versicolor      fa
lse
  59 │         6.6         2.9          4.6         1.3  versicolor      fa
lse
  60 │         5.2         2.7          3.9         1.4  versicolor      fa
lse
  61 │         5.0         2.0          3.5         1.0  versicolor      fa
lse
  62 │         5.9         3.0          4.2         1.5  versicolor      fa
lse
  63 │         6.0         2.2          4.0         1.0  versicolor      fa
lse
  64 │         6.1         2.9          4.7         1.4  versicolor      fa
lse
  65 │         5.6         2.9          3.6         1.3  versicolor      fa
lse
  66 │         6.7         3.1          4.4         1.4  versicolor      fa
lse
  67 │         5.6         3.0          4.5         1.5  versicolor      fa
lse
  68 │         5.8         2.7          4.1         1.0  versicolor      fa
lse
  69 │         6.2         2.2          4.5         1.5  versicolor      fa
lse
  70 │         5.6         2.5          3.9         1.1  versicolor      fa
lse
  71 │         5.9         3.2          4.8         1.8  versicolor      fa
lse
  72 │         6.1         2.8          4.0         1.3  versicolor      fa
lse
  73 │         6.3         2.5          4.9         1.5  versicolor      fa
lse
  74 │         6.1         2.8          4.7         1.2  versicolor      fa
lse
  75 │         6.4         2.9          4.3         1.3  versicolor      fa
lse
  76 │         6.6         3.0          4.4         1.4  versicolor      fa
lse
  77 │         6.8         2.8          4.8         1.4  versicolor      fa
lse
  78 │         6.7         3.0          5.0         1.7  versicolor      fa
lse
  79 │         6.0         2.9          4.5         1.5  versicolor      fa
lse
  80 │         5.7         2.6          3.5         1.0  versicolor      fa
lse
  81 │         5.5         2.4          3.8         1.1  versicolor      fa
lse
  82 │         5.5         2.4          3.7         1.0  versicolor      fa
lse
  83 │         5.8         2.7          3.9         1.2  versicolor      fa
lse
  84 │         6.0         2.7          5.1         1.6  versicolor      fa
lse
  85 │         5.4         3.0          4.5         1.5  versicolor      fa
lse
  86 │         6.0         3.4          4.5         1.6  versicolor      fa
lse
  87 │         6.7         3.1          4.7         1.5  versicolor      fa
lse
  88 │         6.3         2.3          4.4         1.3  versicolor      fa
lse
  89 │         5.6         3.0          4.1         1.3  versicolor      fa
lse
  90 │         5.5         2.5          4.0         1.3  versicolor      fa
lse
  91 │         5.5         2.6          4.4         1.2  versicolor      fa
lse
  92 │         6.1         3.0          4.6         1.4  versicolor      fa
lse
  93 │         5.8         2.6          4.0         1.2  versicolor      fa
lse
  94 │         5.0         2.3          3.3         1.0  versicolor      fa
lse
  95 │         5.6         2.7          4.2         1.3  versicolor      fa
lse
  96 │         5.7         3.0          4.2         1.2  versicolor      fa
lse
  97 │         5.7         2.9          4.2         1.3  versicolor      fa
lse
  98 │         6.2         2.9          4.3         1.3  versicolor      fa
lse
  99 │         5.1         2.5          3.0         1.1  versicolor      fa
lse
 100 │         5.7         2.8          4.1         1.3  versicolor      fa
lse
 101 │         6.3         3.3          6.0         2.5  virginica       fa
lse
 102 │         5.8         2.7          5.1         1.9  virginica       fa
lse
 103 │         7.1         3.0          5.9         2.1  virginica       fa
lse
 104 │         6.3         2.9          5.6         1.8  virginica       fa
lse
 105 │         6.5         3.0          5.8         2.2  virginica       fa
lse
 106 │         7.6         3.0          6.6         2.1  virginica       fa
lse
 107 │         4.9         2.5          4.5         1.7  virginica       fa
lse
 108 │         7.3         2.9          6.3         1.8  virginica       fa
lse
 109 │         6.7         2.5          5.8         1.8  virginica       fa
lse
 110 │         7.2         3.6          6.1         2.5  virginica       fa
lse
 111 │         6.5         3.2          5.1         2.0  virginica       fa
lse
 112 │         6.4         2.7          5.3         1.9  virginica       fa
lse
 113 │         6.8         3.0          5.5         2.1  virginica       fa
lse
 114 │         5.7         2.5          5.0         2.0  virginica       fa
lse
 115 │         5.8         2.8          5.1         2.4  virginica       fa
lse
 116 │         6.4         3.2          5.3         2.3  virginica       fa
lse
 117 │         6.5         3.0          5.5         1.8  virginica       fa
lse
 118 │         7.7         3.8          6.7         2.2  virginica       fa
lse
 119 │         7.7         2.6          6.9         2.3  virginica       fa
lse
 120 │         6.0         2.2          5.0         1.5  virginica       fa
lse
 121 │         6.9         3.2          5.7         2.3  virginica       fa
lse
 122 │         5.6         2.8          4.9         2.0  virginica       fa
lse
 123 │         7.7         2.8          6.7         2.0  virginica       fa
lse
 124 │         6.3         2.7          4.9         1.8  virginica       fa
lse
 125 │         6.7         3.3          5.7         2.1  virginica       fa
lse
 126 │         7.2         3.2          6.0         1.8  virginica       fa
lse
 127 │         6.2         2.8          4.8         1.8  virginica       fa
lse
 128 │         6.1         3.0          4.9         1.8  virginica       fa
lse
 129 │         6.4         2.8          5.6         2.1  virginica       fa
lse
 130 │         7.2         3.0          5.8         1.6  virginica       fa
lse
 131 │         7.4         2.8          6.1         1.9  virginica       fa
lse
 132 │         7.9         3.8          6.4         2.0  virginica       fa
lse
 133 │         6.4         2.8          5.6         2.2  virginica       fa
lse
 134 │         6.3         2.8          5.1         1.5  virginica       fa
lse
 135 │         6.1         2.6          5.6         1.4  virginica       fa
lse
 136 │         7.7         3.0          6.1         2.3  virginica       fa
lse
 137 │         6.3         3.4          5.6         2.4  virginica       fa
lse
 138 │         6.4         3.1          5.5         1.8  virginica       fa
lse
 139 │         6.0         3.0          4.8         1.8  virginica       fa
lse
 140 │         6.9         3.1          5.4         2.1  virginica       fa
lse
 141 │         6.7         3.1          5.6         2.4  virginica       fa
lse
 142 │         6.9         3.1          5.1         2.3  virginica       fa
lse
 143 │         5.8         2.7          5.1         1.9  virginica       fa
lse
 144 │         6.8         3.2          5.9         2.3  virginica       fa
lse
 145 │         6.7         3.3          5.7         2.5  virginica       fa
lse
 146 │         6.7         3.0          5.2         2.3  virginica       fa
lse
 147 │         6.3         2.5          5.0         1.9  virginica       fa
lse
 148 │         6.5         3.0          5.2         2.0  virginica       fa
lse
 149 │         6.2         3.4          5.4         2.3  virginica       fa
lse
 150 │         5.9         3.0          5.1         1.8  virginica       fa
lse
Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 133.33333333333334, lweight = 2.0, rweight = -2.0, further_split = true
))
node to split is next line

mehmehmeh
BitVector
Error: MethodError: no method matching getindex(::DataFrames.DataFrameColum
ns{DataFrames.DataFrame}, ::BitVector, ::Colon)
Closest candidates are:
  getindex(::DataFrames.DataFrameColumns, ::Union{Colon, Regex, AbstractVec
tor{T} where T, DataAPI.All, DataAPI.Between, DataAPI.Cols, InvertedIndices
.InvertedIndex}) at C:\Users\RTX2080\.julia\packages\DataFrames\JHf5N\src\a
bstractdataframe\iteration.jl:202
  getindex(::DataFrames.DataFrameColumns, !Matched::Union{AbstractString, S
igned, Symbol, Unsigned}) at C:\Users\RTX2080\.julia\packages\DataFrames\JH
f5N\src\abstractdataframe\iteration.jl:200
```





To grow the tree a leaf-wise (AKA best-first or or in XGBoost terminology "lossguided") strategy,
you see set the `max_leaves` parameters e.g.
```julia
xgtreemodel3 = jlboost(iris, target; nrounds = 2, max_leaves = 8, max_depth = 0)
```

```
1
150×6 DataFrameColumns
 Row │ SepalLength  SepalWidth  PetalLength  PetalWidth  Species     is_set
osa
     │ Float64      Float64     Float64      Float64     Cat…        Bool
─────┼─────────────────────────────────────────────────────────────────────
────
   1 │         5.1         3.5          1.4         0.2  setosa           t
rue
   2 │         4.9         3.0          1.4         0.2  setosa           t
rue
   3 │         4.7         3.2          1.3         0.2  setosa           t
rue
   4 │         4.6         3.1          1.5         0.2  setosa           t
rue
   5 │         5.0         3.6          1.4         0.2  setosa           t
rue
   6 │         5.4         3.9          1.7         0.4  setosa           t
rue
   7 │         4.6         3.4          1.4         0.3  setosa           t
rue
   8 │         5.0         3.4          1.5         0.2  setosa           t
rue
   9 │         4.4         2.9          1.4         0.2  setosa           t
rue
  10 │         4.9         3.1          1.5         0.1  setosa           t
rue
  11 │         5.4         3.7          1.5         0.2  setosa           t
rue
  12 │         4.8         3.4          1.6         0.2  setosa           t
rue
  13 │         4.8         3.0          1.4         0.1  setosa           t
rue
  14 │         4.3         3.0          1.1         0.1  setosa           t
rue
  15 │         5.8         4.0          1.2         0.2  setosa           t
rue
  16 │         5.7         4.4          1.5         0.4  setosa           t
rue
  17 │         5.4         3.9          1.3         0.4  setosa           t
rue
  18 │         5.1         3.5          1.4         0.3  setosa           t
rue
  19 │         5.7         3.8          1.7         0.3  setosa           t
rue
  20 │         5.1         3.8          1.5         0.3  setosa           t
rue
  21 │         5.4         3.4          1.7         0.2  setosa           t
rue
  22 │         5.1         3.7          1.5         0.4  setosa           t
rue
  23 │         4.6         3.6          1.0         0.2  setosa           t
rue
  24 │         5.1         3.3          1.7         0.5  setosa           t
rue
  25 │         4.8         3.4          1.9         0.2  setosa           t
rue
  26 │         5.0         3.0          1.6         0.2  setosa           t
rue
  27 │         5.0         3.4          1.6         0.4  setosa           t
rue
  28 │         5.2         3.5          1.5         0.2  setosa           t
rue
  29 │         5.2         3.4          1.4         0.2  setosa           t
rue
  30 │         4.7         3.2          1.6         0.2  setosa           t
rue
  31 │         4.8         3.1          1.6         0.2  setosa           t
rue
  32 │         5.4         3.4          1.5         0.4  setosa           t
rue
  33 │         5.2         4.1          1.5         0.1  setosa           t
rue
  34 │         5.5         4.2          1.4         0.2  setosa           t
rue
  35 │         4.9         3.1          1.5         0.2  setosa           t
rue
  36 │         5.0         3.2          1.2         0.2  setosa           t
rue
  37 │         5.5         3.5          1.3         0.2  setosa           t
rue
  38 │         4.9         3.6          1.4         0.1  setosa           t
rue
  39 │         4.4         3.0          1.3         0.2  setosa           t
rue
  40 │         5.1         3.4          1.5         0.2  setosa           t
rue
  41 │         5.0         3.5          1.3         0.3  setosa           t
rue
  42 │         4.5         2.3          1.3         0.3  setosa           t
rue
  43 │         4.4         3.2          1.3         0.2  setosa           t
rue
  44 │         5.0         3.5          1.6         0.6  setosa           t
rue
  45 │         5.1         3.8          1.9         0.4  setosa           t
rue
  46 │         4.8         3.0          1.4         0.3  setosa           t
rue
  47 │         5.1         3.8          1.6         0.2  setosa           t
rue
  48 │         4.6         3.2          1.4         0.2  setosa           t
rue
  49 │         5.3         3.7          1.5         0.2  setosa           t
rue
  50 │         5.0         3.3          1.4         0.2  setosa           t
rue
  51 │         7.0         3.2          4.7         1.4  versicolor      fa
lse
  52 │         6.4         3.2          4.5         1.5  versicolor      fa
lse
  53 │         6.9         3.1          4.9         1.5  versicolor      fa
lse
  54 │         5.5         2.3          4.0         1.3  versicolor      fa
lse
  55 │         6.5         2.8          4.6         1.5  versicolor      fa
lse
  56 │         5.7         2.8          4.5         1.3  versicolor      fa
lse
  57 │         6.3         3.3          4.7         1.6  versicolor      fa
lse
  58 │         4.9         2.4          3.3         1.0  versicolor      fa
lse
  59 │         6.6         2.9          4.6         1.3  versicolor      fa
lse
  60 │         5.2         2.7          3.9         1.4  versicolor      fa
lse
  61 │         5.0         2.0          3.5         1.0  versicolor      fa
lse
  62 │         5.9         3.0          4.2         1.5  versicolor      fa
lse
  63 │         6.0         2.2          4.0         1.0  versicolor      fa
lse
  64 │         6.1         2.9          4.7         1.4  versicolor      fa
lse
  65 │         5.6         2.9          3.6         1.3  versicolor      fa
lse
  66 │         6.7         3.1          4.4         1.4  versicolor      fa
lse
  67 │         5.6         3.0          4.5         1.5  versicolor      fa
lse
  68 │         5.8         2.7          4.1         1.0  versicolor      fa
lse
  69 │         6.2         2.2          4.5         1.5  versicolor      fa
lse
  70 │         5.6         2.5          3.9         1.1  versicolor      fa
lse
  71 │         5.9         3.2          4.8         1.8  versicolor      fa
lse
  72 │         6.1         2.8          4.0         1.3  versicolor      fa
lse
  73 │         6.3         2.5          4.9         1.5  versicolor      fa
lse
  74 │         6.1         2.8          4.7         1.2  versicolor      fa
lse
  75 │         6.4         2.9          4.3         1.3  versicolor      fa
lse
  76 │         6.6         3.0          4.4         1.4  versicolor      fa
lse
  77 │         6.8         2.8          4.8         1.4  versicolor      fa
lse
  78 │         6.7         3.0          5.0         1.7  versicolor      fa
lse
  79 │         6.0         2.9          4.5         1.5  versicolor      fa
lse
  80 │         5.7         2.6          3.5         1.0  versicolor      fa
lse
  81 │         5.5         2.4          3.8         1.1  versicolor      fa
lse
  82 │         5.5         2.4          3.7         1.0  versicolor      fa
lse
  83 │         5.8         2.7          3.9         1.2  versicolor      fa
lse
  84 │         6.0         2.7          5.1         1.6  versicolor      fa
lse
  85 │         5.4         3.0          4.5         1.5  versicolor      fa
lse
  86 │         6.0         3.4          4.5         1.6  versicolor      fa
lse
  87 │         6.7         3.1          4.7         1.5  versicolor      fa
lse
  88 │         6.3         2.3          4.4         1.3  versicolor      fa
lse
  89 │         5.6         3.0          4.1         1.3  versicolor      fa
lse
  90 │         5.5         2.5          4.0         1.3  versicolor      fa
lse
  91 │         5.5         2.6          4.4         1.2  versicolor      fa
lse
  92 │         6.1         3.0          4.6         1.4  versicolor      fa
lse
  93 │         5.8         2.6          4.0         1.2  versicolor      fa
lse
  94 │         5.0         2.3          3.3         1.0  versicolor      fa
lse
  95 │         5.6         2.7          4.2         1.3  versicolor      fa
lse
  96 │         5.7         3.0          4.2         1.2  versicolor      fa
lse
  97 │         5.7         2.9          4.2         1.3  versicolor      fa
lse
  98 │         6.2         2.9          4.3         1.3  versicolor      fa
lse
  99 │         5.1         2.5          3.0         1.1  versicolor      fa
lse
 100 │         5.7         2.8          4.1         1.3  versicolor      fa
lse
 101 │         6.3         3.3          6.0         2.5  virginica       fa
lse
 102 │         5.8         2.7          5.1         1.9  virginica       fa
lse
 103 │         7.1         3.0          5.9         2.1  virginica       fa
lse
 104 │         6.3         2.9          5.6         1.8  virginica       fa
lse
 105 │         6.5         3.0          5.8         2.2  virginica       fa
lse
 106 │         7.6         3.0          6.6         2.1  virginica       fa
lse
 107 │         4.9         2.5          4.5         1.7  virginica       fa
lse
 108 │         7.3         2.9          6.3         1.8  virginica       fa
lse
 109 │         6.7         2.5          5.8         1.8  virginica       fa
lse
 110 │         7.2         3.6          6.1         2.5  virginica       fa
lse
 111 │         6.5         3.2          5.1         2.0  virginica       fa
lse
 112 │         6.4         2.7          5.3         1.9  virginica       fa
lse
 113 │         6.8         3.0          5.5         2.1  virginica       fa
lse
 114 │         5.7         2.5          5.0         2.0  virginica       fa
lse
 115 │         5.8         2.8          5.1         2.4  virginica       fa
lse
 116 │         6.4         3.2          5.3         2.3  virginica       fa
lse
 117 │         6.5         3.0          5.5         1.8  virginica       fa
lse
 118 │         7.7         3.8          6.7         2.2  virginica       fa
lse
 119 │         7.7         2.6          6.9         2.3  virginica       fa
lse
 120 │         6.0         2.2          5.0         1.5  virginica       fa
lse
 121 │         6.9         3.2          5.7         2.3  virginica       fa
lse
 122 │         5.6         2.8          4.9         2.0  virginica       fa
lse
 123 │         7.7         2.8          6.7         2.0  virginica       fa
lse
 124 │         6.3         2.7          4.9         1.8  virginica       fa
lse
 125 │         6.7         3.3          5.7         2.1  virginica       fa
lse
 126 │         7.2         3.2          6.0         1.8  virginica       fa
lse
 127 │         6.2         2.8          4.8         1.8  virginica       fa
lse
 128 │         6.1         3.0          4.9         1.8  virginica       fa
lse
 129 │         6.4         2.8          5.6         2.1  virginica       fa
lse
 130 │         7.2         3.0          5.8         1.6  virginica       fa
lse
 131 │         7.4         2.8          6.1         1.9  virginica       fa
lse
 132 │         7.9         3.8          6.4         2.0  virginica       fa
lse
 133 │         6.4         2.8          5.6         2.2  virginica       fa
lse
 134 │         6.3         2.8          5.1         1.5  virginica       fa
lse
 135 │         6.1         2.6          5.6         1.4  virginica       fa
lse
 136 │         7.7         3.0          6.1         2.3  virginica       fa
lse
 137 │         6.3         3.4          5.6         2.4  virginica       fa
lse
 138 │         6.4         3.1          5.5         1.8  virginica       fa
lse
 139 │         6.0         3.0          4.8         1.8  virginica       fa
lse
 140 │         6.9         3.1          5.4         2.1  virginica       fa
lse
 141 │         6.7         3.1          5.6         2.4  virginica       fa
lse
 142 │         6.9         3.1          5.1         2.3  virginica       fa
lse
 143 │         5.8         2.7          5.1         1.9  virginica       fa
lse
 144 │         6.8         3.2          5.9         2.3  virginica       fa
lse
 145 │         6.7         3.3          5.7         2.5  virginica       fa
lse
 146 │         6.7         3.0          5.2         2.3  virginica       fa
lse
 147 │         6.3         2.5          5.0         1.9  virginica       fa
lse
 148 │         6.5         3.0          5.2         2.0  virginica       fa
lse
 149 │         6.2         3.4          5.4         2.3  virginica       fa
lse
 150 │         5.9         3.0          5.1         1.8  virginica       fa
lse
Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 133.33333333333334, lweight = 2.0, rweight = -2.0, further_split = true
))
node to split is next line

mehmehmeh
BitVector
Error: MethodError: no method matching getindex(::DataFrames.DataFrameColum
ns{DataFrames.DataFrame}, ::BitVector, ::Colon)
Closest candidates are:
  getindex(::DataFrames.DataFrameColumns, ::Union{Colon, Regex, AbstractVec
tor{T} where T, DataAPI.All, DataAPI.Between, DataAPI.Cols, InvertedIndices
.InvertedIndex}) at C:\Users\RTX2080\.julia\packages\DataFrames\JHf5N\src\a
bstractdataframe\iteration.jl:202
  getindex(::DataFrames.DataFrameColumns, !Matched::Union{AbstractString, S
igned, Symbol, Unsigned}) at C:\Users\RTX2080\.julia\packages\DataFrames\JH
f5N\src\abstractdataframe\iteration.jl:200
```




it recommended that you set `max_depth = 0` to avoid a warning message.


Convenience `predict` function is provided. It can be used to score a tree or a vector of trees
```julia
iris.pred1 = JLBoost.predict(xgtreemodel, iris)
iris.pred2 = JLBoost.predict(xgtreemodel2, iris)
iris.pred1_plus_2 = JLBoost.predict(vcat(xgtreemodel, xgtreemodel2), iris)
```

```
Error: UndefVarError: xgtreemodel not defined
```





There are also convenience functions for computing the AUC and gini
```julia
AUC(-iris.pred1, iris.is_setosa)
```

```
Error: ArgumentError: column name :pred1 not found in the data frame
```



```julia
gini(-iris.pred1, iris.is_setosa)
```

```
Error: ArgumentError: column name :pred1 not found in the data frame
```





As a convenience feature, you can adjust the `eta` weight of each tree by multiplying it by a factor e.g.

```Julia
new_tree = 0.3 * trees(xgtreemodel)[1] # weight the first tree by 30%
unique(predict(new_tree, iris) ./ predict(trees(xgtreemodel)[1], iris)) # 0.3
```

#### Feature Importances
One can obtain the feature importance using the `feature_importance` function

```julia
feature_importance(xgtreemodel2, iris)
```

```
Error: UndefVarError: xgtreemodel2 not defined
```





#### Tables.jl integration

Any Tables.jl compatible tabular data structure. So you can use any column accessible table with JLBoost. However, you are advised to define the following methods for `df` as the generic implementation in this package may not be efficient

```julia
nrow(df) # returns the number of rows
ncol(df)
view(df, rows, cols)
```



#### Regression Model
By default `JLBoost.jl` defines it's own `LogitLogLoss` type for  binary classification problems. You may replace the `loss` function-type from the `LossFunctions.jl` `SupervisedLoss` type. E.g for regression models you can choose the leaast squares loss called `L2DistLoss()`

```julia
using DataFrames
using JLBoost
df = DataFrame(x = rand(100) * 100)

df[!, :y] = 2*df.x .+ rand(100)

target = :y
features = [:x]
warm_start = fill(0.0, nrow(df))


using LossFunctions: L2DistLoss
loss = L2DistLoss()
jlboost(df, target, features, warm_start, loss; max_depth=2) # default max_depth = 6
```

```
1
100×2 DataFrameColumns
 Row │ x         y
     │ Float64   Float64
─────┼─────────────────────
   1 │  6.85285   13.7456
   2 │ 58.4736   117.751
   3 │ 78.1222   157.212
   4 │  5.313     11.4048
   5 │  6.5093    13.5425
   6 │ 30.5202    61.6858
   7 │ 97.733    196.213
   8 │ 59.8801   120.025
   9 │ 98.6856   198.205
  10 │ 98.0035   196.887
  11 │  5.23864   10.4779
  12 │ 52.7524   105.625
  13 │ 43.5943    87.8167
  14 │ 35.6612    71.9878
  15 │ 80.0319   160.766
  16 │  4.29095    9.25427
  17 │ 31.0801    62.162
  18 │ 20.1991    40.7497
  19 │ 14.6456    29.4518
  20 │ 32.9749    66.4856
  21 │ 17.46      34.9817
  22 │ 27.9741    56.7754
  23 │ 70.5972   141.849
  24 │ 13.5698    27.429
  25 │ 42.7098    86.0755
  26 │ 72.4424   145.652
  27 │ 74.2599   149.153
  28 │ 35.5619    71.788
  29 │ 36.6619    73.7889
  30 │ 35.2491    71.2513
  31 │ 61.2408   122.885
  32 │ 29.2044    59.0248
  33 │ 34.0556    68.9212
  34 │ 67.9795   136.311
  35 │ 57.8503   115.747
  36 │ 57.1886   114.784
  37 │ 42.8773    86.177
  38 │ 20.512     41.6774
  39 │ 59.7256   119.626
  40 │ 56.2437   112.516
  41 │ 12.7583    25.9961
  42 │ 48.9057    98.4096
  43 │ 81.7244   163.648
  44 │ 94.6588   189.769
  45 │ 20.7686    41.7114
  46 │ 58.5752   117.77
  47 │ 84.5021   169.399
  48 │ 22.2379    44.8093
  49 │ 66.8399   134.406
  50 │ 95.2502   191.181
  51 │ 97.7647   195.796
  52 │ 74.5925   149.99
  53 │ 77.6182   156.06
  54 │ 66.9385   134.774
  55 │  8.99588   18.4246
  56 │ 55.3546   110.743
  57 │ 58.239    117.444
  58 │ 85.4734   171.142
  59 │ 85.6433   171.792
  60 │ 30.5399    61.5666
  61 │ 59.5027   119.334
  62 │ 76.6337   154.098
  63 │ 42.6207    85.269
  64 │ 80.1069   160.754
  65 │ 57.3384   115.209
  66 │ 56.1667   113.15
  67 │ 53.2913   107.259
  68 │ 39.6533    79.3947
  69 │ 68.5851   137.408
  70 │  2.4856     5.22658
  71 │ 75.5626   151.998
  72 │ 82.432    165.532
  73 │ 64.6138   129.438
  74 │ 58.3849   117.159
  75 │  6.56619   13.8999
  76 │ 27.2176    55.1358
  77 │ 58.6813   118.121
  78 │ 11.3445    23.5343
  79 │ 78.3592   156.72
  80 │ 20.3562    40.9388
  81 │ 45.0145    90.465
  82 │ 34.4795    69.7004
  83 │ 92.4534   185.378
  84 │ 54.8518   109.923
  85 │ 62.4688   125.081
  86 │ 37.18      74.4932
  87 │  1.61299    3.23898
  88 │ 95.5286   192.015
  89 │ 34.8021    70.4117
  90 │ 62.8602   126.112
  91 │ 82.1461   165.073
  92 │ 63.6436   127.749
  93 │ 58.2726   117.074
  94 │  2.76223    6.39029
  95 │ 77.6951   156.263
  96 │ 97.4002   194.901
  97 │ 68.3115   136.628
  98 │ 47.3792    95.5124
  99 │ 83.4818   167.059
 100 │ 79.522    159.067
Dict{Any, Any}( => (feature = :x, split_at = 47.37921668095946, cutpt = 42,
 gain = 445051.37349451706, lweight = 49.113659086080446, rweight = 144.690
2273878559, further_split = true))
node to split is next line

mehmehmeh
BitVector
Error: MethodError: no method matching getindex(::DataFrames.DataFrameColum
ns{DataFrames.DataFrame}, ::BitVector, ::Colon)
Closest candidates are:
  getindex(::DataFrames.DataFrameColumns, ::Union{Colon, Regex, AbstractVec
tor{T} where T, DataAPI.All, DataAPI.Between, DataAPI.Cols, InvertedIndices
.InvertedIndex}) at C:\Users\RTX2080\.julia\packages\DataFrames\JHf5N\src\a
bstractdataframe\iteration.jl:202
  getindex(::DataFrames.DataFrameColumns, !Matched::Union{AbstractString, S
igned, Symbol, Unsigned}) at C:\Users\RTX2080\.julia\packages\DataFrames\JH
f5N\src\abstractdataframe\iteration.jl:200
```





### Save & Load models
You save the models using the `JLBoost.save` and load it with the `load` function

```julia
JLBoost.save(xgtreemodel, "model.jlb")
JLBoost.save(trees(xgtreemodel), "model_tree.jlb")
```

```
Error: UndefVarError: xgtreemodel not defined
```



```julia
JLBoost.load("model.jlb")
JLBoost.load("model_tree.jlb")
```

```
Tree 1
eta = 1.0 (tree weight)

   -- PetalLength <= 1.9
   -- PetalLength > 1.9
```





### Fit model on `JDF.JDFFile` - enabling larger-than-RAM model fit
Sometimes, you may want to fit a model on a dataset that is too large to fit into RAM. You can convert the dataset to [JDF format](https://github.com/xiaodaigh/JDF.jl) and then use `JDF.JDFFile` functionalities to fit the models. The interface `jlbosst` for `DataFrame` and `JDFFiLe` are the same.

The key advantage of fitting a model using `JDF.JDFFile` is that not all the data need to be loaded into memory. This is because `JDF` can load the columns one at a time. Hence this will enable larger models to be trained on a single computer.

```julia
using JLBoost, RDatasets, JDF
iris = dataset("datasets", "iris")

iris[!, :is_setosa] = iris[!, :Species] .== "setosa"
target = :is_setosa

features = setdiff(names(iris), [:Species, :is_setosa])

savejdf("iris.jdf", iris)
irisdisk = JDFFile("iris.jdf")

# fit using on disk JDF format
xgtree1 = jlboost(irisdisk, target, features)
xgtree2 = jlboost(iris, target, features; nrounds = 2, max_depth = 2)

# predict using on disk JDF format
iris.pred1 = predict(xgtree1, irisdisk)
iris.pred2 = predict(xgtree2, irisdisk)

# AUC
AUC(-predict(xgtree1, irisdisk), irisdisk[!, :is_setosa])

# gini
gini(-predict(xgtree1, irisdisk), irisdisk[!, :is_setosa])

# clean up
rm("iris.jdf", force=true, recursive=true)
```

```
1
JDF.JDFFile{String}("iris.jdf")
Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 133.33333333333334, lweight = 2.0, rweight = -2.0, further_split = true
))
node to split is next line

mehmehmeh
BitVector
Error: MethodError: no method matching getindex(::JDF.JDFFile{String}, ::Bi
tVector, ::Colon)
Closest candidates are:
  getindex(::JDF.JDFFile, !Matched::Symbol) at C:\Users\RTX2080\.julia\pack
ages\JDF\TKMdl\src\JDFFile.jl:69
  getindex(::JDF.JDFFile, !Matched::String) at C:\Users\RTX2080\.julia\pack
ages\JDF\TKMdl\src\JDFFile.jl:65
```





### MLJ.jl

Integration with MLJ.jl is available via the [JLBoostMLJ.jl](https://github.com/xiaodaigh/JLBoostMLJ.jl) package

### Hackable

## Notes

Currently has a CPU implementation of the `xgboost` binary boosting algorithm as described in the original paper. I am trying to implement the algorithms in the original `xgboost` paper. I want to implement the algorithms mentioned in LigthGBM and Catboost and to port them to GPUs.

There are two similar projects

* [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl)
* [JuML.jl](https://github.com/Statfactory/JuML.jl)
