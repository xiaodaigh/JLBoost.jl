export LogitLogLoss, value, deriv, deriv2

# set up loss functions
# The Flux implemnetation
# logloss = logitbinarycrossentropy

# alternate definition
# softmax(w) = 1/(1 + exp(-w))
# logloss(w, y) = -(y*log(softmax(w)) + (1-y)*log(1-softmax(w)))


#x,y = rand(1000), rand([0.0, 1.10], 1000)
#Flux.logitbinarycrossentropy.(x,y) ≈ logloss.(x, y)

# function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
#   return -sum(y .* logsoftmax(logŷ) .* weight) * 1 // size(y, 2)
# end

import LossFunctions:  SupervisedLoss, deriv, value, deriv2
struct LogitLogLoss <: SupervisedLoss end

logit(w) = 1/(1 + exp(-w))

value(::LogitLogLoss, y::Number, w::Number) = -(y*log(logit(w)) + (1-y)*log(1-logit(w)))

# https://www.wolframalpha.com/input/?i=f%28w%29+%3D+-%28y*log%281%2F%281+%2B+exp%28-w%29%29%29+%2B+%281-y%29*log%281-1%2F%281+%2B+exp%28-w%29%29%29%29%2C+df%2Fdw
deriv(::LogitLogLoss, y::Number, w::Number) = -logit(-w) - y + 1

# https://www.wolframalpha.com/input/?i=f%28w%29+%3D+-%28e%5Ew+%28-1+%2B+y%29+%2B+y%29%2F%281+%2B+e%5Ew%29%2C+df%2Fdw
deriv2(::LogitLogLoss, y::Number, w::Number) = exp(w)*(logit(-w)^2)

g(loss::SupervisedLoss, y, warmstart) = begin
	deriv(loss, y, warmstart)
end

h(loss::SupervisedLoss, y, warmstart) = begin
	deriv2(loss, y, warmstart)
end


# begin: Zygote.jl
# g(loss::Function, y, warmstart) = begin
# 	gres = gradient(x->loss(x, y), warmstart)
# 	gres[1]
# end

# h(loss::Function, y, warmstart) = begin
#     hres = hessian(x->loss(x[1], y), [warmstart])
#     hres[1]
# end
# end: Zygote.jl

# begin: ForwardDiff.jl
# g(loss::Function, y, warmstart) = begin
#     gres = ForwardDiff.gradient(x->loss(x[1], y), [warmstart])
#     gres[1]
# end

# h(loss::Function, y, warmstart) = begin
#     hres = ForwardDiff.hessian(x->loss(x[1], y), [warmstart])
#     hres[1]
# end
# end: ForwardDiff.jl
