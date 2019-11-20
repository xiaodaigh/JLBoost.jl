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

g(loss::SupervisedLoss, y, warmstart) = begin
	deriv(loss, y, 1/(1 + exp(-warmstart)))
end

h(loss::SupervisedLoss, y, warmstart) = begin
	deriv2(loss, y, 1/(1 + exp(-warmstart)))
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
