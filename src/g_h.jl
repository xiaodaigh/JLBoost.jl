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

g(loss::SupervisedLoss, y, prev_w) = begin
	deriv(loss, y, 1/(1 + exp(-prev_w)))
end

h(loss::SupervisedLoss, y, prev_w) = begin
	deriv2(loss, y, 1/(1 + exp(-prev_w)))
end


# begin: Zygote.jl
# g(loss::Function, y, prev_w) = begin
# 	gres = gradient(x->loss(x, y), prev_w)
# 	gres[1]
# end

# h(loss::Function, y, prev_w) = begin
#     hres = hessian(x->loss(x[1], y), [prev_w])
#     hres[1]
# end
# end: Zygote.jl

# begin: ForwardDiff.jl
# g(loss::Function, y, prev_w) = begin
#     gres = ForwardDiff.gradient(x->loss(x[1], y), [prev_w])
#     gres[1]
# end

# h(loss::Function, y, prev_w) = begin
#     hres = ForwardDiff.hessian(x->loss(x[1], y), [prev_w])
#     hres[1]
# end
# end: ForwardDiff.jl