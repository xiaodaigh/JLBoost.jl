using Flux: logitbinarycrossentropy
using Zygote: @adjoint
using ForwardDiff: derivative
using CuArrays
CuArrays.allowscalar(false)

prevw = rand(150000);
target = rand([0, 1.], length(prevw));

@adjoint logitbinarycrossentropy(w, t) = logitbinarycrossentropy(w, t), delta -> (delta*(1/(1+exp(-w)) - t), delta)

g(prevw, target) = Zygote.gradient(prevw->logitbinarycrossentropy(prevw, target), prevw)[1]
g2(prevw, target) = Zygote.gradient(logitbinarycrossentropy, prevw, target)[1]
g3(prevw, target) = ForwardDiff.gradient(x->logitbinarycrossentropy(x[1], target), [prevw])[1]
g4(prevw, target) = ForwardDiff.derivative(prevw->logitbinarycrossentropy(prevw, target), prevw)
g5(prevw, target) = Zygote.gradient(prevw) do prevw
	Zygote.forwarddiff(prew->logitbinarycrossentropy(prew, target), prevw)
end[1]
g6(prevw, target) = deriv(LogitProbLoss(), target, 1 / (1 + exp(-prevw)))
easy(prevw, target) = 1 / (1 + exp(-prevw)) - target


gprev = gpu(prevw)
gtarget = gpu(target)

g6.(gprev, gtarget)


res = logitbinarycrossentropy.(prevw, target)  

value(LogitProbLoss(), target, 1 ./ (1 .+ exp.(-prevw)))

deriv(LogitProbLoss(), target, prevw)

deriv2(LogitProbLoss(), target, prevw)


using LossFunctions

value(LogitProbLoss(), target, prevw)



@benchmark g.($prevw, $target)
@benchmark g2.($prevw, $target)
@benchmark g3.($prevw, $target)
@benchmark g4.($prevw, $target)
@benchmark g5.($prevw, $target)
@benchmark easy.($prevw, $target)
@benchmark g6.($prevw, $target)