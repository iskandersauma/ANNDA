function [RNN,M] = backward(RNN, X, Y, a, h, p, n, m, eta, M)
grad = ComputeGradient(RNN, X, Y, a, h, p, n, m);
eps = 1e-8;

for f = fieldnames(RNN)'
    M.(f{1}) = M.(f{1}) + grad.(f{1}).^2;
    RNN.(f{1}) = RNN.(f{1}) - eta*(grad.(f{1})./(M.(f{1}) + eps).^(0.5));
end

end