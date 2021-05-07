function [loss, a, h, o, p] = forward(RNN, X, Y, hin, n, k, m)
W = RNN.W;
U = RNN.U;
V = RNN.V;
b = RNN.b;
c = RNN.c;
ht = hin;
o = zeros(k,n);
p = zeros(k,n);
h = zeros(m,n);
a = zeros(m,n);
loss = 0;

for i = 1:n
    at = W*ht + U*X(:,i) + b;
    a(:,i) = at;
    ht = tanh(at);
    h(:,i) = ht;
    o(:,i) = V*ht + c;
    pval = exp(o(:,i));
    p(:,i) = pval/sum(pval);
    loss = loss - log(Y(:,i)'*p(:,i));
end
h = [hin, h];
end