function loss = ComputeLoss(X,Y, RNN, h)
W = RNN.W;
U = RNN.U;
V = RNN.V;
b = RNN.b;
c = RNN.c;
n = size(X,2);
loss = 0;
for i = 1:n
   val = W*h + U*X(:,i) + b;
   h = tanh(val);
   o = V*h + c;
   p = exp(o);
   p = p/sum(p);
   loss = loss - log(Y(:,i)'*p);
end
end