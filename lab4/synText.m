function y = synText(RNN, hin, xin, n, k)
W = RNN.W;
U = RNN.U;
V = RNN.V;
b = RNN.b;
c = RNN.c;
h = hin;
x = xin;
y = zeros(1,n);

for i = 1:n
   a = W*h+ U*x + b;
   h = tanh(a);
   o = V*h + c;
   p = exp(o);
   p = p/sum(p);
   
   val = cumsum(p);
   a = rand;
   index = find(val - a > 0);
   ind = index(1);
   
   x = oneHot(ind,k);
   y(i) = ind;
end
end
