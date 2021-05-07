function grad = ComputeGradient(RNN, X, Y, a, h, p, n, m)
W = RNN.W;
V = RNN.V;
gh = zeros(n,m);
ga = zeros(n,m);
g = -(Y - p)';
grad.c = (sum(g))';
grad.V = g'*h(:, 2 : end)';

gh(n,:) = g(n,:)*V;
ga(n,:) = gh(n,:)*diag(1 - tanh(a(:,n)).^2);
for i = n-1: -1 : 1
   gh(i,:) = g(i,:)*V + ga(i+1,:)*W; 
   ga(i,:) = gh(i,:)*diag(1 - tanh(a(:,i)).^2);
end
grad.b = (sum(ga))';
grad.W = ga'*h(:,1:end-1)';
grad.U = ga'*X';
end