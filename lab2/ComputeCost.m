function j = ComputeCost(X,Y,W1,b1,W2,b2,lambda,k)
[~,N] = size(X);
p = EvaluateClassifier(X,W1,b1,W2,b2,k);
j = -sum(log(sum(Y.*p,1)))/N + lambda*(sumsqr(W1)+sumsqr(W2));
end