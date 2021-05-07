function j = ComputeCost(X,Y,W,b,lambda)
j = 0;
[~,N] = size(X);
p = EvaluateClassifier(X,W,b);
j = -sum(log(sum(Y.*p,1)))/N+lambda*sumsqr(W);
end