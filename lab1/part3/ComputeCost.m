function j = ComputeCost(X,Y,W,b,lambda)
p = EvaluateClassifier(X,W,b);
j1 = sum(diag(-log(Y'*p)))/size(X,2);
j2 = lambda*sum(sum(W.^2));
j = j1 + j2;
end