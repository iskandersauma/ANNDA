function j = ComputeCost(X,Y,W,b,lambda,k)
h = intervalues(X,W,b,k);
p = EvaluateClassifier(h,W,b);
j1 = sum(diag(-log(Y'*p)))/size(X,2);
j2 = lambda*sumsqr(W);
j = j1 + j2;
end