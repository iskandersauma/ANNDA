function j = ComputeCost(X,Y,W,b,lambda)
W1 = W{1};
W2 = W{2};

h = intervalues(X,W,b);
p = EvaluateClassifier(h,W,b);
j1 = sum(diag(-log(Y'*p)))/size(X,2);
j2 = lambda*(sumsqr(W1) + sumsqr(W2));
j = j1 + j2;
end