function p = EvaluateClassifier(X,W,b)
b = repmat(b,1,size(X,2));
s = W*X + b;
denom = repmat(sum(exp(s),1), size(W,1),1);
p = exp(s)./denom;
end