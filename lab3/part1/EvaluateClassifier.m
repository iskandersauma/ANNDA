function p = EvaluateClassifier(h,W,b)
W = W{end};
b = b{end};
X = h{end};
b = repmat(b,1,size(X,2));

s = W*X + b;
denom = repmat(sum(exp(s),1), size(s,1),1);
p = exp(s)./denom;
end