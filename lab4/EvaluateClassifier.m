function p = EvaluateClassifier(h,W,b)
W = W{end};
b = b{end};
X = h{end};
b = repmat(b,1,size(X,2));

s = W*X + b;
p = exp(s)./sum(exp(s));
end