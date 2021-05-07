function p = EvaluateClassifier(x,W,b)
W2 = cell2mat(W(2));
b2 = cell2mat(b(2));
h = max(0,x);
s = W2*h + b2;
p = exp(s)./sum(exp(s));
end