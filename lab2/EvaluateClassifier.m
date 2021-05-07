function [P,H,s] = EvaluateClassifier(X,W1,b1, W2, b2,k)
s = bsxfun(@plus,W1*X,b1);
%H=max(0,s);
H = 1./(1+exp(-s));
f = bsxfun(@plus, W2*H,b2);
P = bsxfun(@rdivide,exp(f),sum(exp(f),1));
end