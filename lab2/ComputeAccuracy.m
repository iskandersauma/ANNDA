function acc = ComputeAccuracy(X,y,W1,b1,W2,b2,k)
[~,N] = size(X);
p = EvaluateClassifier(X,W1,b1,W2,b2,k);
[~,t] = max(p);
acc = sum((t-1)'==y)/N;
end