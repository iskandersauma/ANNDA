function acc = ComputeAccuracy(X,y,W,b)
[~,N] = size(X);
p = EvaluateClassifier(X,W,b);
[~,i] = max(p);
acc = sum((i-1)'==y)/N;
end