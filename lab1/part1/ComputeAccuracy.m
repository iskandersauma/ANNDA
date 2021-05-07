function acc = ComputeAccuracy(X,y,W,b)
p = EvaluateClassifier(X,W,b);
[~,index] = max(p);
acc = length(find(y-index == 0))/length(y);
end