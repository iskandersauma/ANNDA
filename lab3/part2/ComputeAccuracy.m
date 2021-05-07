function [acc,index] = ComputeAccuracy(X,y,W,b,k)
h = intervalues(X,W,b,k);
p = EvaluateClassifier(h,W,b);
[~,index] = max(p);
acc = length(find(y-index == 0))/length(y);
index = oneHot(index);
end