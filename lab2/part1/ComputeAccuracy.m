function [acc,index] = ComputeAccuracy(X,y,W,b)
h = intervalues(X,W,b);
p = EvaluateClassifier(h,W,b);
[~,index] = max(p);
acc = length(find(y-index == 0))/length(y);
index = oneHot(index);
end