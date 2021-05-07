function [acc,p] = ComputeAccuracy(X,y,W)
p = EvaluateClassifier(X,W);
[~,index] = max(p);
acc = length(find(y-index == 0))/length(y);
end