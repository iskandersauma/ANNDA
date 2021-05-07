function p = EvaluateClassifier(X,W,b)
Y = W*X + b;
p = exp(Y)./ sum(exp(Y),1);
end