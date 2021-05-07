function j = ComputeCost(X,Y,W,lambda,delta)
p = EvaluateClassifier(X,W);
pc = repmat(sum(p.*Y),size(p,1),1);
margin = p - pc + delta;
j1 = sum(margin(find(margin > 0))) - size(p,2)*delta;
j2 = lambda*sumsqr(W);
j = j1 + j2;
end