function [Wstar, bstar] = MiniBatchGD(X,Y,W,b,lambda,k,d,eta)
p = EvaluateClassifier(X,W,b);
[grad_W,grad_b] = ComputeGradient(X,Y,p,W,lambda,k,d);
Wstar = W - eta*grad_W;
bstar = b - eta*grad_b;
end