function [W,b,jtrain,jtest] = boosting(Xtr,Ytr,Xval,Yval,GDparams,lambda, W, b)
k = size(Ytr,1);
d = size(Xtr,1);

jtrain = zeros(1,GDparams.epochs);
jtest = zeros(1,GDparams.epochs);
for i = 1:GDparams.epochs
   jtrain(i) = ComputeCost(Xtr,Ytr,W,b,lambda);
   jtest(i) = ComputeCost(Xval,Yval,W,b,lambda);
   [W,b] = MiniBatchGD(Xtr,Ytr,W,b,lambda,GDparams);
end

end