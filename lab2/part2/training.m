function [W,b,jtrain, jtest, flag] = training(Xtr,Ytr,Xval,Yval,GDparams, lambda,m)
d = size(Xtr,1);
k = size(Ytr,1);
std = 0.001;
[W,b] = init_parameters(m,d,k,std);

jtrain = zeros(1,GDparams.epochs);
jtest = zeros(1,GDparams.epochs);
decay = 0.95;
rho = 0.9;
flag = 0;

for i = 1:GDparams.epochs
   jtrain(i) = ComputeCost(Xtr,Ytr,W,b,lambda);
   jtest(i) = ComputeCost(Xval,Yval,W,b,lambda);
   if jtrain(i) > 3*jtrain(1)
      flag = 1;
      break;
   end
   [W,b] = MiniBatchGD(Xtr,Ytr,W,b,lambda,GDparams,rho);
   GDparams.eta = decay*GDparams.eta;
end
end