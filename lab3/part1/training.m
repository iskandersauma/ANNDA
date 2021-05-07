function [W,b,jtrain, jtest] = training(Xtr,Ytr,Xval,Yval,GDparams, lambda,k,m)
d = size(Xtr,1);
K_ = size(Ytr,1);
M = [d,m,K_];
std = 0.001;
[W,b] = init_parameters(M,std);

jtrain = zeros(1,GDparams.epochs);
jtest = zeros(1,GDparams.epochs);
decay = 0.95;
alpha = 0.99;
rho = 0.9;

for i = 1:GDparams.epochs
   [W,b,mu,v] = MiniBatchGD(Xtr,Ytr,W,b,lambda,GDparams,rho,alpha,k);
   jtrain(i) = ComputeCost(Xtr,Ytr,W,b,lambda,k);
   jtest(i) = ComputeCost(Xval,Yval,W,b,lambda,k);
   if jtrain(i) > 3*jtrain(1)
      break;
   end
   GDparams.eta = decay*GDparams.eta;
end
end