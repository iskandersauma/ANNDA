function [Wstar, bstar] = MiniBatchGD(X,Y,W,b,lambda,GDparams)
batch_size = GDparams.batch_size;
eta = GDparams.eta;
N = size(X,2);

for i = 1:N/batch_size
   jstart = (i-1)*batch_size+1;
   jend = i*batch_size;
   index = jstart:jend;
   Xbatch = X(:,index);
   Ybatch = Y(:,index);
   p = EvaluateClassifier(Xbatch,W,b);
   [grad_W,grad_b] = ComputeGradient(Xbatch,Ybatch,p,W,lambda);
   W = W - eta*grad_W;
   b = b - eta*grad_b;
end
Wstar = W;
bstar = b;
end