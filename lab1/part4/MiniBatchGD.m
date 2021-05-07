function Wstar = MiniBatchGD(X,Y,W,lambda,GDparams,delta)
batch_size = GDparams.batch_size;
eta = GDparams.eta;
N = size(X,2);

for i = 1:N/batch_size
   jstart = (i-1)*batch_size+1;
   jend = i*batch_size;
   index = jstart:jend;
   Xbatch = X(:,index);
   Ybatch = Y(:,index);
   Xbatch = Xbatch + 0.05*randn(size(Xbatch));
   
   p = EvaluateClassifier(Xbatch,W);
   grad_W = ComputeGradient(Xbatch,Ybatch,p,W,lambda,delta);
   W = W - eta*grad_W;
end
Wstar = W;
end